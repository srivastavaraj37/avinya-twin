"""Receding-horizon MPC-style controller.

Every hour, for the ventilation AND fan decision jointly, the controller
rolls a small library of candidate 12-hour (vent, fan) trajectory *pairs*
forward through a scratch copy of the polyhouse physics, using the actual
future weather over the horizon as a perfect short-term forecast (this
project has no separate weather-forecasting model, and none was asked for
-- see module docstring note below). Each candidate is scored by a cost
that trades off:

    1. VPD-band tracking, against the *achievable* VPD band only. Diagnosed
       2026-08-11 (see CLAUDE.md): during JJA monsoon, ventilation has close
       to zero authority over VPD for ~83% of night hours (ambient RH_out is
       already 85-95%, so opening vents exchanges saturated indoor air for
       near-saturated outdoor air). Penalizing the controller for missing a
       fixed 0.8-1.2 kPa target during those hours punishes it for a target
       it structurally cannot reach and produces a meaningless gradient. So
       at each horizon hour we run two cheap boundary rollouts (vent_frac
       pinned at 0.0 and at 1.0, fan off) to get the *achievable* VPD range
       at that hour, clip the fixed target band into that range, and score
       against the clipped band. An hour with zero achievable range costs
       zero VPD penalty regardless of the outcome -- there was nothing to be
       done about it.
    2. Leaf-wetness-hours, weighted explicitly (not as a DSV proxy -- see
       leaf_wet_weight below). This is deliberately the most heavily-weighted
       term: it's the metric the fan actuator gives the controller real
       authority over (unlike VPD/vent at night), so it is what a retuned
       objective should actually be optimizing.
    3. A hard temperature-safety penalty on predicted T_in, unchanged from
       the original version (VPD/leaf-wet tracking alone has no reason to
       avoid overheating -- a hot, dry hour can score *well* on VPD).
    4. An explicit actuation-switching penalty, w_switch * |action -
       previous action|, applied to both vent and fan across the horizon
       (including the transition from the real plant's last-applied action
       into the candidate's first hour). Added because the original
       objective had no term discouraging chatter and lost to Threshold on
       vent_actuations (330 vs. 180) in the original 90-day comparison.
    5. A small fan energy cost, fan_energy_weight * fan_frac per hour, so
       the fan is used when it earns its keep on leaf-wetness, not by
       default.

The candidate pair with the lowest total horizon cost has its *first* hour's
(vent_frac, fan_frac) applied; the rest of the sequence is discarded and the
whole search repeats next hour with the newly observed state (receding
horizon). vent_policy() and fan_policy() are called separately by
sim.engine.run (vent, then irrigation, then fan) but must agree on the same
joint decision for a given timestamp -- vent_policy() runs the full search
and caches the joint result; fan_policy() reads the cached fan value for the
same timestamp (falling back to a fresh search if called out of the expected
order, e.g. in isolation from a test).

Irrigation is decided separately (not part of the vent/fan search) by a
simple soil-depletion + rain-skip rule: skip if it's raining now or forecast
to rain within the next few hours (the water is coming anyway), otherwise
irrigate when the soil bucket is stressed.

Forecast note: using the dataset's own future rows as a "forecast" is a
perfect-foresight simplification appropriate for a controller *comparison*
study on a fixed historical record. A production controller would replace
this with an actual weather forecast API; nothing else about the controller
would need to change (it only reads T_out/RH_out/I_solar/wind/rain from
whatever DataFrame rows it's handed).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from sim.config import default_config
from sim.polyhouse import Polyhouse, PolyhouseParams
from sim.psychro import rh_from_chi
from sim.soil import SoilBucket

VentCandidate = Callable[[int, "pd.Series[Any]", Polyhouse], float]
FanCandidate = Callable[[int, "pd.Series[Any]", Polyhouse], float]


def _closed(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 0.0


def _low(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 0.2


def _medium(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 0.5


def _high(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 0.8


def _full(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 1.0


def _day_open_night_closed(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 0.8 if row["I_solar"] > 0 else 0.1


def _vent_reactive(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    # Evaluated against the *rolled-out* (predicted) state, not the real
    # plant -- this is what makes it a legitimate candidate trajectory
    # rather than a precomputed constant.
    return 1.0 if ph.t_in > 32.0 else 0.2


DEFAULT_VENT_CANDIDATES: dict[str, VentCandidate] = {
    "closed": _closed,
    "low": _low,
    "medium": _medium,
    "high": _high,
    "full": _full,
    "day_open_night_closed": _day_open_night_closed,
    "reactive": _vent_reactive,
}


def _fan_off(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 0.0


def _fan_on(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    return 1.0


def _fan_reactive(i: int, row: "pd.Series[Any]", ph: Polyhouse) -> float:
    rh_in = rh_from_chi(ph.t_in, ph.chi)
    return 1.0 if rh_in > 92.0 else 0.0


DEFAULT_FAN_CANDIDATES: dict[str, FanCandidate] = {
    "off": _fan_off,
    "on": _fan_on,
    "reactive": _fan_reactive,
}


@dataclass
class MPCController:
    """Receding-horizon candidate-search controller over (vent, fan).

    Args:
        weather_df: The full weather record the run will use. Needed so the
            controller can look ahead of the current timestamp for its
            horizon rollout and its rain-skip check.
        horizon_hours: Planning horizon, hours.
        vpd_low_kpa / vpd_high_kpa: Bounds of the VPD tracking band, kPa
            (clipped per-hour into the achievable range before scoring).
        leaf_wet_weight: Cost weight per predicted leaf-wet hour in the
            horizon. The literal Wallin DSV lookup (sim/disease.py) is a
            nonsmooth, period-and-duration-based integer table -- not a
            per-hour cost that makes sense to roll into a receding-horizon
            score. Predicted wet-hour count is used instead as a monotonic,
            smooth proxy, and is weighted explicitly (not just a DSV proxy)
            because it's the quantity the fan actuator gives the controller
            real authority over -- see CLAUDE.md's Problem-1 finding.
        t_safety_c / t_safety_penalty_per_degree: Soft ceiling on predicted
            T_in and the per-degree penalty for crossing it.
        w_switch: Actuation-switching penalty weight, applied to both vent
            and fan, |action - previous_action| per hour (including the
            transition from the real plant's last committed action).
        fan_energy_weight: Cost weight per fan_frac-hour in the horizon,
            so the fan is used only when its leaf-wet benefit outweighs it.
        irrigation_dose_mm: Depth applied when triggered, mm.
        rain_lookahead_hours: How far ahead to check for incoming rain
            before deciding to irrigate.
        rain_skip_mm: Any hour (now or in the lookahead window) with rain
            at or above this depth cancels irrigation for this decision.
        config: Physics config used both for the real plant's parameters
            (via sim.engine) and for the scratch rollout copies here. Must
            match whatever config the engine run uses, or the controller's
            predictions won't correspond to the actual plant.
        vent_candidates / fan_candidates: Named libraries of candidate
            vent/fan trajectories; the search tries every (vent, fan) pair.
        vpd_weight: Multiplier on the (achievable-clipped) VPD quadratic
            penalty. Exposed as an explicit weight (rather than folded into
            the "1.0" implicit in the cost expression) so it can be swept
            alongside the other objective weights -- see
            scripts/sweep_mpc_weights.py.
    """

    weather_df: pd.DataFrame
    horizon_hours: int = 12
    vpd_low_kpa: float = 0.8
    vpd_high_kpa: float = 1.2
    # Defaults below are the weight set chosen by scripts/sweep_mpc_weights.py's
    # 36-combo grid on one year of Regime A -- see CLAUDE.md's "MPC objective
    # retune" section for the full Pareto table and the justification. Used as
    # the class default (not just a script-local override) so every caller
    # (dashboard, scripts/experiment.py, tests) gets the same chosen weights
    # unless it explicitly asks for something else, per "pick one set, use it
    # everywhere."
    vpd_weight: float = 2.0
    leaf_wet_weight: float = 0.04
    t_safety_c: float = 42.0
    t_safety_penalty_per_degree: float = 5.0
    w_switch: float = 0.15
    fan_energy_weight: float = 0.0
    irrigation_dose_mm: float = 5.0
    rain_lookahead_hours: int = 3
    rain_skip_mm: float = 1.0
    config: dict[str, Any] = field(default_factory=default_config)
    vent_candidates: dict[str, VentCandidate] = field(default_factory=lambda: dict(DEFAULT_VENT_CANDIDATES))
    fan_candidates: dict[str, FanCandidate] = field(default_factory=lambda: dict(DEFAULT_FAN_CANDIDATES))

    def __post_init__(self) -> None:
        self._params = PolyhouseParams.from_config(self.config)
        self._index = self.weather_df.index
        self._last_vent: float = 0.3
        self._last_fan: float = 0.0
        self._cache_ts: pd.Timestamp | None = None
        self._cache_action: tuple[float, float] = (0.3, 0.0)

    def _horizon_rows(self, ts: pd.Timestamp) -> pd.DataFrame:
        loc = self._index.get_loc(ts)
        return self.weather_df.iloc[loc : loc + self.horizon_hours]

    def _boundary_vpd_trajectory(
        self, horizon_rows: pd.DataFrame, t_in0: float, rh_in0: float, vent_frac: float
    ) -> list[float]:
        """VPD at each horizon hour under a fixed vent_frac (0.0 or 1.0), fan off.

        Cheap (no candidate search, just one straight-line rollout); used to
        bound the achievable VPD range per hour, independent of which real
        candidate is being scored.
        """
        ph = Polyhouse(config=self._params, t_in_init_c=t_in0, rh_in_init_pct=rh_in0)
        vpds: list[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _, row in horizon_rows.iterrows():
                out = ph.step(
                    t_out_c=float(row["T_out"]),
                    rh_out_pct=float(row["RH_out"]),
                    i_solar_wm2=float(row["I_solar"]),
                    wind_ms=float(row["wind"]),
                    vent_frac=vent_frac,
                    fan_frac=0.0,
                    dt_seconds=3600.0,
                )
                vpds.append(float(out["VPD"]))
        return vpds

    def _rollout_cost(
        self,
        vent_fn: VentCandidate,
        fan_fn: FanCandidate,
        horizon_rows: pd.DataFrame,
        t_in0: float,
        rh_in0: float,
        vpd_achievable_lo: list[float],
        vpd_achievable_hi: list[float],
    ) -> tuple[float, tuple[float, float]]:
        """Simulate one (vent, fan) candidate pair over the horizon.

        Uses a single 3600s (1-hour) step per horizon hour rather than the
        engine's usual 300s sub-stepping: the polyhouse's exponential
        (integrating-factor) integration is unconditionally stable at any
        dt (see sim/polyhouse.py), so a coarser rollout step is exact for
        this purpose and ~12x cheaper -- important since this rollout runs
        len(vent_candidates)*len(fan_candidates) times every simulated hour.

        Returns:
            (total_cost, (first_hour_vent_frac, first_hour_fan_frac)).
        """
        ph = Polyhouse(config=self._params, t_in_init_c=t_in0, rh_in_init_pct=rh_in0)
        cost = 0.0
        first_action: tuple[float, float] | None = None
        prev_vent, prev_fan = self._last_vent, self._last_fan
        with warnings.catch_warnings():
            # The coarse 1hr rollout step routinely trips Polyhouse's >20%
            # per-step change warning; that check exists to catch numerical
            # instability, and the exponential integrator is unconditionally
            # stable regardless of dt (see sim/polyhouse.py), so this is
            # expected noise from a deliberately coarse planning step, not a
            # sign of a problem.
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, (_, row) in enumerate(horizon_rows.iterrows()):
                vent_frac = vent_fn(i, row, ph)
                fan_frac = fan_fn(i, row, ph)
                if first_action is None:
                    first_action = (vent_frac, fan_frac)
                out = ph.step(
                    t_out_c=float(row["T_out"]),
                    rh_out_pct=float(row["RH_out"]),
                    i_solar_wm2=float(row["I_solar"]),
                    wind_ms=float(row["wind"]),
                    vent_frac=vent_frac,
                    fan_frac=fan_frac,
                    dt_seconds=3600.0,
                )

                # VPD term: target clipped into this hour's achievable range,
                # so an hour with no vent authority costs nothing.
                lo, hi = vpd_achievable_lo[i], vpd_achievable_hi[i]
                if lo > hi:
                    lo, hi = hi, lo
                eff_low = min(max(self.vpd_low_kpa, lo), hi)
                eff_high = min(max(self.vpd_high_kpa, lo), hi)
                vpd = out["VPD"]
                cost += self.vpd_weight * (max(0.0, vpd - eff_high) ** 2 + max(0.0, eff_low - vpd) ** 2)

                if out["leaf_wet"]:
                    cost += self.leaf_wet_weight

                if out["T_in"] > self.t_safety_c:
                    cost += self.t_safety_penalty_per_degree * (out["T_in"] - self.t_safety_c)

                cost += self.w_switch * (abs(vent_frac - prev_vent) + abs(fan_frac - prev_fan))
                cost += self.fan_energy_weight * fan_frac

                prev_vent, prev_fan = vent_frac, fan_frac

        return cost, (first_action if first_action is not None else (0.3, 0.0))

    def _decide(self, state: dict[str, Any], ts: pd.Timestamp) -> tuple[float, float]:
        if self._cache_ts == ts:
            return self._cache_action

        horizon_rows = self._horizon_rows(ts)
        if len(horizon_rows) == 0:
            action = (0.3, 0.0)  # off the end of the dataset; hold a safe middling default
            self._cache_ts, self._cache_action = ts, action
            return action

        t_in0 = state["T_in"]
        rh_in0 = state["RH_in"] if state["RH_in"] is not None else 70.0

        vpd_lo = self._boundary_vpd_trajectory(horizon_rows, t_in0, rh_in0, vent_frac=0.0)
        vpd_hi = self._boundary_vpd_trajectory(horizon_rows, t_in0, rh_in0, vent_frac=1.0)

        best_cost = float("inf")
        best_action = (0.3, 0.0)
        for vent_fn in self.vent_candidates.values():
            for fan_fn in self.fan_candidates.values():
                cost, action = self._rollout_cost(vent_fn, fan_fn, horizon_rows, t_in0, rh_in0, vpd_lo, vpd_hi)
                if cost < best_cost:
                    best_cost = cost
                    best_action = action

        vent_frac = min(max(best_action[0], 0.0), 1.0)
        fan_frac = min(max(best_action[1], 0.0), 1.0)
        action = (vent_frac, fan_frac)
        self._last_vent, self._last_fan = action
        self._cache_ts, self._cache_action = ts, action
        return action

    def vent_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        return self._decide(state, ts)[0]

    def fan_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        return self._decide(state, ts)[1]

    def irrigation_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        lookahead = self._horizon_rows(ts).iloc[: self.rain_lookahead_hours]
        rain_incoming = bool((lookahead["rain"] >= self.rain_skip_mm).any())
        if rain_incoming:
            return 0.0
        if soil.stressed:
            return self.irrigation_dose_mm
        return 0.0
