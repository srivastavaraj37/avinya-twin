"""Single-zone lumped-parameter polyhouse energy and vapour balance.

State variables: T_in (degC, bulk internal air temperature) and chi
(kg water vapour / m3 moist air, bulk internal absolute humidity).
Integrated at a fixed sub-hourly timestep (dt = 300 s).

Both balances are of the linear relaxation form dy/dt = S - k*y, with S and
k evaluated once at the start of each sub-step (holding weather, vent_frac
and the E, U, m_dot coefficients "frozen" over the sub-step) -- this is the
explicit-Euler assumption. Within that frozen-coefficient sub-step, however,
the ODE is solved *exactly* (y(t) = y_eq + (y0-y_eq)*exp(-k*t)) rather than
by a first-order Euler difference. This "exponential Euler" / integrating-
factor scheme is algebraically no harder than plain Euler, is still fully
explicit (no iteration, no matrix solve), and is unconditionally stable --
which plain forward Euler is not here: the vapour-exchange rate constant
k_chi = ACH/3600 reaches ~2-8e-3 /s at high ventilation, and dt*k_chi > 2
(the forward-Euler stability limit for a linear decay) is routinely crossed
at dt=300s and ACH>=24/hr, which forward Euler turns into a diverging
oscillation in chi (verified: RH_in was observed swinging between 0% and
100% every step under sustained high ventilation before this fix). The
temperature relaxation rate k_T stays comfortably below the same limit
given this project's thermal mass, but is solved the same way for
consistency and headroom. This is the same technique used by established
greenhouse climate models (e.g. Vanthoor 2011 "GreenLight") for the same
reason.

Energy balance (W, all terms evaluated at the start of the step):
    C_eff * dT/dt = Q_solar + Q_conduction + Q_ventilation - Q_latent

    Q_solar   = tau * I_solar * A_floor
    Q_cond    = U * A_cover * (T_out - T_in)
    Q_vent    = m_dot * cp_air * (T_out - T_in)
    Q_latent  = lambda_vap * E

Vapour balance (kg/m3/s):
    V * dchi/dt = E - Q_vol * (chi_in - chi_out)

Transpiration (Stanghellini-type two-term model):
    E = A_floor * (coeff_radiation * I_in / lambda_vap
                    + coeff_vpd * LAI * max(VPD_in, 0))

Circulation fan (HAF) actuator, ``fan_frac`` in [0, 1]:
    Unlike the vent, the fan does not exchange air with outside -- it moves
    bulk indoor air past the leaf surface. Physically this thins the laminar
    boundary layer at the leaf, which raises the leaf-surface (not bulk-air)
    vapour transfer coefficient (Stanghellini 1987 Ch. 3; Monteith & Unsworth,
    "Principles of Environmental Physics", boundary-layer resistance falls
    roughly as the inverse square root of air velocity past a flat leaf).
    A thinner boundary layer suppresses condensation on the leaf at a given
    *bulk* RH_in, because the leaf's own microclimate stays drier than the
    surrounding air. We do not model the boundary-layer resistance itself as
    a state variable (that needs a leaf-energy-balance sub-model this project
    doesn't otherwise carry); instead we model its net effect directly as a
    rise in the bulk-RH threshold at which condensation/wetness is judged to
    occur: leaf_wet = RH_in > rh_wet_threshold(fan_frac), where the threshold
    rises linearly from 90% (no fan; the value used everywhere else in this
    project) to 96% at fan_frac=1 (config: fan.rh_wet_threshold_min/_max).
    This 90->96 mapping is a *modelling assumption*, not a measured value --
    see CLAUDE.md's fan section for the sensitivity of downstream results to
    it. The fan also adds a small sensible heat load to the energy balance
    (its motor dissipates as heat into the same air volume) and is tracked
    as an explicit energy cost (fan_kWh) rather than treated as free.

References:
    - Stanghellini, C. (1987). "Transpiration of greenhouse crops."
      PhD thesis, Wageningen -- basis for the two-term (radiation + VPD)
      transpiration model used here, and for the boundary-layer mechanism
      the fan actuator acts on.
    - Monteith, J.L. & Unsworth, M.H., "Principles of Environmental
      Physics" -- leaf boundary-layer resistance vs. air velocity.
    - ASHRAE Fundamentals Handbook -- cp_air, rho_air.
    - Allen et al. (1998), FAO-56 -- lambda_vap.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from sim.psychro import abs_humidity, chi_sat, rh_from_chi, vpd_kpa

# --- Physical constants ------------------------------------------------------

CP_AIR = 1006.0  # J/(kg.K), ASHRAE Fundamentals: specific heat of dry air at constant pressure
RHO_AIR = 1.2  # kg/m3, ASHRAE Fundamentals: standard air density near 20 degC, sea level
LAMBDA_VAP = 2.45e6  # J/kg, FAO-56 (Allen et al. 1998): latent heat of vaporisation of water (~20-25 degC)


@dataclass
class PolyhouseParams:
    """Physical parameters for the single-zone polyhouse model (all from config.yaml)."""

    area_floor_m2: float
    area_cover_m2: float
    height_m: float
    cover_tau: float
    cover_u_wm2k: float
    c_eff_jk_per_m2: float
    ach_min: float
    ach_max_base: float
    wind_coeff: float
    lai: float
    coeff_radiation: float
    coeff_vpd: float
    dt_seconds: float
    stability_warn_frac: float
    fan_power_w: float = 250.0
    rh_wet_threshold_min: float = 90.0
    rh_wet_threshold_max: float = 96.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PolyhouseParams":
        ph = cfg["polyhouse"]
        vent = cfg["ventilation"]
        transp = cfg["transpiration"]
        sim = cfg["simulation"]
        fan = cfg.get("fan", {})
        return cls(
            area_floor_m2=float(ph["area_floor_m2"]),
            area_cover_m2=float(ph["area_cover_m2"]),
            height_m=float(ph["height_m"]),
            cover_tau=float(ph["cover_tau"]),
            cover_u_wm2k=float(ph["cover_U_Wm2K"]),
            c_eff_jk_per_m2=float(ph["C_eff_JK_per_m2"]),
            ach_min=float(vent["ach_min"]),
            ach_max_base=float(vent["ach_max_base"]),
            wind_coeff=float(vent["wind_coeff"]),
            lai=float(transp["lai"]),
            coeff_radiation=float(transp["coeff_radiation"]),
            coeff_vpd=float(transp["coeff_vpd"]),
            dt_seconds=float(sim["dt_seconds"]),
            stability_warn_frac=float(sim["stability_warn_frac"]),
            fan_power_w=float(fan.get("power_w", 250.0)),
            rh_wet_threshold_min=float(fan.get("rh_wet_threshold_min", 90.0)),
            rh_wet_threshold_max=float(fan.get("rh_wet_threshold_max", 96.0)),
        )


class Polyhouse:
    """Stateful single-zone polyhouse: bulk air temperature and humidity."""

    def __init__(
        self,
        config: dict[str, Any] | PolyhouseParams | None = None,
        t_in_init_c: float = 25.0,
        rh_in_init_pct: float = 70.0,
    ) -> None:
        """
        Args:
            config: Either the full nested config dict (as from
                sim.config.default_config()), a pre-built PolyhouseParams,
                or None to load the project default config.yaml.
            t_in_init_c: Initial internal air temperature, degC.
            rh_in_init_pct: Initial internal relative humidity, % (used to
                derive the initial absolute humidity state).
        """
        if config is None:
            from sim.config import default_config

            config = default_config()
        if isinstance(config, PolyhouseParams):
            self.params = config
        else:
            self.params = PolyhouseParams.from_config(config)

        self.volume_m3: float = self.params.area_floor_m2 * self.params.height_m
        self.c_total_jk: float = self.params.c_eff_jk_per_m2 * self.params.area_floor_m2

        self.t_in: float = t_in_init_c
        self.chi: float = abs_humidity(t_in_init_c, rh_in_init_pct)

        # Cumulative diagnostics.
        self.condensed_kg: float = 0.0
        self.transpired_kg: float = 0.0
        self.fan_kwh: float = 0.0

    def air_changes_per_hour(self, wind_ms: float, vent_frac: float) -> float:
        """Ventilation air-change rate, 1/hr.

        ACH = ach_min + (ach_max - ach_min) * vent_frac, where ach_max
        scales up with outdoor wind speed (wind-driven ventilation).

        Args:
            wind_ms: Outdoor wind speed, m/s.
            vent_frac: Vent opening fraction, 0.0 (closed) to 1.0 (fully open).

        Returns:
            Air changes per hour, 1/hr.
        """
        vent_frac = min(max(vent_frac, 0.0), 1.0)
        ach_max = self.params.ach_max_base * (1.0 + self.params.wind_coeff * max(wind_ms, 0.0))
        return self.params.ach_min + (ach_max - self.params.ach_min) * vent_frac

    def step(
        self,
        t_out_c: float,
        rh_out_pct: float,
        i_solar_wm2: float,
        wind_ms: float,
        vent_frac: float,
        fan_frac: float = 0.0,
        dt_seconds: float | None = None,
    ) -> dict[str, float | bool]:
        """Advance the polyhouse state by one explicit-Euler timestep.

        Args:
            t_out_c: Outdoor air temperature, degC.
            rh_out_pct: Outdoor relative humidity, % (0-100).
            i_solar_wm2: Outdoor incoming shortwave radiation, W/m2.
            wind_ms: Outdoor wind speed, m/s.
            vent_frac: Vent opening fraction, 0.0-1.0.
            fan_frac: Circulation (HAF) fan fraction, 0.0-1.0. Does not
                exchange air with outside; raises the leaf-wetness RH
                threshold (boundary-layer effect, see module docstring) and
                adds a small sensible heat load. Defaults to 0.0 (no fan) for
                backward compatibility with vent-only callers.
            dt_seconds: Integration timestep, s. Defaults to config value.

        Returns:
            Dict with keys: T_in (degC), RH_in (%), VPD (kPa), ET_mm (mm
            transpired this step), condensed (kg condensed on the cover this
            step), leaf_wet (bool, RH_in > rh_wet_threshold(fan_frac)),
            fan_kWh (energy this step, kWh).
        """
        dt = dt_seconds if dt_seconds is not None else self.params.dt_seconds
        area = self.params.area_floor_m2
        volume = self.volume_m3
        fan_frac = min(max(fan_frac, 0.0), 1.0)

        ach = self.air_changes_per_hour(wind_ms, vent_frac)
        vol_flow_m3s = volume * ach / 3600.0
        m_dot_kgs = RHO_AIR * vol_flow_m3s

        # --- Energy balance terms, W (evaluated at the pre-step state) -----
        q_solar = self.params.cover_tau * i_solar_wm2 * area
        q_cond = self.params.cover_u_wm2k * self.params.area_cover_m2 * (t_out_c - self.t_in)
        q_vent = m_dot_kgs * CP_AIR * (t_out_c - self.t_in)
        q_fan = self.params.fan_power_w * fan_frac  # motor heat dissipated into the same air volume

        i_in_wm2 = self.params.cover_tau * i_solar_wm2  # transmitted radiation inside, W/m2
        rh_in_pre = rh_from_chi(self.t_in, self.chi)
        vpd_in_kpa = vpd_kpa(self.t_in, rh_in_pre)
        e_kg_s = area * (
            self.params.coeff_radiation * i_in_wm2 / LAMBDA_VAP
            + self.params.coeff_vpd * self.params.lai * max(vpd_in_kpa, 0.0)
        )
        q_latent = LAMBDA_VAP * e_kg_s

        t_old, chi_old = self.t_in, self.chi

        # --- Energy: exact solution of dT/dt = k_T*(T_eq - T) over this
        # sub-step, with k_T and T_eq held constant (frozen-coefficient
        # explicit-Euler assumption; see module docstring).
        conductance_wk = self.params.cover_u_wm2k * self.params.area_cover_m2 + m_dot_kgs * CP_AIR
        k_t = conductance_wk / self.c_total_jk  # 1/s
        if k_t > 0.0:
            t_eq = t_out_c + (q_solar + q_fan - q_latent) / conductance_wk
            t_new = t_eq + (t_old - t_eq) * math.exp(-k_t * dt)
        else:
            t_new = t_old + (q_solar + q_cond + q_vent + q_fan - q_latent) / self.c_total_jk * dt
        d_t = t_new - t_old

        # --- Vapour: exact solution of dchi/dt = k_chi*(chi_eq - chi).
        chi_out = abs_humidity(t_out_c, rh_out_pct)
        k_chi = ach / 3600.0  # 1/s; always > 0 since ach_min > 0
        chi_eq = e_kg_s / (volume * k_chi) + chi_out
        chi_new = chi_eq + (chi_old - chi_eq) * math.exp(-k_chi * dt)
        d_chi = chi_new - chi_old

        self._check_stability(d_t, d_chi, t_old, chi_old, dt)

        # Clamp to saturation; any excess condenses on the (colder) cover.
        chi_sat_new = chi_sat(t_new)
        condensed_step_kg = 0.0
        if chi_new > chi_sat_new:
            condensed_step_kg = (chi_new - chi_sat_new) * volume
            chi_new = chi_sat_new

        self.t_in = t_new
        self.chi = chi_new
        self.condensed_kg += condensed_step_kg
        self.transpired_kg += e_kg_s * dt
        fan_kwh_step = q_fan * dt / 3.6e6  # W * s -> J -> kWh
        self.fan_kwh += fan_kwh_step

        rh_in = rh_from_chi(t_new, chi_new)
        vpd = vpd_kpa(t_new, rh_in)
        et_mm = (e_kg_s * dt) / area  # 1 kg water / m2 == 1 mm depth
        rh_wet_threshold = (
            self.params.rh_wet_threshold_min
            + (self.params.rh_wet_threshold_max - self.params.rh_wet_threshold_min) * fan_frac
        )

        return {
            "T_in": t_new,
            "RH_in": rh_in,
            "VPD": vpd,
            "ET_mm": et_mm,
            "condensed": condensed_step_kg,
            "fan_kWh": fan_kwh_step,
            "leaf_wet": bool(rh_in > rh_wet_threshold),
        }

    def _check_stability(
        self, d_t: float, d_chi: float, t_old: float, chi_old: float, dt: float
    ) -> None:
        """Warn if either state variable moved by more than the configured
        fraction in a single step (CFL-style explicit-Euler sanity check).
        Temperature uses an absolute (Kelvin) basis so the check is well
        defined even when T_in in degC passes through zero.
        """
        t_old_k = t_old + 273.15
        rel_d_t = abs(d_t) / t_old_k
        rel_d_chi = abs(d_chi) / max(chi_old, 1e-6)
        threshold = self.params.stability_warn_frac
        if rel_d_t > threshold or rel_d_chi > threshold:
            warnings.warn(
                f"Polyhouse.step: state changed by more than {threshold:.0%} "
                f"in one dt={dt:.0f}s step (rel_dT={rel_d_t:.2%}, "
                f"rel_dchi={rel_d_chi:.2%}). Timestep may be too large for "
                "stability.",
                RuntimeWarning,
                stacklevel=3,
            )
