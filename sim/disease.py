"""Wallin (1962) late-blight Disease Severity Value (DSV) model, as used in
the BLITECAST forecasting system.

A "wetness period" is a run of consecutive hours with leaf surfaces wet
(condensation or free moisture, here taken from Polyhouse's leaf_wet flag,
RH_in > 90%). When a period ends, its mean temperature and duration look up
a Disease Severity Value (0-4) from the Wallin table; DSVs accumulate over
the season and the cumulative total tracks blight infection pressure.

Reference:
    Wallin, J.R. (1962). "Summary of recent progress in predicting late
    blight epidemics in United States and Canada." American Potato Journal,
    39, 306-312. (Table reproduced as used in the BLITECAST model, e.g.
    Krause, Massie & Hyre 1975, Phytopathology 65:428-432.)
"""

from __future__ import annotations

from dataclasses import dataclass, field


def dsv(mean_temp_c: float, wet_hours: float) -> int:
    """Wallin Disease Severity Value for one leaf-wetness period.

    Args:
        mean_temp_c: Mean air temperature during the wetness period, degC.
        wet_hours: Duration of the continuous wetness period, hours.

    Returns:
        Disease Severity Value, an integer in {0, 1, 2, 3, 4}. Periods with
        a mean temperature outside the Wallin table's covered bands
        (7.2-26.6 degC) contribute 0 (temperature too low or too high for
        the fungus to establish in that window).
    """
    if 7.2 <= mean_temp_c <= 11.6:
        thresholds = (15.0, 19.0, 23.0, 28.0)
    elif 11.7 <= mean_temp_c <= 15.0:
        thresholds = (12.0, 16.0, 19.0, 22.0)
    elif 15.1 <= mean_temp_c <= 26.6:
        thresholds = (9.0, 12.0, 16.0, 19.0)
    else:
        return 0

    for severity, threshold in enumerate(thresholds):
        if wet_hours < threshold:
            return severity
    return 4


@dataclass
class DiseaseModel:
    """Stateful tracker that turns an hourly (T_in, leaf_wet) stream into
    cumulative Wallin DSV.

    Args:
        spray_threshold_dsv: Cumulative DSV at which BLITECAST recommends a
            protectant fungicide spray.
    """

    spray_threshold_dsv: float = 18.0

    cumulative_dsv: int = field(default=0, init=False)
    total_wet_hours: int = field(default=0, init=False)
    spray_threshold_crossed: bool = field(default=False, init=False)

    _period_temps: list[float] = field(default_factory=list, init=False)

    def step(self, t_in_c: float, leaf_wet: bool, end_of_day: bool = False) -> None:
        """Feed one hour of (temperature, leaf-wetness) into the tracker.

        Args:
            t_in_c: Mean internal air temperature during this hour, degC.
            leaf_wet: Whether leaf surfaces were wet during this hour.
            end_of_day: True on the last hour of a calendar day. BLITECAST
                (Krause et al. 1975) evaluates wetness periods and their DSV
                once per day; forcing a period boundary at midnight keeps
                that behaviour even when RH_in never drops below 90% for
                days on end (e.g. a permanently closed-up polyhouse). Without
                this, an unbroken multi-week wet stretch would count as a
                single wetness period, and since the Wallin table saturates
                at DSV=4 for any duration beyond ~19-28 hours, it would
                contribute only one DSV=4 total instead of accumulating
                day after day -- silently *hiding* the sustained-high-
                humidity disease risk this model exists to surface.
        """
        if leaf_wet:
            self._period_temps.append(t_in_c)
            if end_of_day:
                self._close_period()
        else:
            self._close_period()

    def finalize(self) -> None:
        """Close out any still-open wetness period at the end of a run."""
        self._close_period()

    def _close_period(self) -> None:
        if not self._period_temps:
            return
        duration = len(self._period_temps)
        mean_temp = sum(self._period_temps) / duration
        severity = dsv(mean_temp, duration)
        self.cumulative_dsv += severity
        self.total_wet_hours += duration
        self._period_temps = []
        if self.cumulative_dsv >= self.spray_threshold_dsv:
            self.spray_threshold_crossed = True


# ---------------------------------------------------------------------------
# Alternaria solani (early blight) risk model
# ---------------------------------------------------------------------------
#
# Added 2026-08-11 alongside the Wallin model above, not instead of it. Per
# CLAUDE.md's Gate 6b finding, the Wallin (1962) late-blight table's
# favourable band tops out at 26.6 degC -- calibrated for the temperate US
# Northeast potato belt Wallin/BLITECAST were built for -- which Guwahati's
# 28.7 degC JJA mean ambient exceeds on 89 of 92 monsoon days before any
# polyhouse warming, making cumulative Wallin DSV structurally near-zero in
# this climate and not a useful controller-comparison signal on its own.
#
# Alternaria solani (early blight) is a warm-climate fungal pathogen and the
# actual disease risk of practical concern for tomato in this region --
# established early-blight forecasting systems (Madden, Pennypacker & MacNab
# 1978, "FAST"; Pitblado 1992, "TOMCAST", the system commonly used to
# schedule fungicide sprays for tomato early blight in warm-temperate/
# subtropical growing regions) use a favourable temperature window during
# the wetness period that extends into the high 20s degC, unlike Wallin's
# temperate-capped table. The exact two-tier duration/temperature rule
# implemented here (24-29 degC, >=10h -> 1 unit, >=16h -> 2 units) is a
# simplified scheme specified for this project -- it is not a literal
# reproduction of the FAST/TOMCAST DSV tables (which use more numerous,
# finer-grained temperature/duration bands); it follows the same underlying
# mechanism (mean temperature during a continuous wetness period, banded and
# duration-gated) and the same warm-favourable philosophy, at a level of
# detail appropriate to this project's other lookup-table disease model.
#
# References:
#     Madden, L., Pennypacker, S.P., MacNab, A.A. (1978). "FAST, a forecast
#     system for Alternaria solani on tomato." Phytopathology 68:1354-1358.
#     Pitblado, R.E. (1992). "The development and implementation of TOMCAST
#     (TOMato disease foreCASTing) system." Ontario Ministry of Agriculture
#     and Food.


def alternaria_risk_units(mean_temp_c: float, wet_hours: float) -> int:
    """Alternaria solani (early blight) risk units for one leaf-wetness period.

    Favourable condition: mean temperature 24-29 degC sustained through the
    continuous wetness period. Outside that band the period contributes 0
    regardless of duration -- like late blight, early blight infection needs
    a compatible temperature window, just a warmer one (see module notes).

    Args:
        mean_temp_c: Mean air temperature during the wetness period, degC.
        wet_hours: Duration of the continuous wetness period, hours.

    Returns:
        Risk units, integer in {0, 1, 2}.
    """
    if not (24.0 <= mean_temp_c <= 29.0):
        return 0
    if wet_hours >= 16.0:
        return 2
    if wet_hours >= 10.0:
        return 1
    return 0


@dataclass
class AlternariaRisk:
    """Stateful tracker that turns an hourly (T_in, leaf_wet) stream into
    cumulative Alternaria early-blight risk units.

    Mirrors DiseaseModel's period-tracking structure (see its ``step``
    docstring for why periods are also force-closed at day boundaries, not
    just on leaf_wet=False) but scores each period against the warm-climate
    Alternaria band instead of Wallin's temperate one.
    """

    cumulative_risk: int = field(default=0, init=False)
    total_wet_hours: int = field(default=0, init=False)

    _period_temps: list[float] = field(default_factory=list, init=False)

    def step(self, t_in_c: float, leaf_wet: bool, end_of_day: bool = False) -> None:
        """Feed one hour of (temperature, leaf-wetness) into the tracker.

        Args:
            t_in_c: Mean internal air temperature during this hour, degC.
            leaf_wet: Whether leaf surfaces were wet during this hour.
            end_of_day: True on the last hour of a calendar day (forces a
                period boundary at midnight; see DiseaseModel.step).
        """
        if leaf_wet:
            self._period_temps.append(t_in_c)
            if end_of_day:
                self._close_period()
        else:
            self._close_period()

    def finalize(self) -> None:
        """Close out any still-open wetness period at the end of a run."""
        self._close_period()

    def _close_period(self) -> None:
        if not self._period_temps:
            return
        duration = len(self._period_temps)
        mean_temp = sum(self._period_temps) / duration
        self.cumulative_risk += alternaria_risk_units(mean_temp, duration)
        self.total_wet_hours += duration
        self._period_temps = []
