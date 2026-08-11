"""Reactive threshold controller: simple sensor-driven bang-bang control.

- Ventilation: opens on temperature OR humidity crossing an "open"
  threshold, closes on both dropping back below a lower "close" threshold
  (two-sided hysteresis deadband, to avoid chattering the vents open/closed
  every single hour on noisy readings right at one threshold).
- Irrigation: fires a fixed dose whenever the soil bucket is stressed
  (depletion past the FAO-56 readily-available-water threshold), unless it
  is currently raining (skip -- let the rain do the work).
- Fan: on at full power whenever RH_in > 92% -- simple bang-bang, no
  hysteresis (unlike vent_policy). This is a deliberately blunter rule than
  the vent logic: the fan has no thermal downside to chatter (it doesn't
  exchange air or move ACH), so there's no overcooling/overcooling-adjacent
  risk to guard against with a deadband the way there is for vents.

This is the standard "IoT sensor + relay" controller a real polyhouse
retrofit would use: no forecasting, purely reactive to the current
(previous-hour-observed) state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from sim.soil import SoilBucket


@dataclass
class ThresholdController:
    """Reactive bang-bang controller with vent hysteresis.

    Args:
        t_open_c: Open the vents fully if T_in rises above this, degC.
        t_close_c: Allow the vents to close once T_in drops below this
            (must be < t_open_c to form a hysteresis deadband), degC.
        rh_open_pct: Open the vents partially if RH_in rises above this, %.
        rh_close_pct: Allow closing once RH_in drops below this, %.
        vent_open_frac: Vent fraction used when the temperature trigger fires.
        vent_humid_frac: Vent fraction used when only the humidity trigger fires.
        vent_closed_frac: Vent fraction when neither trigger is active
            (baseline infiltration only).
        irrigation_dose_mm: Depth applied when triggered, mm.
        rain_skip_mm: Skip irrigation if this hour's rain meets/exceeds this, mm.
        fan_rh_on_pct: Run the circulation fan at full power if RH_in rises
            above this, %.
    """

    t_open_c: float = 32.0
    t_close_c: float = 29.0
    rh_open_pct: float = 88.0
    rh_close_pct: float = 82.0
    vent_open_frac: float = 1.0
    vent_humid_frac: float = 0.6
    vent_closed_frac: float = 0.1
    irrigation_dose_mm: float = 5.0
    rain_skip_mm: float = 1.0
    fan_rh_on_pct: float = 92.0

    _vent_is_open: bool = False

    def vent_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        t_in = state["T_in"]
        rh_in = state["RH_in"] if state["RH_in"] is not None else 70.0

        if self._vent_is_open:
            # Only close once BOTH signals have dropped back into the safe zone.
            if t_in < self.t_close_c and rh_in < self.rh_close_pct:
                self._vent_is_open = False
        else:
            if t_in > self.t_open_c or rh_in > self.rh_open_pct:
                self._vent_is_open = True

        if not self._vent_is_open:
            return self.vent_closed_frac
        # Temperature breach gets full venting; a humidity-only breach gets
        # a gentler partial opening so we don't overcool a merely-humid night.
        if t_in > self.t_open_c:
            return self.vent_open_frac
        return self.vent_humid_frac

    def irrigation_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        if row["rain"] >= self.rain_skip_mm:
            return 0.0
        if soil.stressed:
            return self.irrigation_dose_mm
        return 0.0

    def fan_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        rh_in = state["RH_in"] if state["RH_in"] is not None else 70.0
        return 1.0 if rh_in > self.fan_rh_on_pct else 0.0
