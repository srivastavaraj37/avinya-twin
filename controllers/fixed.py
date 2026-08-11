"""Fixed-schedule controller: the naive baseline a smallholder would run
with a mechanical timer and no sensors.

- Irrigation: a fixed dose applied at two fixed clock times per day
  (default 06:00 and 15:00), regardless of soil moisture or rain.
- Ventilation: a fixed "day open" fraction during a fixed daylight window
  (default 06:00-18:00) and a fixed lower fraction at night, regardless of
  actual indoor temperature/humidity.
- Fan: always off. No sensors means no trigger to run one.

No feedback, no state beyond the clock -- this is the baseline every
reactive/predictive controller should beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sim.soil import SoilBucket


@dataclass
class FixedController:
    """Timer-driven baseline controller.

    Args:
        irrigation_hours: Clock hours (0-23) at which irrigation fires.
        irrigation_dose_mm: Depth applied at each irrigation event, mm.
        day_start_hour: First hour (inclusive) of the daytime vent window.
        day_end_hour: Last hour (exclusive) of the daytime vent window.
        day_vent_frac: Vent opening fraction during the daytime window.
        night_vent_frac: Vent opening fraction outside the daytime window.
    """

    irrigation_hours: tuple[int, ...] = (6, 15)
    irrigation_dose_mm: float = 4.0
    day_start_hour: int = 6
    day_end_hour: int = 18
    day_vent_frac: float = 0.5
    night_vent_frac: float = 0.0

    def vent_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        if self.day_start_hour <= ts.hour < self.day_end_hour:
            return self.day_vent_frac
        return self.night_vent_frac

    def irrigation_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        if ts.hour in self.irrigation_hours:
            return self.irrigation_dose_mm
        return 0.0

    def fan_policy(
        self, state: dict[str, Any], soil: SoilBucket, row: "pd.Series[Any]", ts: pd.Timestamp
    ) -> float:
        # The naive timer baseline has no sensor input and no fan -- a real
        # smallholder running vents on a clock has no reason to have also
        # installed (and be running) a circulation fan.
        return 0.0
