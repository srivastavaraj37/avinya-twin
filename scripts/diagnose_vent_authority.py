"""Diagnostic: how much control authority does ventilation alone have over
indoor VPD during JJA monsoon, when outdoor air is itself near-saturated?

Runs the 90-day monsoon window (2025-06-01..2025-08-29, same as
scripts/experiment.py) twice -- vent_frac permanently 0.0 and permanently
1.0, no irrigation -- and at every hour computes the achievable VPD range
|VPD(vent=1) - VPD(vent=0)|. Reports the mean range and the fraction of
hours where that range is narrower than 0.2 kPa (i.e. hours where no vent
setting can move indoor VPD by more than 0.2 kPa either way).

This quantifies the hypothesis written up in CLAUDE.md's "Problem 1" finding:
ambient RH_out of 85-95% during JJA means ventilation is mostly exchanging
saturated indoor air for near-saturated outdoor air, so it has little
authority over VPD or leaf wetness -- which is why Fixed/Threshold/Predictive
land on nearly identical VPD-band and DSV numbers in results/fixed.csv etc.

Usage:
    python scripts/diagnose_vent_authority.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.engine import run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WEATHER_CSV = REPO_ROOT / "data" / "guwahati_2025.csv"
WINDOW_START = "2025-06-01"
WINDOW_END = "2025-08-30"  # exclusive -> 90 days

NARROW_RANGE_KPA = 0.2


def _no_irrigation(state, soil, row, ts) -> float:
    return 0.0


def main() -> int:
    df = pd.read_csv(WEATHER_CSV, index_col=0, parse_dates=True)
    window = df[(df.index >= WINDOW_START) & (df.index < WINDOW_END)]
    assert len(window) >= 90 * 24, "expected >= 90 days of hourly monsoon weather"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res_closed = run(window, lambda s, so, r, t: 0.0, _no_irrigation, show_progress=False)
        res_open = run(window, lambda s, so, r, t: 1.0, _no_irrigation, show_progress=False)

    achievable_range = (res_open["VPD"] - res_closed["VPD"]).abs()

    mean_range = float(achievable_range.mean())
    median_range = float(achievable_range.median())
    frac_narrow = float((achievable_range < NARROW_RANGE_KPA).mean())
    n_hours = len(achievable_range)

    jja_rh_out_mean = float(window["RH_out"].mean())

    # Same statistic split day (06-18h) vs night (else), since the daytime
    # solar-driven greenhouse effect gives vents more thermal (and therefore
    # VPD) authority than at night.
    hour = achievable_range.index.hour
    day_mask = (hour >= 6) & (hour < 18)
    day_frac_narrow = float((achievable_range[day_mask] < NARROW_RANGE_KPA).mean())
    night_frac_narrow = float((achievable_range[~day_mask] < NARROW_RANGE_KPA).mean())

    print(f"Window: {WINDOW_START} to {WINDOW_END} ({n_hours} hours)")
    print(f"Mean JJA RH_out: {jja_rh_out_mean:.2f} %")
    print()
    print(f"Mean achievable VPD range (vent=1 vs vent=0):   {mean_range:.4f} kPa")
    print(f"Median achievable VPD range:                     {median_range:.4f} kPa")
    print(f"Fraction of hours with range < {NARROW_RANGE_KPA} kPa:        {frac_narrow:.4f} ({frac_narrow * 100:.1f}%)")
    print(f"  -- daytime (06-18h):                           {day_frac_narrow:.4f} ({day_frac_narrow * 100:.1f}%)")
    print(f"  -- night (18-06h):                              {night_frac_narrow:.4f} ({night_frac_narrow * 100:.1f}%)")
    print()
    print("Percentiles of achievable range (kPa):")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q*100):02d}: {achievable_range.quantile(q):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
