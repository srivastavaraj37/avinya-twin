"""Phase 8 physical-realism gates (V1-V8).

These are the gates that decide whether the coupled polyhouse physics is
trustworthy enough to build controllers on top of. Bounds are not weakened
to make a test pass; if a gate fails, the underlying model must change.

V1-V6 run a 7-day June (monsoon) simulation with a passive policy: vent
fixed at 0.3, no irrigation. The window (2025-06-22 to 2025-06-29) was
chosen as a representative monsoon week (mean cloud cover 75%, mean RH_out
87%, intermittent rain -- see scripts/validate.py output), not cherry-picked
for extremity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.engine import run
from sim.polyhouse import Polyhouse

WEATHER_CSV = Path(__file__).resolve().parent.parent / "data" / "guwahati_2025.csv"
VALIDATION_START = "2025-06-22"
VALIDATION_END = "2025-06-29"  # exclusive


def _load_validation_week() -> pd.DataFrame:
    if not WEATHER_CSV.exists():
        pytest.skip("weather CSV not fetched yet; run data/fetch_weather.py")
    df = pd.read_csv(WEATHER_CSV, index_col=0, parse_dates=True)
    week = df[(df.index >= VALIDATION_START) & (df.index < VALIDATION_END)]
    assert len(week) == 7 * 24, f"expected 168 hourly rows, got {len(week)}"
    return week


def _passive_vent(state, soil, row, ts) -> float:
    return 0.3


def _no_irrigation(state, soil, row, ts) -> float:
    return 0.0


@pytest.fixture(scope="module")
def passive_run() -> pd.DataFrame:
    week = _load_validation_week()
    res = run(week, _passive_vent, _no_irrigation, show_progress=False)
    res = res.copy()
    res["T_out"] = week["T_out"].to_numpy()
    return res


def test_v1_daytime_greenhouse_effect(passive_run: pd.DataFrame):
    day = passive_run[(passive_run.index.hour >= 11) & (passive_run.index.hour <= 15)]
    gap = (day["T_in"] - day["T_out"]).mean()
    assert 4.0 <= gap <= 16.0, f"V1 FAIL: daytime T_in-T_out gap = {gap:.2f} degC"


def test_v2_predawn_humidity(passive_run: pd.DataFrame):
    predawn = passive_run[(passive_run.index.hour >= 3) & (passive_run.index.hour <= 6)]
    mean_rh = predawn["RH_in"].mean()
    assert mean_rh > 92.0, f"V2 FAIL: pre-dawn mean RH_in = {mean_rh:.2f}%"


def test_v3_peak_hourly_et(passive_run: pd.DataFrame):
    peak_et = passive_run["ET_hr"].max()
    assert 0.25 <= peak_et <= 0.70, f"V3 FAIL: peak hourly ET = {peak_et:.3f} mm/hr"


def test_v4_daily_total_et(passive_run: pd.DataFrame):
    daily = passive_run.groupby(passive_run.index.date)["ET_hr"].sum()
    assert daily.min() >= 2.5, f"V4 FAIL: min daily ET = {daily.min():.2f} mm/day"
    assert daily.max() <= 6.0, f"V4 FAIL: max daily ET = {daily.max():.2f} mm/day"


def test_v5_temperature_physically_plausible(passive_run: pd.DataFrame):
    assert passive_run["T_in"].min() >= 5.0, f"V5 FAIL: T_in min = {passive_run['T_in'].min():.2f}"
    assert passive_run["T_in"].max() <= 55.0, f"V5 FAIL: T_in max = {passive_run['T_in'].max():.2f}"


def test_v6_night_leaf_wetness_signature(passive_run: pd.DataFrame):
    night = passive_run[(passive_run.index.hour >= 20) | (passive_run.index.hour <= 5)]
    frac_wet = night["leaf_wet"].mean()
    assert frac_wet > 0.5, f"V6 FAIL: night leaf_wet fraction = {frac_wet:.2f}"


def test_v7_ventilation_monotonically_cools_and_dries():
    week = _load_validation_week()
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    mean_t_in = []
    mean_rh_in = []
    for level in levels:
        res = run(week, lambda s, so, r, t, v=level: v, _no_irrigation, show_progress=False)
        day = res[(res.index.hour >= 11) & (res.index.hour <= 15)]
        mean_t_in.append(day["T_in"].mean())
        mean_rh_in.append(day["RH_in"].mean())

    assert all(a > b for a, b in zip(mean_t_in, mean_t_in[1:])), (
        f"V7 FAIL: mean daytime T_in not monotonically decreasing across vent levels: {mean_t_in}"
    )
    assert all(a > b for a, b in zip(mean_rh_in, mean_rh_in[1:])), (
        f"V7 FAIL: mean daytime RH_in not monotonically decreasing across vent levels: {mean_rh_in}"
    )


def test_v8_no_solar_steady_state_converges_to_outside_temp():
    # RH_out=95% (a realistic Guwahati monsoon night humidity) keeps the
    # VPD-driven latent term small so the steady state genuinely approaches
    # T_out, rather than sitting at a permanent evaporative-cooling offset.
    ph = Polyhouse(t_in_init_c=35.0, rh_in_init_pct=60.0)
    out = None
    for _ in range(12 * 48):  # 48 simulated hours -> steady state
        out = ph.step(t_out_c=25.0, rh_out_pct=95.0, i_solar_wm2=0.0, wind_ms=1.0, vent_frac=0.3)
    assert abs(out["T_in"] - 25.0) < 0.5, f"V8 FAIL: |T_in - T_out| = {abs(out['T_in'] - 25.0):.3f}"
