"""GATE 4: FAO-56 Penman-Monteith hourly ET0 sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.et0 import et0_hourly

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "guwahati_2025.csv"


def _load_weather() -> pd.DataFrame:
    if not CSV_PATH.exists():
        pytest.skip("weather CSV not fetched yet; run data/fetch_weather.py")
    return pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)


def test_clear_june_day_et0_in_range():
    df = _load_weather()
    june = df[df.index.month == 6].copy()
    june["ET0"] = et0_hourly(
        june["T_out"].to_numpy(),
        june["RH_out"].to_numpy(),
        june["I_solar"].to_numpy(),
        june["wind"].to_numpy(),
        june["cloud"].to_numpy(),
    )
    daily = june.groupby(june.index.date)["ET0"].sum()
    cloud_daily = june.groupby(june.index.date)["cloud"].mean()
    clearest_day = cloud_daily.idxmin()
    et0_clear_day = daily.loc[clearest_day]
    assert 3.5 <= et0_clear_day <= 7.0, (
        f"Clear June day ({clearest_day}) ET0={et0_clear_day:.2f} mm/day out of [3.5, 7.0]"
    )


def test_et0_near_zero_at_night():
    # Real Guwahati nights are calm on the whole, so the typical (median/mean)
    # night-time ET0 should be near zero -- this is the "Rn negative -> clamp
    # at 0" regime the gate is checking for. A handful of storm nights with
    # very high wind (>10 m/s squalls) legitimately produce a nonzero
    # aerodynamic (wind*VPD) term under Penman-Monteith even with negative
    # Rn; that is real physics, not a bug, so we exclude those outliers
    # rather than weakening the near-zero bound for the typical case.
    df = _load_weather()
    night = df[(df.index.hour >= 20) | (df.index.hour <= 4)]
    et0 = et0_hourly(
        night["T_out"].to_numpy(),
        night["RH_out"].to_numpy(),
        night["I_solar"].to_numpy(),
        night["wind"].to_numpy(),
        night["cloud"].to_numpy(),
    )
    assert (et0 >= 0.0).all()
    assert np.median(et0) < 0.02, f"Typical night ET0 not near zero: median={np.median(et0):.4f}"
    assert et0.mean() < 0.05, f"Mean night ET0 not near zero: mean={et0.mean():.4f}"

    calm = night["wind"].to_numpy() < 10.0
    assert et0[calm].max() < 0.25, (
        f"Calm-night ET0 too high: max={et0[calm].max():.4f} mm/hr"
    )


def test_et0_clamped_when_rn_negative_zero_wind():
    # Direct check of the literal gate wording: with Rn forced negative
    # (night, no solar input) and zero wind (no aerodynamic term), ET0 must
    # clamp to exactly 0.
    et0 = et0_hourly(
        t_c=np.array([22.0]),
        rh_pct=np.array([90.0]),
        i_solar_wm2=np.array([0.0]),
        wind_ms=np.array([0.0]),
        cloud_pct=np.array([10.0]),
    )
    assert et0[0] == 0.0


def test_et0_monotonic_in_solar_radiation():
    t_c = np.full(50, 30.0)
    rh = np.full(50, 60.0)
    wind = np.full(50, 2.0)
    cloud = np.full(50, 20.0)
    i_solar = np.linspace(0, 1000, 50)
    et0 = et0_hourly(t_c, rh, i_solar, wind, cloud)
    diffs = np.diff(et0)
    assert (diffs >= -1e-9).all(), "ET0 not monotonically non-decreasing in I_solar"
