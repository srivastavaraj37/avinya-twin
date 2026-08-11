"""Unit sanity checks for the coupled polyhouse energy + vapour balance.

Full physical-realism gates (V1-V8) live in tests/test_validation.py; this
file checks the model's basic mechanics in isolation.
"""

from __future__ import annotations

import warnings

import pytest

from sim.polyhouse import Polyhouse
from sim.psychro import abs_humidity


def test_daytime_solar_heats_interior_above_outside():
    ph = Polyhouse(t_in_init_c=30.0, rh_in_init_pct=70.0)
    out = None
    for _ in range(12 * 3):  # 3 simulated hours of strong sun, low vent
        out = ph.step(t_out_c=30.0, rh_out_pct=70.0, i_solar_wm2=800.0, wind_ms=1.0, vent_frac=0.1)
    assert out["T_in"] > 30.0


def test_no_solar_no_wind_relaxes_toward_outside_temp():
    # With I_solar=0 and matched outdoor/indoor humidity (so the VPD-driven
    # latent term is small), T_in should relax toward T_out over time.
    ph = Polyhouse(t_in_init_c=40.0, rh_in_init_pct=90.0)
    out = None
    for _ in range(12 * 6):  # 6 hours
        out = ph.step(t_out_c=25.0, rh_out_pct=90.0, i_solar_wm2=0.0, wind_ms=1.0, vent_frac=0.5)
    assert abs(out["T_in"] - 25.0) < 1.5


def test_condensation_when_ventilation_is_shut_and_humidity_is_high():
    ph = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=95.0)
    total_condensed = 0.0
    for _ in range(12 * 4):  # 4 hours, vents closed, cool humid outside air
        out = ph.step(t_out_c=22.0, rh_out_pct=98.0, i_solar_wm2=0.0, wind_ms=0.5, vent_frac=0.0)
        total_condensed += out["condensed"]
    assert total_condensed > 0.0
    assert ph.condensed_kg == pytest.approx(total_condensed)


def test_leaf_wet_flag_matches_rh_threshold():
    ph = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=95.0)
    out = ph.step(t_out_c=22.0, rh_out_pct=98.0, i_solar_wm2=0.0, wind_ms=0.5, vent_frac=0.0)
    assert out["leaf_wet"] == (out["RH_in"] > 90.0)


def test_stability_warning_fires_for_oversized_timestep():
    ph = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=60.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ph.step(
            t_out_c=45.0,
            rh_out_pct=20.0,
            i_solar_wm2=900.0,
            wind_ms=0.0,
            vent_frac=0.0,
            dt_seconds=36000.0,  # 10x the normal 300s sub-step
        )
    assert any(issubclass(x.category, RuntimeWarning) for x in caught)


def test_vent_frac_out_of_range_is_clamped():
    ph = Polyhouse()
    ach_over = ph.air_changes_per_hour(wind_ms=1.0, vent_frac=5.0)
    ach_at_1 = ph.air_changes_per_hour(wind_ms=1.0, vent_frac=1.0)
    assert ach_over == ach_at_1


def test_more_ventilation_cools_and_dries_faster_toward_outside():
    # Two identical hot/humid starts; the one with vents open should end
    # closer to (cooler, drier) outside air than the one with vents shut.
    common = dict(t_out_c=28.0, rh_out_pct=80.0, i_solar_wm2=0.0, wind_ms=2.0)
    ph_closed = Polyhouse(t_in_init_c=38.0, rh_in_init_pct=95.0)
    ph_open = Polyhouse(t_in_init_c=38.0, rh_in_init_pct=95.0)
    for _ in range(12 * 2):
        out_closed = ph_closed.step(vent_frac=0.0, **common)
        out_open = ph_open.step(vent_frac=1.0, **common)
    assert out_open["T_in"] < out_closed["T_in"]


def test_fan_defaults_to_zero_and_preserves_original_90pct_threshold():
    # fan_frac defaults to 0.0, which must reproduce the original (pre-fan)
    # leaf-wet behaviour exactly: threshold pinned at 90%.
    ph = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=95.0)
    out = ph.step(t_out_c=22.0, rh_out_pct=98.0, i_solar_wm2=0.0, wind_ms=0.5, vent_frac=0.0)
    assert out["leaf_wet"] == (out["RH_in"] > 90.0)


def test_fan_raises_the_leaf_wet_threshold():
    # A regime whose steady-state RH_in lands in the 90-96% band (checked
    # numerically: settles ~93%) with and without the fan at full power. At
    # 93% bulk RH, fan-off (threshold 90%) reads wet and fan-on (threshold
    # 96%) reads dry -- the whole point of the boundary-layer effect.
    common = dict(t_out_c=26.0, rh_out_pct=80.0, i_solar_wm2=0.0, wind_ms=0.5, vent_frac=0.3)
    ph_no_fan = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=80.0)
    ph_fan = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=80.0)
    wet_no_fan = wet_fan = 0
    out_no_fan = out_fan = None
    for _ in range(12 * 10):
        out_no_fan = ph_no_fan.step(fan_frac=0.0, **common)
        out_fan = ph_fan.step(fan_frac=1.0, **common)
        wet_no_fan += int(out_no_fan["leaf_wet"])
        wet_fan += int(out_fan["leaf_wet"])
    assert 90.0 < out_no_fan["RH_in"] < 96.0, f"test regime drifted out of band: {out_no_fan['RH_in']}"
    assert wet_no_fan > wet_fan
    assert out_no_fan["leaf_wet"] is True
    assert out_fan["leaf_wet"] is False


def test_fan_adds_sensible_heat():
    # Identical conditions, vents closed, fan on vs off; the fan-on run
    # should end warmer (its motor heat has nowhere to go but the same air).
    common = dict(t_out_c=25.0, rh_out_pct=70.0, i_solar_wm2=0.0, wind_ms=1.0, vent_frac=0.2)
    ph_off = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=70.0)
    ph_on = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=70.0)
    out_off = out_on = None
    for _ in range(12 * 2):
        out_off = ph_off.step(fan_frac=0.0, **common)
        out_on = ph_on.step(fan_frac=1.0, **common)
    assert out_on["T_in"] > out_off["T_in"]


def test_fan_kwh_accumulates_at_rated_power():
    # At fan_frac=1.0 for exactly one hour, fan_kwh should equal
    # fan_power_w/1000 (the configured rating), regardless of vent/weather.
    ph = Polyhouse(t_in_init_c=25.0, rh_in_init_pct=70.0)
    expected_kwh = ph.params.fan_power_w / 1000.0
    for _ in range(12):  # 12 * 300s = 1 hour
        ph.step(t_out_c=25.0, rh_out_pct=70.0, i_solar_wm2=0.0, wind_ms=1.0, vent_frac=0.0, fan_frac=1.0)
    assert ph.fan_kwh == pytest.approx(expected_kwh, rel=1e-6)
