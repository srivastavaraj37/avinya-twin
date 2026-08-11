"""GATE 3: FAO-56 soil water bucket sanity checks."""

from __future__ import annotations

import random

from sim.soil import SoilBucket


def test_taw_default():
    bucket = SoilBucket()
    assert abs(bucket.taw - 68.0) < 1e-9


def test_water_balance_closes_random_steps():
    rng = random.Random(42)
    bucket = SoilBucket(d_init_mm=10.0)
    d_initial = bucket.d

    for _ in range(1000):
        etc_mm = rng.uniform(0.0, 3.0)
        irrigation_mm = rng.uniform(0.0, 10.0) if rng.random() < 0.3 else 0.0
        rain_mm = rng.uniform(0.0, 15.0) if rng.random() < 0.2 else 0.0
        bucket.step(etc_mm, irrigation_mm, rain_mm)

        # D must always stay within physical bounds.
        assert -1e-9 <= bucket.d <= bucket.taw + 1e-9

    d_final = bucket.d
    # Derived closure identity (see SoilBucket.step docstring):
    #   cum_inflow - cum_et_actual - runoff == d_initial - d_final
    lhs = bucket.cum_inflow_mm - bucket.cum_et_actual_mm - bucket.runoff_mm
    rhs = d_initial - d_final
    assert abs(lhs - rhs) < 1e-6
    assert abs(bucket.balance_residual()) < 1e-6


def test_depletion_never_out_of_bounds():
    rng = random.Random(7)
    bucket = SoilBucket()
    for _ in range(1000):
        bucket.step(
            etc_mm=rng.uniform(0.0, 8.0),
            irrigation_mm=rng.uniform(0.0, 2.0),
            rain_mm=0.0,
        )
        assert bucket.d >= 0.0
        assert bucket.d <= bucket.taw
