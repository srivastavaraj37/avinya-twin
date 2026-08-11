"""GATE 6 (part 1): Wallin DSV lookup-table sanity checks.

The full 90-day monsoon-run comparison (vents closed vs. open) requires the
engine (Phase 7) and lives in tests/test_validation.py /
tests/test_engine_disease.py alongside it.
"""

from __future__ import annotations

from sim.disease import AlternariaRisk, DiseaseModel, alternaria_risk_units, dsv


def test_dsv_table_reference_values():
    assert dsv(20.0, 10) == 1
    assert dsv(20.0, 25) == 4
    assert dsv(30.0, 20) == 0


def test_dsv_table_band_boundaries():
    # 7.2-11.6 band thresholds: <15->0, <19->1, <23->2, <28->3, else 4
    assert dsv(9.0, 14) == 0
    assert dsv(9.0, 15) == 1
    assert dsv(9.0, 30) == 4
    # 11.7-15.0 band thresholds: <12->0, <16->1, <19->2, <22->3, else 4
    assert dsv(13.0, 11) == 0
    assert dsv(13.0, 22) == 4
    # Below the lowest band -> 0
    assert dsv(5.0, 100) == 0


def test_disease_model_accumulates_and_flags_threshold():
    model = DiseaseModel(spray_threshold_dsv=18)
    # 20 consecutive wet hours at 20 degC -> dsv(20, 20): band 15.1-26.6,
    # thresholds <9->0,<12->1,<16->2,<19->3, else 4; 20 >= 19 -> DSV 4.
    for _ in range(20):
        model.step(t_in_c=20.0, leaf_wet=True)
    model.step(t_in_c=20.0, leaf_wet=False)  # closes the period
    assert model.cumulative_dsv == dsv(20.0, 20)
    assert model.total_wet_hours == 20
    assert model.spray_threshold_crossed is False


def test_disease_model_finalize_closes_open_period():
    model = DiseaseModel()
    for _ in range(10):
        model.step(t_in_c=20.0, leaf_wet=True)
    assert model.cumulative_dsv == 0  # period still open
    model.finalize()
    assert model.cumulative_dsv == dsv(20.0, 10)


def test_alternaria_risk_units_temperature_band():
    # In-band (24-29 degC): duration-gated.
    assert alternaria_risk_units(26.0, 9) == 0
    assert alternaria_risk_units(26.0, 10) == 1
    assert alternaria_risk_units(26.0, 15) == 1
    assert alternaria_risk_units(26.0, 16) == 2
    assert alternaria_risk_units(26.0, 40) == 2
    # Band edges are inclusive.
    assert alternaria_risk_units(24.0, 16) == 2
    assert alternaria_risk_units(29.0, 16) == 2
    # Out of band (below 24 or above 29): zero regardless of duration --
    # this is the warm-climate-favourable band Wallin's table lacks.
    assert alternaria_risk_units(23.9, 20) == 0
    assert alternaria_risk_units(29.1, 20) == 0
    assert alternaria_risk_units(35.0, 40) == 0


def test_alternaria_model_accumulates_seasonally():
    model = AlternariaRisk()
    # First period: 26 degC, 16 wet hours -> 2 units.
    for _ in range(16):
        model.step(t_in_c=26.0, leaf_wet=True)
    model.step(t_in_c=26.0, leaf_wet=False)  # closes the period
    assert model.cumulative_risk == 2
    assert model.total_wet_hours == 16

    # Second period on top: 27 degC, 10 wet hours -> +1 unit, accumulates.
    for _ in range(10):
        model.step(t_in_c=27.0, leaf_wet=True)
    model.step(t_in_c=27.0, leaf_wet=False)
    assert model.cumulative_risk == 3
    assert model.total_wet_hours == 26


def test_alternaria_model_finalize_closes_open_period():
    model = AlternariaRisk()
    for _ in range(16):
        model.step(t_in_c=26.0, leaf_wet=True)
    assert model.cumulative_risk == 0  # period still open
    model.finalize()
    assert model.cumulative_risk == 2


def test_alternaria_model_forces_day_boundary_close():
    # Same rationale as DiseaseModel: a sustained, unbroken wet stretch
    # (e.g. a permanently sealed polyhouse) must not collapse into a single
    # period that hides the accumulation -- end_of_day forces a close.
    model = AlternariaRisk()
    for day in range(3):
        for hour in range(24):
            model.step(t_in_c=26.0, leaf_wet=True, end_of_day=(hour == 23))
    # 3 closed periods of 24h each (>=16h -> 2 units) = 6, not one 72h period.
    assert model.cumulative_risk == 6
    assert model.total_wet_hours == 72
