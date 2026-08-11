"""Top-level simulation loop coupling weather, the polyhouse, soil, and disease models.

Policies are injected callables, so arbitrary vent/irrigation/fan controllers
can be compared against the same physical plant:

    vent_policy(state, soil, weather_row, timestamp) -> float in [0, 1]
    irrigation_policy(state, soil, weather_row, timestamp) -> float, mm this hour
    fan_policy(state, soil, weather_row, timestamp) -> float in [0, 1]

``state`` is a dict snapshot of the plant *before* the current hour is
simulated (T_in, RH_in, VPD, soil_pct, depletion, day_index, ...), so
policies make same-hour decisions from the previous hour's observed state --
they cannot see the future.

``fan_policy`` defaults to None, meaning "no fan, always 0.0" -- every caller
written before the circulation-fan actuator existed (tests, scripts/validate.py,
older controller code) keeps running unmodified with byte-identical output,
since Polyhouse.step's own fan_frac default is likewise 0.0.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import pandas as pd

from sim.config import default_config
from sim.disease import AlternariaRisk, DiseaseModel
from sim.et0 import et0_hourly, kc_for_day
from sim.polyhouse import Polyhouse
from sim.soil import SoilBucket

VentPolicy = Callable[[dict[str, Any], SoilBucket, "pd.Series[Any]", pd.Timestamp], float]
IrrigationPolicy = Callable[[dict[str, Any], SoilBucket, "pd.Series[Any]", pd.Timestamp], float]
FanPolicy = Callable[[dict[str, Any], SoilBucket, "pd.Series[Any]", pd.Timestamp], float]


def run(
    weather_df: pd.DataFrame,
    vent_policy: VentPolicy,
    irrigation_policy: IrrigationPolicy,
    fan_policy: FanPolicy | None = None,
    config: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the coupled polyhouse simulation over an hourly weather record.

    Args:
        weather_df: Hourly weather, indexed by timestamp, with columns
            T_out (degC), RH_out (%), I_solar (W/m2), wind (m/s), rain (mm).
            (See data/fetch_weather.py.)
        vent_policy: Callable returning vent opening fraction, 0.0-1.0.
        irrigation_policy: Callable returning irrigation applied this hour, mm.
        fan_policy: Callable returning circulation-fan fraction, 0.0-1.0.
            Defaults to None (always 0.0, no fan) for backward compatibility.
        config: Nested config dict (as from sim.config.default_config()).
            Defaults to the project's config.yaml.
        show_progress: Print a lightweight progress indicator for long runs.

    Returns:
        DataFrame indexed by timestamp with columns: T_in, RH_in, VPD,
        ET_hr, ETc_hr, soil_pct, depletion, irrigation, rain, vent, fan,
        leaf_wet, condensed, fan_kWh, dsv_cumulative, alternaria_risk,
        runoff.
    """
    if fan_policy is None:
        fan_policy = lambda state, soil, row, ts: 0.0  # noqa: E731
    if config is None:
        config = default_config()

    substeps_per_hour = int(config["simulation"]["substeps_per_hour"])
    dt_seconds = float(config["simulation"]["dt_seconds"])
    elevation_m = float(config["location"]["elevation_m"])
    albedo = float(config["polyhouse"]["albedo"])
    spray_threshold = float(config["disease"]["spray_threshold_dsv"])
    soil_cfg = config["soil"]

    if len(weather_df) == 0:
        raise ValueError("weather_df is empty")

    first_row = weather_df.iloc[0]
    polyhouse = Polyhouse(
        config=config, t_in_init_c=float(first_row["T_out"]), rh_in_init_pct=float(first_row["RH_out"])
    )
    soil = SoilBucket(
        fc=float(soil_cfg["fc"]), wp=float(soil_cfg["wp"]), zr_m=float(soil_cfg["zr_m"]), p=float(soil_cfg["p"])
    )
    disease = DiseaseModel(spray_threshold_dsv=spray_threshold)
    alternaria = AlternariaRisk()

    n_hours = len(weather_df)
    run_start_date = weather_df.index[0].normalize()

    state: dict[str, Any] = {
        "T_in": polyhouse.t_in,
        "RH_in": None,
        "VPD": None,
        "soil_pct": soil.moisture_pct,
        "depletion": soil.d,
        "day_index": 0,
    }

    records: list[dict[str, Any]] = []
    t_wall_start = time.time()

    for i, (ts, row) in enumerate(weather_df.iterrows()):
        day_index = (ts.normalize() - run_start_date).days
        state["day_index"] = day_index

        vent_frac = min(max(float(vent_policy(state, soil, row, ts)), 0.0), 1.0)
        irrigation_mm = max(float(irrigation_policy(state, soil, row, ts)), 0.0)
        fan_frac = min(max(float(fan_policy(state, soil, row, ts)), 0.0), 1.0)

        et_hr_mm = 0.0
        condensed_hr_kg = 0.0
        fan_kwh_hr = 0.0
        wet_substeps = 0
        t_in_last = rh_in_last = vpd_last = None
        for _ in range(substeps_per_hour):
            out = polyhouse.step(
                t_out_c=float(row["T_out"]),
                rh_out_pct=float(row["RH_out"]),
                i_solar_wm2=float(row["I_solar"]),
                wind_ms=float(row["wind"]),
                vent_frac=vent_frac,
                fan_frac=fan_frac,
                dt_seconds=dt_seconds,
            )
            et_hr_mm += out["ET_mm"]
            condensed_hr_kg += out["condensed"]
            fan_kwh_hr += out["fan_kWh"]
            if out["leaf_wet"]:
                wet_substeps += 1
            t_in_last, rh_in_last, vpd_last = out["T_in"], out["RH_in"], out["VPD"]

        # Hourly leaf-wetness: wet for the majority of the constituent
        # 5-minute sub-steps (matches the disease model's hourly cadence).
        leaf_wet_hour = wet_substeps >= (substeps_per_hour / 2.0)

        et0_mm = float(
            et0_hourly(
                t_c=float(row["T_out"]),
                rh_pct=float(row["RH_out"]),
                i_solar_wm2=float(row["I_solar"]),
                wind_ms=float(row["wind"]),
                cloud_pct=float(row["cloud"]),
                elevation_m=elevation_m,
                albedo=albedo,
            )
        )
        kc = kc_for_day(day_index)
        etc_hr_mm = kc * et0_mm

        depletion = soil.step(etc_mm=etc_hr_mm, irrigation_mm=irrigation_mm, rain_mm=float(row["rain"]))

        disease.step(t_in_c=t_in_last, leaf_wet=leaf_wet_hour, end_of_day=(ts.hour == 23))
        alternaria.step(t_in_c=t_in_last, leaf_wet=leaf_wet_hour, end_of_day=(ts.hour == 23))

        state = {
            "T_in": t_in_last,
            "RH_in": rh_in_last,
            "VPD": vpd_last,
            "soil_pct": soil.moisture_pct,
            "depletion": depletion,
            "day_index": day_index,
        }

        records.append(
            {
                "timestamp": ts,
                "T_in": t_in_last,
                "RH_in": rh_in_last,
                "VPD": vpd_last,
                "ET_hr": et_hr_mm,
                "ETc_hr": etc_hr_mm,
                "soil_pct": soil.moisture_pct,
                "depletion": depletion,
                "irrigation": irrigation_mm,
                "rain": float(row["rain"]),
                "vent": vent_frac,
                "fan": fan_frac,
                "leaf_wet": leaf_wet_hour,
                "condensed": condensed_hr_kg,
                "fan_kWh": fan_kwh_hr,
                "dsv_cumulative": disease.cumulative_dsv,
                "alternaria_risk": alternaria.cumulative_risk,
                "runoff": soil.runoff_mm,
            }
        )

        if show_progress and n_hours > 1 and (i % max(1, n_hours // 20) == 0 or i == n_hours - 1):
            frac = (i + 1) / n_hours
            elapsed = time.time() - t_wall_start
            sys.stdout.write(
                f"\r  engine.run: {frac:6.1%}  ({i + 1}/{n_hours} hours, {elapsed:5.1f}s elapsed)"
            )
            sys.stdout.flush()

    if show_progress and n_hours > 1:
        sys.stdout.write("\n")

    disease.finalize()
    alternaria.finalize()
    if records:
        records[-1]["dsv_cumulative"] = disease.cumulative_dsv
        records[-1]["alternaria_risk"] = alternaria.cumulative_risk

    return pd.DataFrame.from_records(records).set_index("timestamp")
