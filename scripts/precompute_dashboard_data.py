"""Precompute every controller's output across both regimes and every
available year, once, for the Streamlit dashboard to read.

The dashboard used to call sim.engine.run on every page load (dashboard/app.py's
old run_window()). That's fine locally, but the MPC controller takes ~130-210s
per 90-day window (a joint 7-vent x 3-fan candidate search, re-simulated every
hour) -- far too slow for a Streamlit Community Cloud free-tier container (1
CPU, throttled, 1GB RAM) to run live on every page view or rerun. This script
does that work once, locally, and writes compact result files the deployed
app reads instead of simulating: dashboard/app.py's default (precomputed)
mode never calls sim.engine.run.

Writes one Parquet file per controller (results/precomputed/{fixed,threshold,
mpc}.parquet), each holding every (regime, year) window stacked together --
so the dashboard can slice by regime+year+controller without re-running
anything. Columns are trimmed to exactly what dashboard/app.py reads (see its
module docstring / the list below) and downcast to float32 -- dashboard charts
don't need float64 precision, and Parquet (unlike CSV) actually stores the
narrower dtype more compactly on disk.

Usage:
    python scripts/precompute_dashboard_data.py
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from controllers.fixed import FixedController  # noqa: E402
from controllers.mpc import MPCController  # noqa: E402
from controllers.threshold import ThresholdController  # noqa: E402
from experiment import (  # noqa: E402
    MPC_WEIGHTS,
    available_years,
    regime_a_windows,
    regime_b_windows,
)
from fetch_weather import load_multi_year_weather  # noqa: E402
from sim.engine import run  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "precomputed"

# Exactly the columns dashboard/app.py reads (see build_live_figure,
# compute_metrics, summarize_controller, find_failure_events, svg_polyhouse) --
# every other engine.run() output column (ETc_hr, ET_hr, depletion, condensed,
# runoff) is dropped. T_out isn't an engine output; it's the matching weather
# row, joined in here the same way dashboard/app.py's old run_window() did.
# alternaria_risk added so the Controller Comparison page can show it
# per-(regime,year) without a separate data source -- it's the
# climate-appropriate disease metric (see CLAUDE.md's Gate 6b finding).
KEEP_COLUMNS = [
    "T_in", "T_out", "RH_in", "VPD", "vent", "fan",
    "irrigation", "rain", "leaf_wet", "dsv_cumulative", "alternaria_risk", "fan_kWh", "soil_pct",
]

MAX_WORKERS = 10


def _make_controller(key: str, window: pd.DataFrame) -> Any:
    if key == "fixed":
        return FixedController()
    if key == "threshold":
        return ThresholdController()
    if key == "mpc":
        return MPCController(weather_df=window, **MPC_WEIGHTS)
    raise ValueError(key)


def run_one(regime_key: str, year: int, controller_key: str, start: str, end: str, data_dir: str) -> dict[str, Any]:
    """Run one (regime, year, controller) window. Runs in a worker process."""
    weather = load_multi_year_weather(data_dir=Path(data_dir))
    window = weather[(weather.index >= start) & (weather.index < end)]

    ctrl = _make_controller(controller_key, window)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = run(window, ctrl.vent_policy, ctrl.irrigation_policy, ctrl.fan_policy, show_progress=False)
    res["T_out"] = window["T_out"].to_numpy()
    res = res[KEEP_COLUMNS].copy()

    for col in res.columns:
        if col == "leaf_wet":
            res[col] = res[col].astype(bool)
        elif pd.api.types.is_float_dtype(res[col]):
            res[col] = res[col].astype("float32")
        elif pd.api.types.is_integer_dtype(res[col]):
            # dsv_cumulative is a small running int count (DiseaseModel.cumulative_dsv);
            # int64 by default, never remotely close to int16's +-32767 range.
            res[col] = res[col].astype("int16")

    res.insert(0, "controller", pd.Categorical([controller_key] * len(res)))
    res.insert(0, "year", pd.array([year] * len(res), dtype="int16"))
    res.insert(0, "regime", pd.Categorical([regime_key] * len(res)))

    elapsed = time.time() - t0
    print(f"  [{regime_key} {year} {controller_key}] {elapsed:.1f}s, {len(res)} rows")
    return {"regime": regime_key, "year": year, "controller": controller_key, "frame": res}


def main() -> int:
    print("Loading cached multi-year weather record...")
    weather = load_multi_year_weather()
    all_years = sorted(weather.index.year.unique())

    regime_windows = {
        "A": regime_a_windows(all_years),
        "B": regime_b_windows(all_years),
    }

    tasks: list[tuple[str, int, str, str, str]] = []
    for regime_key, windows in regime_windows.items():
        years = available_years(weather, windows)
        print(f"Regime {regime_key}: {len(years)} available year(s): {years}")
        for year in years:
            start, end = windows[year]
            for controller_key in ("fixed", "threshold", "mpc"):
                tasks.append((regime_key, year, controller_key, start, end))

    print(f"\nRunning {len(tasks)} (regime, year, controller) windows with {MAX_WORKERS} parallel workers...")
    t0 = time.time()
    frames: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(run_one, regime_key, year, controller_key, start, end, str(REPO_ROOT / "data")): (
                regime_key, year, controller_key,
            )
            for regime_key, year, controller_key, start, end in tasks
        }
        for fut in as_completed(futures):
            frames.append(fut.result())
    print(f"\nAll windows done in {time.time() - t0:.0f}s total.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for controller_key in ("fixed", "threshold", "mpc"):
        parts = [f["frame"] for f in frames if f["controller"] == controller_key]
        parts.sort(key=lambda df: (df["regime"].iloc[0], int(df["year"].iloc[0])))
        combined = pd.concat(parts)
        out_path = OUT_DIR / f"{controller_key}.parquet"
        combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=True)
        size_kb = out_path.stat().st_size / 1024
        print(f"Wrote {out_path} ({len(combined)} rows, {size_kb:.0f} KB)")

    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"\nTotal results/precomputed/ size: {total_bytes / 1024:.0f} KB ({total_bytes / 1024 / 1024:.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
