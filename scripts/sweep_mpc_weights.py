"""Coarse grid sweep over the MPC objective's weights.

Runs on ONE year of Regime A (2025-06-01..2025-08-29, the same monsoon
window as scripts/experiment.py's default year) per explicit direction: the
sweep picks a single weight set that is then used everywhere -- every year,
both regimes -- not retuned per-regime or per-year.

Reports water vs. leaf-wet-hours vs. actuations (the three axes that matter
post-fan: see CLAUDE.md's Problem-1 finding for why VPD/DSV are no longer
the primary comparison signal) and computes the Pareto front over those
three (all minimized). Writes:

    results/weight_sweep.csv         -- every combination tried
    results/weight_sweep_pareto.csv  -- the non-dominated subset

Runs combinations in parallel (ProcessPoolExecutor) since each is an
independent ~130s MPC run and this machine has cores to spare.

Usage:
    python scripts/sweep_mpc_weights.py
"""

from __future__ import annotations

import itertools
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controllers.mpc import MPCController  # noqa: E402
from sim.engine import run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WEATHER_CSV = REPO_ROOT / "data" / "guwahati_2025.csv"
RESULTS_DIR = REPO_ROOT / "results"

WINDOW_START = "2025-06-01"
WINDOW_END = "2025-08-30"  # exclusive -> 90 days, matches scripts/experiment.py Regime A

# Coarse grid: 3 x 3 x 2 x 2 = 36 combinations ("a few dozen").
VPD_WEIGHTS = [0.5, 1.0, 2.0]
LEAF_WET_WEIGHTS = [0.04, 0.08, 0.16]
W_SWITCHES = [0.05, 0.15]
FAN_ENERGY_WEIGHTS = [0.0, 0.02]

MAX_WORKERS = 10


def load_window() -> pd.DataFrame:
    df = pd.read_csv(WEATHER_CSV, index_col=0, parse_dates=True)
    return df[(df.index >= WINDOW_START) & (df.index < WINDOW_END)]


def run_one(params: dict[str, float]) -> dict[str, Any]:
    window = load_window()
    ctrl = MPCController(weather_df=window, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = run(window, ctrl.vent_policy, ctrl.irrigation_policy, ctrl.fan_policy, show_progress=False)
    return {
        **params,
        "water_L_per_m2": float(res["irrigation"].sum()),
        "leaf_wet_hours": int(res["leaf_wet"].sum()),
        "vent_actuations": int((res["vent"].diff().abs() > 1e-9).sum()),
        "fan_actuations": int((res["fan"].diff().abs() > 1e-9).sum()),
        "fan_kWh": float(res["fan_kWh"].sum()),
        "cumulative_dsv": float(res["dsv_cumulative"].iloc[-1]),
        "alternaria_risk": int(res["alternaria_risk"].iloc[-1]),
        "pct_vpd_band": float((res["VPD"].between(0.8, 1.2)).mean() * 100.0),
    }


def pareto_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """True where a row is not dominated by any other row on `cols` (all minimized)."""
    objs = df[cols].to_numpy()
    n = len(df)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if all(objs[j] <= objs[i]) and any(objs[j] < objs[i]):
                dominated[i] = True
                break
    return pd.Series([not d for d in dominated], index=df.index)


def main() -> int:
    combos = [
        dict(vpd_weight=v, leaf_wet_weight=l, w_switch=w, fan_energy_weight=f)
        for v, l, w, f in itertools.product(VPD_WEIGHTS, LEAF_WET_WEIGHTS, W_SWITCHES, FAN_ENERGY_WEIGHTS)
    ]
    print(f"Sweeping {len(combos)} weight combinations over {WINDOW_START}..{WINDOW_END} "
          f"with {MAX_WORKERS} parallel workers...")

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, c): c for c in combos}
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            print(f"  [{i}/{len(combos)}] done ({time.time() - t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "weight_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    df["total_actuations"] = df["vent_actuations"] + df["fan_actuations"]
    df["pareto"] = pareto_mask(df, ["water_L_per_m2", "leaf_wet_hours", "total_actuations"])
    pareto_df = df[df["pareto"]].sort_values("leaf_wet_hours")
    print(f"\nPareto front ({len(pareto_df)} of {len(df)} combos), water vs leaf_wet_hours vs total_actuations:")
    print(pareto_df.to_string(index=False))

    pareto_path = RESULTS_DIR / "weight_sweep_pareto.csv"
    pareto_df.to_csv(pareto_path, index=False)
    print(f"\nWrote {pareto_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
