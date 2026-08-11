"""Two-regime, multi-year controller comparison experiment.

Runs the three controllers in controllers/ (fixed schedule, reactive
threshold, receding-horizon MPC over vent+fan) on the identical physical
plant (sim.engine.run), across every complete year the cached weather record
covers, for TWO separate climate regimes -- never pooled together, since
they are physically different problems:

    Regime A "Monsoon"     Jun 1 - Aug 29   High humidity, disease-dominated.
    Regime B "Dry season"  Nov 1 - Jan 29   Low humidity, water/VPD-dominated
                                             (the actual rabi tomato window
                                             in Assam; spans into the
                                             following calendar year).

For each regime, every controller is run once per available year and the
per-year results are aggregated to mean +/- sd -- a result that only holds
in one year is not reported as a result. Requires data/guwahati_<year>.csv
to already be cached (python data/fetch_weather.py); does not touch the
network.

Outputs:
    results/regime_A_raw.csv, results/regime_B_raw.csv
        One row per (year, controller): every metric, not aggregated.
    results/regime_A_summary.csv, results/regime_B_summary.csv
        One row per controller: mean +/- sd across years, for the metrics
        listed in CLAUDE.md's acceptance table (water, leaf-wet hours,
        Alternaria risk, Wallin DSV, % hours in the *achievable* VPD band,
        vent actuations, fan kWh).
    figures/regime_comparison.png
        Mean +/- sd bars per controller, both regimes, three headline
        metrics (water, leaf-wet hours, fan kWh).

The MPC controller's objective weights are fixed constants here (see
MPC_WEIGHTS below), chosen once by scripts/sweep_mpc_weights.py on one year
of Regime A and used unchanged for every year and both regimes -- per
explicit direction, weights are not retuned per-regime. See CLAUDE.md's
"MPC objective retune" section for the justification.

Usage:
    python scripts/experiment.py
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

from controllers.fixed import FixedController  # noqa: E402
from controllers.mpc import MPCController  # noqa: E402
from controllers.threshold import ThresholdController  # noqa: E402
from fetch_weather import load_multi_year_weather  # noqa: E402
from sim.engine import run  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

VPD_LOW_KPA = 0.8
VPD_HIGH_KPA = 1.2

# Chosen once by scripts/sweep_mpc_weights.py's 36-combo grid on one year of
# Regime A (results/weight_sweep_pareto.csv) -- the only two combinations on
# the water/leaf-wet-hours/actuations Pareto front where fan_energy_weight=0
# beat the Threshold baseline (990 leaf-wet hours, 424 actuations) on BOTH
# leaf-wet hours and actuations simultaneously; every fan_energy_weight=0.02
# combo tested either lost badly on actuations or overcorrected the fan off
# and regressed leaf-wet hours back above Threshold's. This is also
# controllers.mpc.MPCController's class default now, so every caller uses it
# unless it explicitly overrides -- kept spelled out here for a
# self-documenting, reproducible experiment script. See CLAUDE.md's "MPC
# objective retune" section for the full Pareto table and rationale.
MPC_WEIGHTS: dict[str, float] = dict(
    vpd_weight=2.0,
    leaf_wet_weight=0.04,
    w_switch=0.15,
    fan_energy_weight=0.0,
)

METRICS = [
    "water_L_per_m2",
    "leaf_wet_hours",
    "alternaria_risk",
    "wallin_dsv",
    "pct_hours_achievable_vpd_band",
    "vent_actuations",
    "fan_kWh",
]

CONTROLLER_COLORS = {"fixed": "#4c6b8a", "threshold": "#2a6f77", "mpc": "#d1495b"}


def _closed(state, soil, row, ts) -> float:
    return 0.0


def _full(state, soil, row, ts) -> float:
    return 1.0


def regime_a_windows(years: list[int]) -> dict[int, tuple[str, str]]:
    """Jun 1 - Aug 29, end exclusive -> 90 days, labeled by that year."""
    return {y: (f"{y}-06-01", f"{y}-08-30") for y in years}


def regime_b_windows(years: list[int]) -> dict[int, tuple[str, str]]:
    """Nov 1 - Jan 29 (spanning into year+1), end exclusive -> 90 days.

    Labeled by the starting (November) year, e.g. 2021 means the
    2021-11-01..2022-01-29 dry season.
    """
    return {y: (f"{y}-11-01", f"{y + 1}-01-30") for y in years}


def available_years(weather: pd.DataFrame, windows: dict[int, tuple[str, str]], min_hours: int = 90 * 24) -> list[int]:
    years = []
    for y, (start, end) in windows.items():
        n = int(((weather.index >= start) & (weather.index < end)).sum())
        if n >= min_hours:
            years.append(y)
        else:
            print(f"  skipping {y} ({start}..{end}): only {n} hours cached (need {min_hours})")
    return years


def achievable_vpd_bounds(window: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-hour achievable VPD range: vent pinned at 0 and at 1, fan off.

    Same diagnostic used in scripts/diagnose_vent_authority.py and inside
    controllers/mpc.py's own objective -- reused here so the reported
    "% hours in achievable VPD band" metric means the same thing the MPC is
    actually optimizing against, for all three controllers alike.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res_lo = run(window, _closed, _closed, _closed, show_progress=False)
        res_hi = run(window, _full, _closed, _closed, show_progress=False)
    return res_lo["VPD"], res_hi["VPD"]


def pct_hours_achievable_vpd_band(res: pd.DataFrame, vpd_lo: pd.Series, vpd_hi: pd.Series) -> float:
    """Fraction of hours the controller's actual VPD lands in [0.8,1.2] kPa
    clipped into that hour's achievable range (see achievable_vpd_bounds).

    Caveat: vpd_lo/vpd_hi come from two *constant*-vent boundary runs, but
    VPD is a path-dependent state (it depends on the polyhouse's own recent
    history, not just the current hour's vent setting) -- so a real
    controller's own trajectory, which took neither constant-vent path, can
    occasionally land slightly outside the [vpd_lo(h), vpd_hi(h)] envelope
    those two boundary runs suggest. This is a deliberate, consistent
    approximation (the same one controllers/mpc.py's own objective uses),
    not a rigorous instantaneous bound -- treat it as a good proxy for
    "was the target achievable," not an exact constraint. Because of this,
    this metric is NOT guaranteed to be >= the naive fixed-band percentage
    -- an hour whose achievable ceiling sits entirely below the 0.8-1.2
    target collapses to a single achievable point, which a real trajectory
    essentially never hits exactly, correctly reading as close to a "miss"
    for that hour rather than a lenient pass.
    """
    lo = vpd_lo.reindex(res.index).to_numpy()
    hi = vpd_hi.reindex(res.index).to_numpy()
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    eff_low = np.clip(VPD_LOW_KPA, lo2, hi2)
    eff_high = np.clip(VPD_HIGH_KPA, lo2, hi2)
    vpd = res["VPD"].to_numpy()
    hit = (vpd >= eff_low) & (vpd <= eff_high)
    return float(np.mean(hit) * 100.0)


def make_controllers(window: pd.DataFrame) -> dict[str, Any]:
    return {
        "fixed": FixedController(),
        "threshold": ThresholdController(),
        "mpc": MPCController(weather_df=window, **MPC_WEIGHTS),
    }


def run_window_all_controllers(regime: str, year: int, start: str, end: str, weather_csv_dir: str) -> list[dict[str, Any]]:
    """Run all 3 controllers over one (regime, year) window. Runs in a worker process."""
    weather = load_multi_year_weather(data_dir=Path(weather_csv_dir))
    window = weather[(weather.index >= start) & (weather.index < end)]

    vpd_lo, vpd_hi = achievable_vpd_bounds(window)
    controllers = make_controllers(window)

    rows: list[dict[str, Any]] = []
    for name, ctrl in controllers.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = run(window, ctrl.vent_policy, ctrl.irrigation_policy, ctrl.fan_policy, show_progress=False)
        elapsed = time.time() - t0

        rows.append({
            "regime": regime,
            "year": year,
            "controller": name,
            "water_L_per_m2": float(res["irrigation"].sum()),
            "leaf_wet_hours": int(res["leaf_wet"].sum()),
            "alternaria_risk": int(res["alternaria_risk"].iloc[-1]),
            "wallin_dsv": float(res["dsv_cumulative"].iloc[-1]),
            "pct_hours_achievable_vpd_band": pct_hours_achievable_vpd_band(res, vpd_lo, vpd_hi),
            "vent_actuations": int((res["vent"].diff().abs() > 1e-9).sum()),
            "fan_kWh": float(res["fan_kWh"].sum()),
            "elapsed_s": round(elapsed, 1),
        })
    detail = ", ".join(f"{r['controller']}={r['elapsed_s']}s" for r in rows)
    print(f"  [{regime} {year}] done ({detail})")
    return rows


def run_all_regimes(
    regime_windows: dict[str, dict[int, tuple[str, str]]], weather: pd.DataFrame, max_workers: int = 10
) -> dict[str, pd.DataFrame]:
    """Run every (regime, year) window across both regimes in one shared
    process pool, so all tasks compete for cores together instead of one
    regime finishing before the next starts -- twice as fast on a machine
    with cores to spare (12 here) since neither regime alone needs all of
    them.
    """
    tasks: list[tuple[str, int, str, str]] = []
    for regime, windows in regime_windows.items():
        years = available_years(weather, windows)
        print(f"\n{regime}: {len(years)} available year(s): {years}")
        tasks.extend((regime, y, *windows[y]) for y in years)

    rows_by_regime: dict[str, list[dict[str, Any]]] = {r: [] for r in regime_windows}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_window_all_controllers, r, y, start, end, str(REPO_ROOT / "data")): r
            for r, y, start, end in tasks
        }
        for fut in as_completed(futures):
            regime = futures[fut]
            rows_by_regime[regime].extend(fut.result())

    return {regime: pd.DataFrame(rows) for regime, rows in rows_by_regime.items()}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    agg = raw.groupby("controller")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_years"] = raw.groupby("controller").size()
    return agg.reset_index()


def plot_regime_comparison(summaries: dict[str, pd.DataFrame]) -> None:
    metrics = [("water_L_per_m2", "Water (L/m²)"), ("leaf_wet_hours", "Leaf-wet hours"), ("fan_kWh", "Fan energy (kWh)")]
    regimes = list(summaries.keys())
    fig, axes = plt.subplots(len(regimes), len(metrics), figsize=(4 * len(metrics), 4 * len(regimes)), squeeze=False)

    for r, regime in enumerate(regimes):
        df = summaries[regime].set_index("controller")
        controllers = [c for c in ["fixed", "threshold", "mpc"] if c in df.index]
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r][c]
            means = [df.loc[ctrl, f"{metric}_mean"] for ctrl in controllers]
            sds = [df.loc[ctrl, f"{metric}_std"] for ctrl in controllers]
            colors = [CONTROLLER_COLORS[ctrl] for ctrl in controllers]
            ax.bar(controllers, means, yerr=sds, color=colors, capsize=4)
            ax.set_title(f"{regime}: {label}")
            ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "regime_comparison.png", dpi=150)
    plt.close(fig)


def main() -> int:
    print("Loading cached multi-year weather record...")
    weather = load_multi_year_weather()
    print(f"  {len(weather)} hourly rows, {weather.index.min()} .. {weather.index.max()}")

    all_years = sorted(weather.index.year.unique())
    regime_windows = {
        "Regime A (Monsoon)": regime_a_windows(all_years),
        "Regime B (Dry season)": regime_b_windows(all_years),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_by_regime = run_all_regimes(regime_windows, weather)

    summaries: dict[str, pd.DataFrame] = {}
    for label, raw in raw_by_regime.items():
        key = "A" if label.startswith("Regime A") else "B"
        raw_path = RESULTS_DIR / f"regime_{key}_raw.csv"
        raw.to_csv(raw_path, index=False)
        print(f"Wrote {raw_path}")

        summary = summarize(raw)
        summary_path = RESULTS_DIR / f"regime_{key}_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
        print(summary.to_string(index=False))
        summaries[label] = summary

    plot_regime_comparison(summaries)
    print(f"\nWrote {FIGURES_DIR / 'regime_comparison.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
