"""Phase 8 validation runner.

Runs the 7-day June (monsoon) passive-policy simulation (vent fixed at 0.3,
no irrigation), evaluates gates V1-V8, prints a PASS/FAIL table with the
measured value against each bound, writes figures/validation.png and
figures/et_profile.png, and exits non-zero if any gate fails.

Usage:
    python scripts/validate.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.engine import run  # noqa: E402
from sim.polyhouse import Polyhouse  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WEATHER_CSV = REPO_ROOT / "data" / "guwahati_2025.csv"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"
VALIDATION_START = "2025-06-22"
VALIDATION_END = "2025-06-29"  # exclusive

# A small, deliberate palette: warm = inside (heated), cool = outside
# (ambient); one hue per series, no rainbow. Used consistently across both
# figures.
COLOR_T_IN = "#d1495b"  # warm red -- internal temperature
COLOR_T_OUT = "#4c6b8a"  # cool blue-gray -- outside temperature
COLOR_RH_IN = "#2a6f77"  # teal -- internal humidity
COLOR_WETNESS_LINE = "#8a8a8a"
COLOR_VPD = "#5c4d7d"  # purple -- vapour pressure deficit
COLOR_OPTIMAL_BAND = "#8fbf9f"
COLOR_ET = "#c98a2e"  # amber -- evapotranspiration


@dataclass
class GateResult:
    name: str
    description: str
    measured: str
    bounds: str
    passed: bool


def _passive_vent(state, soil, row, ts) -> float:
    return 0.3


def _no_irrigation(state, soil, row, ts) -> float:
    return 0.0


def load_validation_week() -> pd.DataFrame:
    if not WEATHER_CSV.exists():
        raise SystemExit(
            f"Weather CSV not found at {WEATHER_CSV}. Run: python data/fetch_weather.py"
        )
    df = pd.read_csv(WEATHER_CSV, index_col=0, parse_dates=True)
    week = df[(df.index >= VALIDATION_START) & (df.index < VALIDATION_END)]
    if len(week) != 7 * 24:
        raise SystemExit(
            f"Expected 168 hourly rows for {VALIDATION_START}..{VALIDATION_END}, got {len(week)}"
        )
    return week


def evaluate_gates(passive_res: pd.DataFrame, week: pd.DataFrame) -> list[GateResult]:
    gates: list[GateResult] = []

    day = passive_res[(passive_res.index.hour >= 11) & (passive_res.index.hour <= 15)]
    gap = (day["T_in"] - day["T_out"]).mean()
    gates.append(
        GateResult(
            "V1",
            "Daytime (11-15h) mean T_in - T_out",
            f"{gap:.2f} degC",
            "[4.0, 16.0] degC",
            4.0 <= gap <= 16.0,
        )
    )

    predawn = passive_res[(passive_res.index.hour >= 3) & (passive_res.index.hour <= 6)]
    mean_rh = predawn["RH_in"].mean()
    gates.append(
        GateResult(
            "V2",
            "Pre-dawn (03-06h) mean RH_in",
            f"{mean_rh:.2f} %",
            "> 92 %",
            mean_rh > 92.0,
        )
    )

    peak_et = passive_res["ET_hr"].max()
    gates.append(
        GateResult(
            "V3",
            "Peak hourly ET",
            f"{peak_et:.3f} mm/hr",
            "[0.25, 0.70] mm/hr",
            0.25 <= peak_et <= 0.70,
        )
    )

    daily = passive_res.groupby(passive_res.index.date)["ET_hr"].sum()
    gates.append(
        GateResult(
            "V4",
            "Daily total ET (min..max over the week)",
            f"{daily.min():.2f}..{daily.max():.2f} mm/day",
            "[2.5, 6.0] mm/day",
            bool(daily.min() >= 2.5 and daily.max() <= 6.0),
        )
    )

    t_min, t_max = passive_res["T_in"].min(), passive_res["T_in"].max()
    gates.append(
        GateResult(
            "V5",
            "T_in physical plausibility band",
            f"[{t_min:.2f}, {t_max:.2f}] degC",
            "within [5, 55] degC",
            bool(t_min >= 5.0 and t_max <= 55.0),
        )
    )

    night = passive_res[(passive_res.index.hour >= 20) | (passive_res.index.hour <= 5)]
    frac_wet = night["leaf_wet"].mean()
    gates.append(
        GateResult(
            "V6",
            "Fraction of night hours leaf_wet",
            f"{frac_wet:.2f}",
            "> 0.5",
            frac_wet > 0.5,
        )
    )

    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    mean_t_in, mean_rh_in = [], []
    for level in levels:
        res = run(week, lambda s, so, r, t, v=level: v, _no_irrigation, show_progress=False)
        d = res[(res.index.hour >= 11) & (res.index.hour <= 15)]
        mean_t_in.append(d["T_in"].mean())
        mean_rh_in.append(d["RH_in"].mean())
    t_monotonic = all(a > b for a, b in zip(mean_t_in, mean_t_in[1:]))
    rh_monotonic = all(a > b for a, b in zip(mean_rh_in, mean_rh_in[1:]))
    gates.append(
        GateResult(
            "V7",
            f"Monotonic cooling/drying vs vent_frac {levels}",
            f"T_in={[round(float(x), 1) for x in mean_t_in]}, RH_in={[round(float(x), 1) for x in mean_rh_in]}",
            "both strictly decreasing",
            bool(t_monotonic and rh_monotonic),
        )
    )

    ph = Polyhouse(t_in_init_c=35.0, rh_in_init_pct=60.0)
    out = None
    for _ in range(12 * 48):
        out = ph.step(t_out_c=25.0, rh_out_pct=95.0, i_solar_wm2=0.0, wind_ms=1.0, vent_frac=0.3)
    diff = abs(out["T_in"] - 25.0)
    gates.append(
        GateResult(
            "V8",
            "No-solar steady-state convergence to T_out",
            f"|T_in-T_out| = {diff:.3f} degC",
            "< 0.5 degC",
            diff < 0.5,
        )
    )

    return gates


def write_gate_results_json(gates: list[GateResult]) -> Path:
    """Serialize the gate table to results/gate_results.json for the
    dashboard's Validation & Limitations page to read -- so that page shows
    the real gate table this script actually produced, not numbers typed
    into dashboard/app.py by hand and left to drift out of sync.
    """
    payload = {
        "window_start": VALIDATION_START,
        "window_end": VALIDATION_END,
        "gates": [{**asdict(g), "passed": bool(g.passed)} for g in gates],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "gate_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_gate_table(gates: list[GateResult]) -> None:
    name_w = max(len(g.name) for g in gates)
    desc_w = max(len(g.description) for g in gates)
    meas_w = max(len(g.measured) for g in gates)
    bound_w = max(len(g.bounds) for g in gates)

    header = f"{'Gate':<{name_w}}  {'Description':<{desc_w}}  {'Measured':<{meas_w}}  {'Bounds':<{bound_w}}  Result"
    print(header)
    print("-" * len(header))
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(
            f"{g.name:<{name_w}}  {g.description:<{desc_w}}  {g.measured:<{meas_w}}  "
            f"{g.bounds:<{bound_w}}  {status}"
        )


def plot_validation_figure(passive_res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(passive_res.index, passive_res["T_out"], color=COLOR_T_OUT, lw=2, label="T_out (outside)")
    ax.plot(passive_res.index, passive_res["T_in"], color=COLOR_T_IN, lw=2, label="T_in (inside)")
    ax.set_ylabel("Temperature (degC)")
    ax.set_title("7-day passive-policy validation run (vent=0.3, no irrigation)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(passive_res.index, passive_res["RH_in"], color=COLOR_RH_IN, lw=2, label="RH_in")
    ax.axhline(90.0, color=COLOR_WETNESS_LINE, lw=1.5, ls="--", label="90% leaf-wetness threshold")
    ax.set_ylabel("Relative humidity (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axhspan(0.8, 1.2, color=COLOR_OPTIMAL_BAND, alpha=0.35, label="0.8-1.2 kPa optimal band")
    ax.plot(passive_res.index, passive_res["VPD"], color=COLOR_VPD, lw=2, label="VPD")
    ax.set_ylabel("VPD (kPa)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "validation.png", dpi=150)
    plt.close(fig)


def plot_et_profile(passive_res: pd.DataFrame) -> None:
    diurnal = passive_res.groupby(passive_res.index.hour)["ET_hr"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diurnal.index, diurnal.to_numpy(), color=COLOR_ET, lw=2.5, marker="o", markersize=4)
    ax.fill_between(diurnal.index, 0, diurnal.to_numpy(), color=COLOR_ET, alpha=0.15)
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("Mean hourly ET (mm/hr)")
    ax.set_title("Mean diurnal transpiration profile (7-day validation window)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "et_profile.png", dpi=150)
    plt.close(fig)


def main() -> int:
    week = load_validation_week()
    print(f"Running passive-policy validation over {VALIDATION_START}..{VALIDATION_END} "
          f"({len(week)} hourly rows)...")
    passive_res = run(week, _passive_vent, _no_irrigation, show_progress=True)
    passive_res = passive_res.copy()
    passive_res["T_out"] = week["T_out"].to_numpy()

    gates = evaluate_gates(passive_res, week)
    print()
    print_gate_table(gates)

    plot_validation_figure(passive_res)
    plot_et_profile(passive_res)
    print(f"\nWrote {FIGURES_DIR / 'validation.png'}")
    print(f"Wrote {FIGURES_DIR / 'et_profile.png'}")

    json_path = write_gate_results_json(gates)
    print(f"Wrote {json_path}")

    n_fail = sum(1 for g in gates if not g.passed)
    if n_fail:
        print(f"\n{n_fail} gate(s) FAILED.")
        return 1
    print("\nAll gates PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
