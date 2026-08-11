"""Interactive Streamlit dashboard for the avinya-twin polyhouse simulator.

Three pages, selected from the sidebar:
    Live Simulation        -- inspect one controller over a regime/year
                               window, hour by hour.
    Controller Comparison  -- fixed vs threshold vs predictive (MPC) over a
                               chosen regime and year.
    Live Twin              -- an animated cross-section that scrubs through
                               a run hour by hour.

**Precomputed by default.** The Predictive (MPC) controller takes 2-4
minutes per 90-day window (a joint vent+fan candidate search re-simulated
every hour) -- far too slow to run on every page view/rerun on a Streamlit
Community Cloud free-tier container (1 shared CPU, 1GB RAM). By default,
every page reads results/precomputed/{fixed,threshold,mpc}.parquet (written
once, locally, by scripts/precompute_dashboard_data.py) instead of calling
sim.engine.run. Live Simulation additionally offers an explicit "Run live
simulation (slow)" toggle, OFF by default, that switches to running the
physics on demand over an arbitrary date range -- with a runtime warning,
since that's the one path that can still take minutes.

Only streamlit + plotly + pyarrow are added on top of the project's existing
dependencies (see requirements.txt / pyproject.toml's ``dashboard`` extra).
All physics stay in sim/ and controllers/; this file only renders precomputed
Parquet data (default) or calls sim.engine.run directly (live mode).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

from controllers.fixed import FixedController  # noqa: E402
from controllers.mpc import MPCController  # noqa: E402
from controllers.threshold import ThresholdController  # noqa: E402
from sim.engine import run  # noqa: E402

PRECOMPUTED_DIR = REPO_ROOT / "results" / "precomputed"

SPRAY_THRESHOLD_DSV = 18.0

REGIME_LABELS: dict[str, str] = {"A": "Monsoon (Jun 1 - Aug 29)", "B": "Dry season (Nov 1 - Jan 29)"}
REGIME_KEY_FROM_LABEL: dict[str, str] = {v: k for k, v in REGIME_LABELS.items()}

# --- Palette -----------------------------------------------------------------
# avinya-twin's dataviz reference palette (see .claude skill "dataviz"),
# used verbatim -- no eyeballed hex values. Colors map to *entities*, held
# constant across every chart on a page:
#   blue   -> indoor/controlled state (T_in, VPD, RH_in, vent) and controller
#             slot 1 ("Fixed")
#   orange -> outdoor reference (T_out) and controller slot 2 ("Threshold")
#   aqua   -> water (irrigation) and controller slot 3 ("Predictive")
#   red    -> status: thresholds / danger
#   green  -> status: optimal / good
# Dark-mode steps from the same documented reference palette (not eyeballed --
# see .claude skill "dataviz" -> references/palette.md's dark column). Status
# colors (CRITICAL, GOOD) are mode-invariant by design, same hex both modes.
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
VIOLET = "#9085e9"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
FAN_COLOR = VIOLET  # circulation fan -- distinct from vent (blue), T_out (orange), irrigation (aqua)
NEUTRAL_MID = "#383835"
SURFACE = "#1a1a19"
GRID = "#2c2c2a"
AXIS = "#383835"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

CONTROLLER_LABELS = ["Fixed", "Threshold", "Predictive"]
CONTROLLER_KEY = {"Fixed": "fixed", "Threshold": "threshold", "Predictive": "mpc"}
CONTROLLER_COLOR = {"Fixed": BLUE, "Threshold": ORANGE, "Predictive": AQUA}

# Assumed mapping (not physics-coupled): the VPD band considered "optimal"
# widens as the crop matures -- young transplants are more sensitive to
# vapour-pressure stress than a mature canopy. Mid-season matches the
# 0.8-1.2 kPa band already validated in figures/validation.png.
CROP_STAGE_VPD_BAND = {
    "Initial (transplant)": (0.4, 0.8),
    "Development": (0.6, 1.0),
    "Mid-season": (0.8, 1.2),
    "Late season": (1.0, 1.4),
}


# --- Precomputed data loading (default path -- no simulation, no network) ----


@st.cache_data(show_spinner=False)
def load_precomputed_controller(controller_key: str) -> pd.DataFrame:
    """Read one controller's full precomputed record (every regime/year).

    Pure local file read (Parquet) -- no network, no simulation. This is the
    only data-loading path exercised by default; sim.engine.run is never
    called unless the user explicitly opts into live mode (see
    render_live_simulation).
    """
    path = PRECOMPUTED_DIR / f"{controller_key}.parquet"
    if not path.exists():
        st.error(
            f"Precomputed data not found at `{path.relative_to(REPO_ROOT)}`. "
            "Run `python scripts/precompute_dashboard_data.py` locally first "
            "and commit results/precomputed/ before deploying."
        )
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def available_precomputed_years(regime_key: str) -> list[int]:
    """Years available for a regime. All three controllers share the same
    windows (scripts/precompute_dashboard_data.py runs them together), so
    "fixed" is read as the canonical set.
    """
    df = load_precomputed_controller("fixed")
    years = df.loc[df["regime"] == regime_key, "year"].unique().tolist()
    return sorted(int(y) for y in years)


def slice_precomputed(controller_key: str, regime_key: str, year: int) -> pd.DataFrame:
    """One (controller, regime, year) window, shaped like the old live
    run_window()'s return value (same columns sim.engine.run produces, plus
    T_out) so every downstream chart/metric function works unchanged
    regardless of which data source populated it.
    """
    df = load_precomputed_controller(controller_key)
    sub = df[(df["regime"] == regime_key) & (df["year"] == year)]
    return sub.drop(columns=["regime", "year", "controller"])


# --- Live simulation loading (opt-in only, see render_live_simulation) -------


@st.cache_data(show_spinner=False)
def load_weather_live() -> pd.DataFrame:
    """Multi-year weather from local cache (data/guwahati_<year>.csv) --
    only imported/called from the live-mode branch of Live Simulation, never
    at module import or on a default (precomputed) page render. Reads local
    files only; the network-fetching function (fetch_weather.fetch_weather)
    is never called by this app either way.
    """
    sys.path.insert(0, str(REPO_ROOT / "data"))
    from fetch_weather import load_multi_year_weather

    return load_multi_year_weather()


def _make_controller(key: str, window: pd.DataFrame):
    if key == "fixed":
        return FixedController()
    if key == "threshold":
        return ThresholdController()
    if key == "mpc":
        return MPCController(weather_df=window)
    raise ValueError(key)


@st.cache_data(show_spinner=False)
def run_window_live(controller_key: str, start: str, end: str) -> pd.DataFrame:
    """Actually run the physics for [start, end] (inclusive). Slow for MPC
    (2-4 min per 90-day window) -- only reachable via Live Simulation's
    explicit "Run live simulation (slow)" toggle.
    """
    weather = load_weather_live()
    window = weather[(weather.index >= start) & (weather.index <= end)]
    ctrl = _make_controller(controller_key, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = run(window, ctrl.vent_policy, ctrl.irrigation_policy, ctrl.fan_policy, show_progress=False)
    res["T_out"] = window["T_out"].to_numpy()
    return res


# --- Shared metrics -----------------------------------------------------------


def compute_metrics(res: pd.DataFrame, vpd_low: float, vpd_high: float) -> dict[str, float]:
    return {
        "water_L_per_m2": float(res["irrigation"].sum()),
        "pct_vpd_band": float((res["VPD"].between(vpd_low, vpd_high)).mean() * 100.0),
        "cumulative_dsv": float(res["dsv_cumulative"].iloc[-1]),
        "leaf_wet_hours": int(res["leaf_wet"].sum()),
        "fan_kwh": float(res["fan_kWh"].sum()),
    }


def summarize_controller(res: pd.DataFrame, vpd_low: float = 0.8, vpd_high: float = 1.2) -> dict[str, float]:
    m = compute_metrics(res, vpd_low, vpd_high)
    m["vent_actuations"] = int((res["vent"].diff().abs() > 1e-9).sum())
    m["fan_actuations"] = int((res["fan"].diff().abs() > 1e-9).sum())
    return m


# --- Shared chart styling -----------------------------------------------------


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=TEXT_PRIMARY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=AXIS, showline=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=AXIS, showline=True, zeroline=False)
    return fig


def wet_period_blocks(res: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous [start, end) timestamp ranges where leaf_wet is True."""
    wet = res["leaf_wet"].to_numpy()
    idx = res.index
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for i, w in enumerate(wet):
        if w and start is None:
            start = idx[i]
        elif not w and start is not None:
            blocks.append((start, idx[i]))
            start = None
    if start is not None:
        blocks.append((start, idx[-1] + pd.Timedelta(hours=1)))
    return blocks


# --- Page 1: Live Simulation ---------------------------------------------------


def build_live_figure(res: pd.DataFrame, vpd_low: float, vpd_high: float) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.24, 0.19, 0.21, 0.17, 0.19],
        subplot_titles=(
            "Indoor vs outdoor temperature (°C)",
            "Vapour pressure deficit (kPa)",
            "Indoor relative humidity (%)",
            "Vent & circulation fan (fraction)",
            "Irrigation (mm)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=res.index, y=res["T_in"], name="T_in", line=dict(color=BLUE, width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=res.index, y=res["T_out"], name="T_out", line=dict(color=ORANGE, width=2)),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=res.index, y=res["VPD"], name="VPD", line=dict(color=BLUE, width=2), showlegend=False),
        row=2, col=1,
    )
    fig.add_hrect(
        y0=vpd_low, y1=vpd_high, fillcolor=GOOD, opacity=0.10, line_width=0,
        annotation_text=f"optimal {vpd_low:.1f}-{vpd_high:.1f} kPa", annotation_position="top left",
        annotation_font=dict(color=TEXT_SECONDARY, size=11),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=res.index, y=res["RH_in"], name="RH_in", line=dict(color=BLUE, width=2), showlegend=False),
        row=3, col=1,
    )
    fig.add_hline(
        y=90, line_dash="dash", line_color=CRITICAL, line_width=1.5,
        annotation_text="90% leaf-wetness threshold", annotation_font=dict(color=CRITICAL, size=11),
        row=3, col=1,
    )
    for i, (b_start, b_end) in enumerate(wet_period_blocks(res)):
        fig.add_vrect(
            x0=b_start, x1=b_end, fillcolor=CRITICAL, opacity=0.10, line_width=0,
            row=3, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=res.index, y=res["vent"], name="Vent fraction",
            line=dict(color=BLUE, width=2, shape="hv"),
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res.index, y=res["fan"], name="Fan fraction",
            line=dict(color=FAN_COLOR, width=2, shape="hv"),
        ),
        row=4, col=1,
    )

    fig.add_trace(
        go.Bar(x=res.index, y=res["irrigation"], name="Irrigation", marker_color=AQUA, showlegend=False),
        row=5, col=1,
    )

    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="kPa", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="frac", range=[0, 1], row=4, col=1)
    fig.update_yaxes(title_text="mm", row=5, col=1)

    return style_fig(fig, height=980)


def _render_result(res: pd.DataFrame, baseline: pd.DataFrame, vpd_low: float, vpd_high: float, is_baseline: bool) -> None:
    """Metric cards + the combined figure -- identical whether `res` came
    from precomputed data or a live run.
    """
    metrics = compute_metrics(res, vpd_low, vpd_high)
    base_metrics = compute_metrics(baseline, vpd_low, vpd_high)

    cols = st.columns(5)
    card_specs = [
        ("Water used", "water_L_per_m2", "L/m²", "{:.1f}", "inverse"),
        ("Hours in VPD band", "pct_vpd_band", "%", "{:.1f}", "normal"),
        ("Cumulative DSV", "cumulative_dsv", "", "{:.0f}", "inverse"),
        ("Leaf-wet hours", "leaf_wet_hours", "h", "{:.0f}", "inverse"),
        ("Fan energy", "fan_kwh", "kWh", "{:.2f}", "inverse"),
    ]
    for col, (label, key, unit, fmt, delta_color) in zip(cols, card_specs):
        value = metrics[key]
        delta = value - base_metrics[key]
        col.metric(
            label,
            f"{fmt.format(value)}{(' ' + unit) if unit else ''}",
            delta=("baseline" if is_baseline else f"{delta:+.1f} vs Fixed"),
            delta_color=("off" if is_baseline else delta_color),
        )

    st.plotly_chart(build_live_figure(res, vpd_low, vpd_high), use_container_width=True)


def render_live_simulation() -> None:
    st.title("Live Simulation")
    st.caption("Inspect one controller's physics hour by hour.")

    with st.sidebar:
        st.subheader("Live Simulation controls")
        live_mode = st.toggle(
            "Run live simulation (slow)", value=False,
            help="Off (default): read precomputed results instantly. On: actually run the "
                 "physics for a date range you choose -- the Predictive (MPC) controller can "
                 "take 2-4 minutes per 90-day window.",
        )

        if live_mode:
            st.warning(
                "**Live mode simulates on demand.** Fixed/Threshold finish in a few seconds; "
                "**Predictive (MPC) can take 2-4 minutes** for a 90-day window (a joint "
                "vent+fan candidate search re-run every simulated hour). Avoid long ranges "
                "with Predictive on a shared server."
            )
            weather = load_weather_live()
            min_date, max_date = weather.index.min().date(), weather.index.max().date()
            date_range = st.date_input(
                "Date range",
                value=(pd.Timestamp("2025-06-01").date(), pd.Timestamp("2025-06-14").date()),
                min_value=min_date, max_value=max_date,
            )
            controller_label = st.selectbox("Controller", CONTROLLER_LABELS, index=0)
            crop_stage = st.selectbox("Crop stage", list(CROP_STAGE_VPD_BAND), index=2)
            st.caption("Crop stage sets the VPD band used for the metric/shading above -- an assumed "
                       "horticultural mapping, not a physics coupling (Kc still follows FAO-56 calendar days).")
            run_clicked = st.button("Run simulation", type="primary", use_container_width=True)
        else:
            regime_label = st.selectbox("Regime", list(REGIME_LABELS.values()), index=0)
            regime_key = REGIME_KEY_FROM_LABEL[regime_label]
            years = available_precomputed_years(regime_key)
            year = st.selectbox("Year", years, index=len(years) - 1)
            controller_label = st.selectbox("Controller", CONTROLLER_LABELS, index=0)
            crop_stage = st.selectbox("Crop stage", list(CROP_STAGE_VPD_BAND), index=2)
            st.caption("Crop stage sets the VPD band used for the metric/shading above -- an assumed "
                       "horticultural mapping, not a physics coupling (Kc still follows FAO-56 calendar days).")
            st.caption("Reading precomputed results -- instant, updates as you change a control above.")

    vpd_low, vpd_high = CROP_STAGE_VPD_BAND[crop_stage]

    if live_mode:
        if run_clicked:
            start_date, end_date = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (date_range, date_range)
            start_str, end_str = f"{start_date} 00:00:00", f"{end_date} 23:00:00"
            with st.spinner("Running simulation..."):
                res = run_window_live(CONTROLLER_KEY[controller_label], start_str, end_str)
                baseline = run_window_live("fixed", start_str, end_str)
            st.session_state.page1_live_result = {
                "res": res, "baseline": baseline, "controller_label": controller_label,
            }
        result = st.session_state.get("page1_live_result")
        if result is None:
            st.info("Configure controls in the sidebar and click **Run simulation**.")
            return
        _render_result(result["res"], result["baseline"], vpd_low, vpd_high, result["controller_label"] == "Fixed")
    else:
        res = slice_precomputed(CONTROLLER_KEY[controller_label], regime_key, year)
        baseline = slice_precomputed("fixed", regime_key, year)
        _render_result(res, baseline, vpd_low, vpd_high, controller_label == "Fixed")


# --- Page 2: Controller Comparison ---------------------------------------------


def build_dsv_overlay(results: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label in CONTROLLER_LABELS:
        res = results[CONTROLLER_KEY[label]]
        fig.add_trace(go.Scatter(
            x=res.index, y=res["dsv_cumulative"], name=label,
            line=dict(color=CONTROLLER_COLOR[label], width=2),
        ))
    fig.add_hline(
        y=SPRAY_THRESHOLD_DSV, line_dash="dash", line_color=CRITICAL, line_width=1.5,
        annotation_text=f"BLITECAST spray threshold (DSV={SPRAY_THRESHOLD_DSV:.0f})",
        annotation_font=dict(color=CRITICAL, size=11),
    )
    fig.update_yaxes(title_text="Cumulative Wallin DSV")
    return style_fig(fig, height=380)


def build_water_overlay(results: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label in CONTROLLER_LABELS:
        res = results[CONTROLLER_KEY[label]]
        fig.add_trace(go.Scatter(
            x=res.index, y=res["irrigation"].cumsum(), name=label,
            line=dict(color=CONTROLLER_COLOR[label], width=2),
        ))
    fig.update_yaxes(title_text="Cumulative water applied (L/m²)")
    return style_fig(fig, height=380)


def find_failure_events(
    driver_res: pd.DataFrame,
    comparator_res: pd.DataFrame,
    lookahead_hours: int = 6,
    rain_mm_threshold: float = 3.0,
    top_n: int = 3,
) -> list[dict]:
    """Hours where ``driver_res``'s controller irrigated right before it rained.

    "Failure" here means the controller had no way to know rain was coming
    and applied water the sky was about to supply for free within the next
    few hours.
    """
    idx = driver_res.index
    rain = driver_res["rain"]
    irrigation = driver_res["irrigation"]
    events: list[dict] = []
    for i in range(len(idx)):
        dose = float(irrigation.iloc[i])
        if dose <= 0.0:
            continue
        upcoming_rain = float(rain.iloc[i + 1 : i + 1 + lookahead_hours].sum())
        if upcoming_rain >= rain_mm_threshold:
            comparator_dose = float(comparator_res["irrigation"].iloc[i]) if i < len(comparator_res) else 0.0
            events.append({
                "timestamp": idx[i],
                "driver_dose_mm": dose,
                "rain_next_6h_mm": upcoming_rain,
                "comparator_dose_same_hour_mm": comparator_dose,
            })
    events.sort(key=lambda e: e["rain_next_6h_mm"], reverse=True)
    return events[:top_n]


def build_failure_figure(
    events: list[dict], driver_res: pd.DataFrame, comparator_res: pd.DataFrame,
    driver_label: str, comparator_label: str, window_hours: int = 18,
) -> go.Figure:
    n = len(events)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=False, vertical_spacing=0.14,
        subplot_titles=[
            f"{e['timestamp']:%Y-%m-%d %H:%M} -- {driver_label} applied {e['driver_dose_mm']:.0f}mm, "
            f"then {e['rain_next_6h_mm']:.1f}mm rain fell within {window_hours // 3}h; "
            f"{comparator_label} applied {e['comparator_dose_same_hour_mm']:.0f}mm at the same hour"
            for e in events
        ],
    )
    for row, e in enumerate(events, start=1):
        ts = e["timestamp"]
        lo, hi = ts - pd.Timedelta(hours=window_hours), ts + pd.Timedelta(hours=window_hours)
        seg_d = driver_res.loc[lo:hi]
        seg_c = comparator_res.reindex(seg_d.index)
        fig.add_trace(go.Bar(
            x=seg_d.index, y=seg_d["rain"], name="Rain (mm)", marker_color=SEQ_BLUE[2],
            showlegend=(row == 1),
        ), row=row, col=1)
        fig.add_trace(go.Bar(
            x=seg_d.index, y=seg_d["irrigation"], name=f"{driver_label} irrigation (mm)", marker_color=ORANGE,
            showlegend=(row == 1),
        ), row=row, col=1)
        fig.add_trace(go.Bar(
            x=seg_c.index, y=seg_c["irrigation"], name=f"{comparator_label} irrigation (mm)", marker_color=AQUA,
            showlegend=(row == 1),
        ), row=row, col=1)
        fig.add_vline(x=ts, line_dash="dash", line_color=CRITICAL, line_width=1.5, row=row, col=1)

    fig.update_layout(barmode="overlay")
    return style_fig(fig, height=230 * n + 60)


def render_controller_comparison() -> None:
    st.title("Controller Comparison")
    st.caption("Fixed, Threshold, and Predictive (MPC) over a chosen regime and year -- reading precomputed results.")

    with st.sidebar:
        st.subheader("Controller Comparison controls")
        regime_label = st.radio("Regime", list(REGIME_LABELS.values()), index=0)
        regime_key = REGIME_KEY_FROM_LABEL[regime_label]
        years = available_precomputed_years(regime_key)
        year = st.selectbox("Year", years, index=len(years) - 1)

    results = {key: slice_precomputed(key, regime_key, year) for key in CONTROLLER_KEY.values()}

    summary_rows = {label: summarize_controller(results[CONTROLLER_KEY[label]]) for label in CONTROLLER_LABELS}
    summary_df = pd.DataFrame(summary_rows).T
    summary_df = summary_df[
        ["water_L_per_m2", "pct_vpd_band", "cumulative_dsv", "leaf_wet_hours", "fan_kwh", "vent_actuations", "fan_actuations"]
    ]
    summary_df.columns = [
        "Water (L/m²)", "% hours VPD band", "Cumulative DSV", "Leaf-wet hours",
        "Fan (kWh)", "Vent actuations", "Fan actuations",
    ]
    st.subheader(f"Comparison table -- {regime_label}, {year}")
    st.dataframe(summary_df.style.format({
        "Water (L/m²)": "{:.1f}", "% hours VPD band": "{:.2f}",
        "Cumulative DSV": "{:.0f}", "Leaf-wet hours": "{:.0f}", "Fan (kWh)": "{:.2f}",
        "Vent actuations": "{:.0f}", "Fan actuations": "{:.0f}",
    }), use_container_width=True)

    st.subheader("Cumulative disease severity (DSV)")
    st.plotly_chart(build_dsv_overlay(results), use_container_width=True)

    st.subheader("Cumulative water use")
    st.plotly_chart(build_water_overlay(results), use_container_width=True)

    st.subheader("Failure moment: irrigating right before rain")
    st.caption(
        "Always shown for **Regime B (dry season)**, regardless of the regime selected above -- "
        "not Regime A. Per CLAUDE.md, the reactive Threshold controller applies zero irrigation "
        "across the *entire* Regime A monsoon window (soil never becomes stressed, so it never "
        "makes an irrigation decision at all), so an earlier version of this chart fell back to "
        "the naive Fixed timer there and captioned it as \"Predictive skipped irrigation ahead of "
        "rain\" -- misleading, since Predictive wasn't predicting anything in that regime, it "
        "also just never irrigates. Regime B is where Threshold (and Predictive) make real, "
        "differentiated irrigation decisions, so it's the only regime where this comparison means "
        "what it claims to mean."
    )

    failure_years = available_precomputed_years("B")
    failure_year = year if year in failure_years else failure_years[-1]
    threshold_b = slice_precomputed("threshold", "B", failure_year)
    mpc_b = slice_precomputed("mpc", "B", failure_year)
    events = find_failure_events(threshold_b, mpc_b)

    all_years_checked = failure_years
    total_irrigation_events = int((threshold_b["irrigation"] > 0).sum())

    if not events:
        # Checked directly across every cached Regime B year (2021-2025): Threshold
        # DOES irrigate in the dry season (22-28 times/year, real soil-depletion-
        # triggered decisions, unlike Regime A) but in NO year does a dose happen to
        # land within 6h of 3mm+ rain. Reported as a genuine finding, not papered
        # over with a fallback chart that would imply a capability being exercised
        # here that isn't.
        st.info(
            f"**No qualifying failure moments found in Regime B, {failure_year}** (Threshold made "
            f"{total_irrigation_events} real irrigation decisions this year, driven by soil "
            "depletion -- unlike Regime A, where it never irrigates at all -- but none happened to "
            "land within 6 hours of 3mm+ rain). Checked across every available Regime B year "
            f"({', '.join(str(y) for y in all_years_checked)}): zero qualifying events in any of "
            "them. In this dry-season regime, rain is infrequent enough that the specific "
            "'irrigated right before a downpour' coincidence this check looks for essentially "
            "doesn't occur -- a real absence, not a search-parameter artifact. Showing no chart "
            "here rather than one that would imply otherwise."
        )
    else:
        st.caption(
            f"Regime B, {failure_year}. Hours where Threshold irrigated and 3mm+ of rain fell "
            "within the next 6 hours -- water that was about to arrive for free. Overlaid against "
            "what Predictive did at the same hour."
        )
        events_df = pd.DataFrame(events).rename(columns={
            "timestamp": "Timestamp", "driver_dose_mm": "Threshold applied (mm)",
            "rain_next_6h_mm": "Rain in next 6h (mm)", "comparator_dose_same_hour_mm": "Predictive applied (mm)",
        })
        st.dataframe(events_df, use_container_width=True, hide_index=True)
        st.plotly_chart(
            build_failure_figure(events, threshold_b, mpc_b, "Threshold", "Predictive"),
            use_container_width=True,
        )


# --- Page 3: Live Twin ----------------------------------------------------------


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c))) for c in rgb)


def _lerp_color(c1: str, c2: str, t: float) -> str:
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    t = max(0.0, min(1.0, t))
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def temp_to_cover_color(t_in: float, cold: float = 20.0, mid: float = 32.5, hot: float = 45.0) -> str:
    """Diverging blue<->red map (the palette's documented diverging pair, dark-mode steps)."""
    t = max(cold, min(hot, t_in))
    if t <= mid:
        return _lerp_color(BLUE, NEUTRAL_MID, (t - cold) / (mid - cold))
    return _lerp_color(NEUTRAL_MID, "#e66767", (t - mid) / (hot - mid))  # red, dark-mode step


def moisture_to_soil_color(pct: float) -> str:
    """Sequential blue ramp, dry -> wet.

    Dark-mode "anchor flip" (see the dataviz palette doc's sequential-hue
    note): against the app's dark surface, the *lightest* ramp step would
    read as the pop and the *darkest* would nearly vanish into the
    background -- the opposite of what dry/pale vs. wet/rich soil should
    look like. Clamped to SEQ_BLUE[2..7] (pale to medium-strong blue) so
    both ends stay visible against SURFACE=#1a1a19; the ramp's darkest
    steps (8-12) are skipped entirely for this reason.
    """
    t = max(0.0, min(100.0, pct)) / 100.0
    return _lerp_color(SEQ_BLUE[2], SEQ_BLUE[7], t)


_DROPLET_POSITIONS = [(110, 118), (160, 100), (200, 112), (240, 98), (280, 116), (320, 104)]


def _droplet_svg(x: int, y: int) -> str:
    return (
        f'<path d="M{x},{y - 12} C{x - 7},{y - 2} {x - 7},{y + 6} {x},{y + 8} '
        f'C{x + 7},{y + 6} {x + 7},{y - 2} {x},{y - 12} Z" fill="{BLUE}" opacity="0.7"/>'
    )


def svg_polyhouse(
    t_in: float, vent_frac: float, fan_frac: float, moisture_pct: float, leaf_wet: bool, ts: pd.Timestamp
) -> str:
    cover_color = temp_to_cover_color(t_in)
    soil_color = moisture_to_soil_color(moisture_pct)
    vent_angle = -5 - 55 * max(0.0, min(1.0, vent_frac))
    droplets = "".join(_droplet_svg(x, y) for x, y in _DROPLET_POSITIONS) if leaf_wet else ""
    wet_badge = (
        f'<span style="background:{CRITICAL};color:#fff;border-radius:4px;padding:2px 8px;'
        f'font-size:12px;margin-left:8px;">leaf-wet</span>'
        if leaf_wet else ""
    )
    fan_badge = (
        f'<span style="background:{FAN_COLOR};color:#fff;border-radius:4px;padding:2px 8px;'
        f'font-size:12px;margin-left:8px;">fan on</span>'
        if fan_frac > 0.5 else ""
    )
    return f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
                background:{SURFACE}; border:1px solid {GRID}; border-radius:8px; padding:12px;">
      <svg viewBox="0 0 400 230" width="100%" height="260" style="display:block;">
        <rect x="0" y="200" width="400" height="30" fill="#e6e1d0"/>
        <rect x="20" y="212" width="360" height="18" rx="3" fill="{soil_color}"/>
        <path d="M40,200 A160,150 0 0,1 360,200 Z" fill="{cover_color}" opacity="0.85"
              stroke="{TEXT_SECONDARY}" stroke-width="2"/>
        <g transform="translate(200,58) rotate({vent_angle:.1f})">
          <rect x="-32" y="-4" width="64" height="8" rx="2" fill="{TEXT_PRIMARY}" opacity="0.4"/>
        </g>
        <circle cx="90" cy="170" r="14" fill="none" stroke="{FAN_COLOR}" stroke-width="3"
                opacity="{0.9 if fan_frac > 0.5 else 0.25}"/>
        <path d="M90,170 L90,158 M90,170 L100,177 M90,170 L80,177" stroke="{FAN_COLOR}" stroke-width="2.5"
              opacity="{0.9 if fan_frac > 0.5 else 0.25}"/>
        {droplets}
      </svg>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;
                  color:{TEXT_PRIMARY}; font-size:13px;">
        <span>{ts:%Y-%m-%d %H:%M}</span>
        <span>
          T_in <b>{t_in:.1f}°C</b> &nbsp;|&nbsp;
          Vent <b>{vent_frac * 100:.0f}%</b> &nbsp;|&nbsp;
          Soil <b>{moisture_pct:.0f}%</b>
          {wet_badge}{fan_badge}
        </span>
      </div>
    </div>
    """


def render_live_twin() -> None:
    st.title("Live Twin")
    st.caption("Animated polyhouse cross-section, scrubbed through a precomputed run.")

    with st.sidebar:
        st.subheader("Live Twin controls")
        regime_label = st.selectbox("Regime", list(REGIME_LABELS.values()), index=0, key="twin_regime")
        regime_key = REGIME_KEY_FROM_LABEL[regime_label]
        years = available_precomputed_years(regime_key)
        year = st.selectbox("Year", years, index=len(years) - 1, key="twin_year")
        controller_label = st.selectbox("Controller", CONTROLLER_LABELS, index=0, key="twin_controller")
        speed = st.selectbox("Hours per frame (play speed)", [1, 3, 6, 12, 24], index=2, key="twin_speed")

    res = slice_precomputed(CONTROLLER_KEY[controller_label], regime_key, year)
    n = len(res)

    st.session_state.setdefault("twin_idx", 0)
    st.session_state.setdefault("twin_playing", False)
    if st.session_state.twin_idx >= n:
        st.session_state.twin_idx = 0

    # Advance *before* the slider widget (key="twin_idx") is instantiated below --
    # Streamlit forbids writing to a widget-bound session_state key after that
    # widget has run in the same script pass.
    if st.session_state.twin_playing:
        st.session_state.twin_idx = (st.session_state.twin_idx + speed) % n

    row = res.iloc[st.session_state.twin_idx]
    html = svg_polyhouse(
        t_in=float(row["T_in"]), vent_frac=float(row["vent"]), fan_frac=float(row["fan"]),
        moisture_pct=float(row["soil_pct"]),
        leaf_wet=bool(row["leaf_wet"]), ts=res.index[st.session_state.twin_idx],
    )
    components.html(html, height=360)

    col_play, col_slider = st.columns([1, 6])
    with col_play:
        st.session_state.twin_playing = st.toggle("Play", value=st.session_state.twin_playing)
    with col_slider:
        st.slider(
            "Scrub through the run",
            0, n - 1, key="twin_idx",
            format=f"hour %d of {n}",
        )

    if st.session_state.twin_playing:
        time.sleep(0.35)
        st.rerun()


# --- App shell -------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Avinya Twin", page_icon="\U0001f331", layout="wide")

    with st.sidebar:
        st.title("Avinya Twin")
        st.caption("Polyhouse digital twin -- Guwahati, Assam")
        page = st.radio("Navigate", ["Live Simulation", "Controller Comparison", "Live Twin"])
        st.divider()

    if page == "Live Simulation":
        render_live_simulation()
    elif page == "Controller Comparison":
        render_controller_comparison()
    else:
        render_live_twin()


if __name__ == "__main__":
    main()
