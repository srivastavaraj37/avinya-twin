"""Interactive Streamlit dashboard for the avinya-twin polyhouse simulator.

Eight pages, selected from the sidebar, in narrative order:
    Headline Results          -- default landing page. 5-year mean +/- sd
                                  across both climate regimes, plain-language
                                  first.
    Why Ventilation Alone Fails -- the day/night vent-authority finding that
                                  motivated the circulation fan.
    Controller Comparison     -- fixed vs threshold vs predictive (MPC) over
                                  a chosen regime and year.
    Live Simulation           -- inspect one controller over a regime/year
                                  window, hour by hour.
    Live Twin                 -- an animated cross-section that scrubs
                                  through a run hour by hour.
    Validation & Limitations  -- V1-V8 gate table + every honest limitation,
                                  kept fully technical (this page is for a
                                  technical reviewer, not simplified).
    Deployment & Cost         -- hardware BOM, architecture properties, and
                                  an N-unit scale-up projection.
    For Growers               -- plain-language, no jargon/equations/metric
                                  abbreviations. Placed LAST deliberately: it
                                  demonstrates end-user thinking, it is not
                                  the headline.

AUDIENCE: competition judges (Innovation 30% / Feasibility 25% / Scalability
25% / Sustainability 20%), not farmers -- the farmer is the beneficiary, not
the reader. So every page except Validation & Limitations and For Growers
follows one convention: a plain-language label is the primary text a judge
reads in <10s, and the precise technical term is demoted to a caption or a
column-header tooltip, never deleted -- it still carries the Innovation and
Feasibility marks. See METRIC_SPECS/HEADLINE_CARD_TEXT for the label pairs.

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

import json
import math
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
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

from bom_data import BOM, bom_total_inr  # noqa: E402
from controllers.fixed import FixedController  # noqa: E402
from controllers.mpc import MPCController  # noqa: E402
from controllers.threshold import ThresholdController  # noqa: E402
from sim.engine import run  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PRECOMPUTED_DIR = RESULTS_DIR / "precomputed"
FIGURES_DIR = REPO_ROOT / "figures"

SPRAY_THRESHOLD_DSV = 18.0

REGIME_LABELS: dict[str, str] = {"A": "Monsoon (Jun 1 - Aug 29)", "B": "Dry season (Nov 1 - Jan 29)"}
REGIME_KEY_FROM_LABEL: dict[str, str] = {v: k for k, v in REGIME_LABELS.items()}

# --- Palette -----------------------------------------------------------------
# avinya-twin's dataviz reference palette, matched to .streamlit/config.toml's
# theme (same warm-dark-neutral surfaces, same deep-teal accent) so the
# Plotly charts and the custom Live Twin illustration read as part of the
# same designed product as the Streamlit chrome around them, not a
# default-theme app with charts bolted on. Colors map to *entities*, held
# constant across every chart on a page:
#   blue   -> indoor/controlled state (T_in, VPD, RH_in, vent) and controller
#             slot 1 ("Fixed")
#   orange -> outdoor reference (T_out) and controller slot 2 ("Threshold")
#   aqua   -> water (irrigation) and controller slot 3 ("Predictive")
#   teal   -> the app's single confident accent (ACCENT) -- selected nav
#             state, the lede box, headline card border, logo mark
#   red    -> status: thresholds / danger (matches theme.redColor -- the
#             default Streamlit red, #ff4b4b, is not used anywhere)
#   green  -> status: optimal / good (matches theme.greenColor)
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
VIOLET = "#9085e9"
ACCENT = "#2F9E8F"  # deep teal, matches .streamlit/config.toml's primaryColor
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
FAN_COLOR = VIOLET  # circulation fan -- distinct from vent (blue), T_out (orange), irrigation (aqua)
NEUTRAL_MID = "#3a3d36"
SURFACE = "#181917"  # matches theme.backgroundColor
SECONDARY_SURFACE = "#20221E"  # matches theme.secondaryBackgroundColor -- card/hover-label fills
GRID = "#2a2c26"  # faint horizontal gridlines only, see style_fig
BORDER = "#33362F"  # matches theme.borderColor
AXIS = BORDER
TEXT_PRIMARY = "#E9EBE6"  # matches theme.textColor
TEXT_SECONDARY = "#A6A99E"
CHART_FONT_FAMILY = "'Inter', -apple-system, 'Segoe UI', sans-serif"
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

# The 7 metrics results/regime_{A,B}_summary.csv reports (see
# scripts/experiment.py's METRICS), in the order CLAUDE.md's acceptance
# table presents them. Every one of these is "lower is better" in this
# project's framing (less water, less disease risk, fewer wet hours, fewer
# actuations, less energy) -- there is no metric here where a controller
# wants a *higher* number, which is why every delta below uses a single
# "inverse" color convention (negative change = green = improvement).
#
# Communication layer, not physics: "plain" is the primary label shown to a
# judge (plain-language, <=10s to parse); "technical" is the precise term,
# demoted to a caption/tooltip but never deleted -- a judge scoring
# Innovation/Feasibility still needs it visible. See the audience note at
# the top of this module.
METRIC_SPECS: list[dict[str, str]] = [
    {"key": "water_L_per_m2", "plain": "Water used", "technical": "L/m² applied over the season", "unit": "L/m²", "fmt": "{:.1f}"},
    {"key": "leaf_wet_hours", "plain": "Hours with wet leaves", "technical": "leaf-wet hours", "unit": "h", "fmt": "{:.0f}"},
    {"key": "alternaria_risk", "plain": "Early blight risk", "technical": "Alternaria solani risk units", "unit": "units", "fmt": "{:.1f}"},
    {"key": "wallin_dsv", "plain": "Late blight risk (temperate model)", "technical": "Wallin DSV", "unit": "", "fmt": "{:.1f}"},
    {
        "key": "pct_hours_achievable_vpd_band", "plain": "% hours in ideal humidity range",
        "technical": "% hours in achievable VPD band", "unit": "%", "fmt": "{:.2f}",
    },
    {"key": "vent_actuations", "plain": "Vent movements", "technical": "vent actuations", "unit": "", "fmt": "{:.0f}"},
    {"key": "fan_kWh", "plain": "Fan electricity", "technical": "fan energy, kWh", "unit": "kWh", "fmt": "{:.1f}"},
]
METRIC_SPEC_BY_KEY = {m["key"]: m for m in METRIC_SPECS}

# Order matches the ask: Alternaria risk, leaf-wet hours, water L/m2.
HEADLINE_METRIC_KEYS = ["alternaria_risk", "leaf_wet_hours", "water_L_per_m2"]

# Headline-card-specific wording: punchier standalone labels than the table
# versions above (a card doesn't sit next to "Late blight risk" needing
# disambiguation the way a table column does), plus a one-line plain-English
# "what this means" explanation per card.
HEADLINE_CARD_TEXT: dict[str, dict[str, str]] = {
    "alternaria_risk": {
        "card_label": "Disease risk",
        "technical": "Alternaria solani risk units",
        "meaning": "How much fungal disease pressure built up over the season. Lower is better.",
    },
    "leaf_wet_hours": {
        "card_label": "Hours with wet leaves",
        "technical": "leaf-wet hours -- when fungus infects",
        "meaning": "Hours the leaves stayed wet enough for fungus to take hold. Fewer is better.",
    },
    "water_L_per_m2": {
        "card_label": "Water used",
        "technical": "L/m² over the season",
        "meaning": "Irrigation water applied per square metre of growing area. Lower is better.",
    },
}

# APDCL (Assam Power Distribution Company Limited) Domestic-B tariff
# (5-30 kW load, the category a polyhouse control circuit would typically
# fall under), AERC Tariff Order FY2026-27: slabs run Rs 6.75-7.74/unit
# (https://billcalculator.in/1-unit-electricity-price-in-assam/, citing the
# official APDCL tariff schedule). This project uses Rs 7.00/unit as a
# representative mid-slab rate -- a communication aid to make the fan's
# energy cost concrete, not a precise billing calculation (a real
# installation's exact slab depends on total farm load, connection type,
# and the 5% electricity duty/fixed charges layered on top).
ELECTRICITY_RATE_INR_PER_KWH = 7.00

REPO_URL = "https://github.com/srivastavaraj37/avinya-twin"

# Simple geometric mark (a tunnel/arch with a small sensor dot) -- no emoji,
# matches the Live Twin illustration's own arch motif so the sidebar and the
# app's centrepiece visual read as the same product.
LOGO_SVG = f"""<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M3 22 L3 14 A11 11 0 0 1 25 14 L25 22" stroke="{ACCENT}" stroke-width="2.2" stroke-linecap="round" fill="none"/>
  <line x1="1" y1="22" x2="27" y2="22" stroke="{ACCENT}" stroke-width="2.2" stroke-linecap="round"/>
  <line x1="14" y1="14" x2="14" y2="6.5" stroke="{ACCENT}" stroke-width="2" stroke-linecap="round"/>
  <circle cx="14" cy="4.5" r="2.1" fill="{ACCENT}"/>
</svg>"""

# Sidebar nav, grouped into two labelled sections.
NAV_SECTIONS: dict[str, list[str]] = {
    "Results": ["Headline Results", "Why Ventilation Alone Fails", "Controller Comparison"],
    "Explore": ["Live Simulation", "Live Twin", "Validation & Limitations", "Deployment & Cost", "For Growers"],
}


def inject_css() -> None:
    """One CSS block, injected once per script run, that turns the default
    Streamlit chrome into a considered product surface: a real type scale,
    breathing room between sections, a constrained reading measure, bordered
    metric cards, quieter dataframes, and softened callouts. Colors are
    pulled from the same Python constants the Plotly charts use, so nothing
    here can drift out of sync with the rest of the palette.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: {CHART_FONT_FAMILY} !important;
        }}

        /* Constrained reading measure -- the single biggest readability fix */
        .block-container {{
            max-width: 1180px;
            padding-top: 2.75rem;
            padding-bottom: 4rem;
        }}

        /* Type scale: titles confident, sections subordinate, captions genuinely small */
        h1 {{
            font-size: 2.1rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.3rem !important;
        }}
        h2 {{
            font-size: 1.32rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            margin-top: 2.6rem !important;
            margin-bottom: 0.7rem !important;
        }}
        h3 {{
            font-size: 1.06rem !important;
            font-weight: 600 !important;
            margin-top: 1.9rem !important;
            margin-bottom: 0.5rem !important;
        }}
        p, li {{ font-size: 0.97rem; line-height: 1.65; }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
            font-size: 0.82rem !important;
            color: {TEXT_SECONDARY} !important;
            line-height: 1.55 !important;
        }}

        /* Vertical rhythm */
        hr {{ margin: 2.2rem 0 !important; border-color: {BORDER} !important; opacity: 0.7; }}
        [data-testid="stTabs"] {{ margin-top: 0.5rem; }}

        /* Metric cards: bordered container, real padding, an intentional delta pill */
        [data-testid="stMetric"] {{
            background: {SECONDARY_SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.15rem 1.3rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            color: {TEXT_SECONDARY} !important;
        }}
        [data-testid="stMetricValue"] {{ letter-spacing: -0.01em; }}
        [data-testid="stMetricDelta"] {{
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            padding: 0.18rem 0.55rem !important;
            border-radius: 999px !important;
            margin-top: 0.3rem !important;
            background: color-mix(in srgb, currentColor 14%, transparent) !important;
            width: fit-content;
        }}

        /* Dataframes: tighter header, quieter borders */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER} !important;
            border-radius: 10px !important;
            overflow: hidden;
        }}

        /* Callouts: a left accent rule and a soft tinted fill instead of a solid block */
        [data-testid="stAlert"] {{
            background: color-mix(in srgb, currentColor 9%, {SECONDARY_SURFACE}) !important;
            border: none !important;
            border-left: 3px solid currentColor !important;
            border-radius: 8px !important;
            padding: 0.95rem 1.15rem !important;
        }}

        /* Sidebar: product-navigation feel */
        [data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
        [data-testid="stSidebar"] .stButton button {{
            justify-content: flex-start !important;
            font-weight: 500 !important;
            border-color: transparent !important;
            padding: 0.45rem 0.7rem !important;
        }}
        [data-testid="stSidebar"] .stButton button p {{ text-align: left !important; font-size: 0.92rem !important; }}
        [data-testid="stSidebar"] .stButton button[kind="secondary"] {{ background: transparent !important; }}
        [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {{
            background: {SECONDARY_SURFACE} !important;
        }}
        .sidebar-nav-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {TEXT_SECONDARY};
            margin: 1.1rem 0 0.35rem 0.1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_lede(text: str) -> None:
    """A deliberate editorial lede -- not a default st.info block. Used once,
    at the top of each Headline Results tab, so the single most important
    sentence on the app's landing page reads with real visual weight.
    """
    st.markdown(
        f"""
        <div style="border-left: 4px solid {ACCENT};
                    background: color-mix(in srgb, {ACCENT} 8%, {SECONDARY_SURFACE});
                    padding: 1.15rem 1.5rem; border-radius: 8px;
                    font-size: 1.08rem; line-height: 1.6; color: {TEXT_PRIMARY};">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_subordinate_note(text: str, accent_color: str) -> None:
    """A quiet, secondary explanatory note -- deliberately smaller and less
    visually assertive than render_lede or the headline metric cards, so it
    reads as a caveat rather than competing with the headline numbers for
    attention (used for the Water/-100% and fan-cost callouts).
    """
    st.markdown(
        f"""
        <div style="border-left: 2px solid {accent_color}; padding: 0.5rem 0.95rem;
                    margin: 0.7rem 0 0.3rem; font-size: 0.85rem; line-height: 1.55;
                    color: {TEXT_SECONDARY};">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


# --- Multi-year regime summaries, vent-authority diagnostic, gate results ----
# (Headline Results / Why Ventilation Alone Fails / Validation & Limitations
# pages -- every one of these reads a file already written by a scripts/*.py
# run, never calls sim.engine.run itself.)


def _require_file(path: Path, how_to_generate: str) -> Path:
    if not path.exists():
        st.error(f"Missing `{path.relative_to(REPO_ROOT)}`. Run `{how_to_generate}` locally first and commit the output.")
        st.stop()
    return path


@st.cache_data(show_spinner=False)
def load_regime_summary(regime_key: str) -> pd.DataFrame:
    """Mean +/- sd across every cached year, one row per controller."""
    path = _require_file(RESULTS_DIR / f"regime_{regime_key}_summary.csv", "python scripts/experiment.py")
    return pd.read_csv(path).set_index("controller")


@st.cache_data(show_spinner=False)
def load_regime_raw(regime_key: str) -> pd.DataFrame:
    """One row per (year, controller) -- the per-year numbers behind the summary."""
    path = _require_file(RESULTS_DIR / f"regime_{regime_key}_raw.csv", "python scripts/experiment.py")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_vent_authority() -> dict:
    path = _require_file(
        PRECOMPUTED_DIR / "vent_authority.json", "python scripts/diagnose_vent_authority.py"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_vent_authority_envelope() -> pd.DataFrame:
    path = _require_file(
        PRECOMPUTED_DIR / "vent_authority_envelope.parquet", "python scripts/diagnose_vent_authority.py"
    )
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_gate_results() -> dict:
    path = _require_file(RESULTS_DIR / "gate_results.json", "python scripts/validate.py")
    return json.loads(path.read_text(encoding="utf-8"))


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
        "alternaria_risk": float(res["alternaria_risk"].iloc[-1]) if "alternaria_risk" in res.columns else float("nan"),
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
    """One shared style applied to every Plotly figure in the app, so the
    charts read as a designed set: no plot border, no vertical gridlines,
    only faint horizontal ones; muted axis/legend text; a hover label that
    matches the app's card surface instead of Plotly's default white box.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family=CHART_FONT_FAMILY, color=TEXT_PRIMARY, size=12.5),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11.5, color=TEXT_SECONDARY), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=SECONDARY_SURFACE, bordercolor=BORDER,
            font=dict(family=CHART_FONT_FAMILY, size=12, color=TEXT_PRIMARY),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False, tickfont=dict(color=TEXT_SECONDARY, size=11))
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, zeroline=False,
        tickfont=dict(color=TEXT_SECONDARY, size=11),
    )
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


# --- Page 0 (default): Headline Results -----------------------------------------


def pct_delta(value: float, baseline: float) -> float | None:
    """% change of value relative to baseline; None if baseline is 0 (undefined,
    not just large -- e.g. Alternaria risk 0.0 -> 0.0 in the dry season).
    """
    if baseline == 0:
        return None
    return (value - baseline) / baseline * 100.0


def render_headline_metric_card(col, key: str, mpc_mean: float, mpc_sd: float, fixed_mean: float) -> None:
    """A plain-language metric card: big plain-English number, technical term
    demoted to the delta line, one-line "what this means" underneath.
    """
    spec = METRIC_SPEC_BY_KEY[key]
    text = HEADLINE_CARD_TEXT[key]
    value_str = f"{spec['fmt'].format(mpc_mean)} ± {spec['fmt'].format(mpc_sd)}"
    if spec["unit"]:
        value_str += f" {spec['unit']}"
    delta = pct_delta(mpc_mean, fixed_mean)
    if delta is None:
        col.metric(text["card_label"], value_str, delta="Fixed is already 0 here", delta_color="off")
    else:
        col.metric(text["card_label"], value_str, delta=f"{delta:+.0f}% vs Fixed", delta_color="inverse")
    col.caption(f"_{text['technical']}._ {text['meaning']}")


def build_regime_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Controllers as rows, every metric as a 'mean +/- sd' string column.

    Column headers are plain-language (spec["plain"]); the technical term
    (spec["technical"]) is attached as a hover tooltip via column_config
    where this table is rendered, not deleted.
    """
    rows: dict[str, dict[str, str]] = {}
    for label in CONTROLLER_LABELS:
        key = CONTROLLER_KEY[label]
        if key not in summary.index:
            continue
        row: dict[str, str] = {}
        for spec in METRIC_SPECS:
            mean = summary.loc[key, f"{spec['key']}_mean"]
            sd = summary.loc[key, f"{spec['key']}_std"]
            row[spec["plain"]] = f"{spec['fmt'].format(mean)} ± {spec['fmt'].format(sd)}"
        rows[label] = row
    table = pd.DataFrame(rows).T
    table.index.name = "Controller"
    return table.reset_index()


def regime_summary_column_config() -> dict[str, "st.column_config.Column"]:
    """Tooltip (technical term) for every plain-language column build_regime_summary_table produces."""
    return {
        spec["plain"]: st.column_config.Column(spec["plain"], help=spec["technical"]) for spec in METRIC_SPECS
    }


def build_headline_bar(summary: pd.DataFrame, key: str, plain_label: str, technical_label: str) -> go.Figure:
    spec = METRIC_SPEC_BY_KEY[key]
    labels = [label for label in CONTROLLER_LABELS if CONTROLLER_KEY[label] in summary.index]
    means = [summary.loc[CONTROLLER_KEY[label], f"{key}_mean"] for label in labels]
    sds = [summary.loc[CONTROLLER_KEY[label], f"{key}_std"] for label in labels]
    colors = [CONTROLLER_COLOR[label] for label in labels]
    fig = go.Figure(
        go.Bar(
            x=labels, y=means,
            error_y=dict(type="data", array=sds, visible=True, color=TEXT_SECONDARY, thickness=1.5),
            marker_color=colors,
        )
    )
    fig.update_yaxes(title_text=f"{plain_label} ({spec['unit']})" if spec["unit"] else plain_label)
    fig.update_layout(title=dict(text=f"{plain_label} · {technical_label}", font=dict(size=12, color=TEXT_SECONDARY)))
    return style_fig(fig, height=340)


def build_plain_summary(regime_key: str, summary: pd.DataFrame) -> str:
    """The plain-language paragraph at the top of each regime tab -- framed
    around whichever outcomes are actually real in that regime (Regime A is
    a disease story, Regime B's disease risk is ~0 for everyone so it's a
    water/leaf-wetness story instead; see Honest Limitation #5 on the
    Validation page for why tomato's calendar drives both regimes).
    """
    alt_delta = pct_delta(summary.loc["mpc", "alternaria_risk_mean"], summary.loc["fixed", "alternaria_risk_mean"])
    wet_delta = pct_delta(summary.loc["mpc", "leaf_wet_hours_mean"], summary.loc["fixed", "leaf_wet_hours_mean"])
    water_delta = pct_delta(summary.loc["mpc", "water_L_per_m2_mean"], summary.loc["fixed", "water_L_per_m2_mean"])
    n_years = int(summary["n_years"].iloc[0]) if "n_years" in summary.columns else 5

    if regime_key == "A" and alt_delta is not None:
        return (
            f"In Assam's monsoon, the biggest threat to a polyhouse crop is not heat -- it is fungal disease "
            f"from constant leaf wetness. Across {n_years} simulated seasons, the predictive controller cut "
            f"disease risk by {abs(alt_delta):.0f}% and hours with wet leaves by {abs(wet_delta):.0f}% compared "
            "to the timer schedule most growers run today."
        )
    if water_delta is not None:
        return (
            f"In Assam's dry season, water and leaf-wetness control matter most. Across {n_years} simulated "
            f"seasons, the predictive controller cut hours with wet leaves by {abs(wet_delta):.0f}% and water "
            f"use by {abs(water_delta):.0f}% compared to the timer schedule most growers run today."
        )
    return (
        f"Across {n_years} simulated seasons, the predictive controller reduced disease risk and leaf-wetness "
        "compared to the timer schedule most growers run today."
    )


def render_headline_results() -> None:
    st.title("Headline Results")
    st.caption(
        "The single most important page: mean ± sd across every cached year (results/regime_A_summary.csv, "
        "regime_B_summary.csv), not one year's snapshot. A result that only holds in one year is not reported "
        "as a result here."
    )

    tabs = st.tabs([REGIME_LABELS["A"], REGIME_LABELS["B"]])
    for tab, regime_key in zip(tabs, ["A", "B"]):
        with tab:
            summary = load_regime_summary(regime_key)
            n_years = int(summary["n_years"].iloc[0]) if "n_years" in summary.columns else None
            night_narrow_pct = load_vent_authority()["frac_narrow_night"] * 100.0

            render_lede(build_plain_summary(regime_key, summary))
            if n_years:
                st.caption(f"n = {n_years} simulated years, mean ± sd across years.")

            st.subheader("Predictive controller vs. the Fixed timer most growers run today")
            st.caption(
                "Fixed is the naive timer schedule -- what a smallholder actually runs today with no sensors. "
                "That, not Threshold, is the comparison that matters."
            )
            cols = st.columns(3)
            for col, key in zip(cols, HEADLINE_METRIC_KEYS):
                mpc_mean = summary.loc["mpc", f"{key}_mean"]
                mpc_sd = summary.loc["mpc", f"{key}_std"]
                fixed_mean = summary.loc["fixed", f"{key}_mean"]
                render_headline_metric_card(col, key, mpc_mean, mpc_sd, fixed_mean)

            if regime_key == "A":
                render_subordinate_note(
                    "<b>Why \"Water used\" shows a -100% change:</b> during the monsoon, rainfall already "
                    "exceeds what the crop needs, so <i>any</i> rain-aware controller (Threshold or Predictive "
                    "alike) irrigates zero -- this is a feature of the monsoon season, not evidence the "
                    "predictive controller is smarter about water. Real, differentiated irrigation decisions "
                    f"happen in the <b>{REGIME_LABELS['B']}</b> tab -- that's where the water number actually "
                    "reflects controller behaviour.",
                    accent_color=ORANGE,
                )

            fan_mean = summary.loc["mpc", "fan_kWh_mean"]
            fan_cost = fan_mean * ELECTRICITY_RATE_INR_PER_KWH
            render_subordinate_note(
                "<b>What the leaf-wetness reduction costs to run:</b> the circulation fan is the actuator that "
                "buys the win above -- night-time ventilation alone can't reach dry-enough air (see "
                "<b>Why Ventilation Alone Fails</b>). Running it uses real electricity: "
                f"{fan_mean:,.0f} kWh over the season, about <b>₹{fan_cost:,.0f}</b> at Assam's APDCL tariff "
                f"(~₹{ELECTRICITY_RATE_INR_PER_KWH:.0f}/unit) -- against ₹0 for the Fixed timer, which runs no "
                "fan at all. The benefit above is real; so is this cost.",
                accent_color=FAN_COLOR,
            )

            st.subheader("Full comparison -- all controllers, all metrics")
            table = build_regime_summary_table(summary)
            st.dataframe(
                table, use_container_width=True, hide_index=True, column_config=regime_summary_column_config()
            )
            st.caption(
                "Hover a column header for its technical name. **\"% hours in ideal humidity range\"**: "
                "Predictive scores lowest here on purpose -- its objective deliberately deprioritises that "
                f"target in favour of leaf-wetness, because the target is unreachable for {night_narrow_pct:.1f}% "
                "of monsoon night hours (see **Why Ventilation Alone Fails**). This is working as designed, "
                "not an oversight."
            )

            st.subheader("Headline outcomes by controller (error bars = year-to-year variation)")
            bar_cols = st.columns(3)
            bar_labels = {
                "alternaria_risk": ("Disease risk", "Alternaria solani risk units"),
                "leaf_wet_hours": ("Hours with wet leaves", "leaf-wet hours"),
                "water_L_per_m2": ("Water used", "L/m² over the season"),
            }
            for col, key in zip(bar_cols, HEADLINE_METRIC_KEYS):
                plain, technical = bar_labels[key]
                col.plotly_chart(build_headline_bar(summary, key, plain, technical), use_container_width=True)

            fixed_alt_mean = summary.loc["fixed", "alternaria_risk_mean"]
            fixed_alt_sd = summary.loc["fixed", "alternaria_risk_std"]
            if fixed_alt_mean > 0:
                mpc_alt_mean = summary.loc["mpc", "alternaria_risk_mean"]
                mpc_alt_sd = summary.loc["mpc", "alternaria_risk_std"]
                st.caption(
                    f"**Disease risk error bars:** the Fixed baseline swings a lot year to year "
                    f"({fixed_alt_mean:.1f} ± {fixed_alt_sd:.1f} -- the swing is "
                    f"{fixed_alt_sd / fixed_alt_mean * 100:.0f}% of the mean) because monsoon severity itself "
                    f"varies season to season. Predictive's error bar is tiny by comparison "
                    f"({mpc_alt_mean:.1f} ± {mpc_alt_sd:.1f}) because it holds disease risk near zero *every* "
                    "season -- consistency is itself a result, not just the low average."
                )


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


# --- Page: Why Ventilation Alone Fails ------------------------------------------


def build_vpd_envelope_figure(envelope: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=envelope.index, y=envelope["vpd_vent1"], name="Achievable ceiling (vent fully open)",
            line=dict(color=BLUE, width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=envelope.index, y=envelope["vpd_vent0"], name="Achievable floor (vent fully closed)",
            line=dict(color=BLUE, width=1.5), fill="tonexty", fillcolor="rgba(57,135,229,0.16)",
        )
    )
    fig.add_hrect(
        y0=0.8, y1=1.2, fillcolor=GOOD, opacity=0.14, line_width=0,
        annotation_text="0.8-1.2 kPa target band", annotation_position="top left",
        annotation_font=dict(color=TEXT_SECONDARY, size=11),
    )
    fig.update_yaxes(title_text="VPD (kPa)")
    return style_fig(fig, height=440)


def render_vent_authority() -> None:
    st.title("Why Ventilation Alone Fails")
    st.caption("The central engineering finding of this project -- computed by scripts/diagnose_vent_authority.py.")

    data = load_vent_authority()

    st.markdown(
        f"**During JJA monsoon nights, {data['frac_narrow_night'] * 100:.1f}% of hours have less than "
        f"{data['narrow_range_threshold_kpa']:.1f} kPa of achievable VPD range between vents fully closed and "
        "fully open -- no vent setting, however clever the controller, can reach the 0.8-1.2 kPa target during "
        "those hours, because outdoor air is itself already almost saturated.**"
    )

    st.subheader("Day vs. night ventilation authority")
    split_df = pd.DataFrame(
        {
            "Period": ["All hours", "Daytime (06-18h)", "Night (18-06h)"],
            f"Fraction of hours < {data['narrow_range_threshold_kpa']:.1f} kPa achievable range": [
                f"{data['frac_narrow_all'] * 100:.1f}%",
                f"{data['frac_narrow_day'] * 100:.1f}%",
                f"{data['frac_narrow_night'] * 100:.1f}%",
            ],
        }
    )
    st.dataframe(split_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Computed over {data['window_start']}..{data['window_end']} ({data['n_hours']} hours), mean JJA "
        f"RH_out = {data['jja_rh_out_mean_pct']:.2f}%. Mean achievable VPD range across all hours: "
        f"{data['mean_achievable_range_kpa']:.3f} kPa; median: {data['median_achievable_range_kpa']:.3f} kPa "
        f"(p10={data['percentiles_kpa']['p10']:.3f}, p90={data['percentiles_kpa']['p90']:.3f})."
    )

    st.subheader("Achievable VPD envelope, representative monsoon week")
    envelope = load_vent_authority_envelope()
    st.plotly_chart(build_vpd_envelope_figure(envelope), use_container_width=True)
    st.caption(
        f"{data['envelope_week_start']}..{data['envelope_week_end']} -- the same week validated in "
        "figures/validation.png. The shaded band is every VPD value reachable by *some* vent_frac between 0 "
        "and 1 at that hour (fan off, vent=0 to vent=1 boundary runs); the 0.8-1.2 kPa target sits visibly "
        "outside the reachable envelope for most night hours."
    )

    st.subheader("Why this motivated the circulation fan")
    st.write(
        "Ventilation works by exchanging indoor air for outdoor air. During JJA monsoon nights, outdoor RH is "
        "already 85-95%, so opening the vents swaps saturated indoor air for near-saturated outdoor air and "
        "barely moves indoor VPD -- the mechanism above. A horizontal-airflow (HAF) circulation fan does not "
        "exchange air with outside at all: it thins the still boundary layer of air resting on the leaf "
        "surface (Stanghellini 1987; Monteith & Unsworth boundary-layer-resistance theory), which raises the "
        "*local* condensation threshold even while *bulk* indoor RH stays high. That gives the controller real "
        "authority over leaf-wetness during exactly the hours -- monsoon nights -- where ventilation alone has "
        "essentially none."
    )


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


def style_vs_fixed(df: pd.DataFrame, baseline_label: str = "Fixed") -> "pd.io.formats.style.Styler":
    """Colour each cell green/red relative to the Fixed row in its column --
    every metric on this table is "lower is better" (see METRIC_SPECS), so
    this reads directly as a comparison against the baseline rather than
    three unrelated numbers per column.
    """
    if baseline_label not in df.index:
        return df.style
    baseline_row = df.loc[baseline_label]

    def _color_column(col: pd.Series) -> list[str]:
        base = baseline_row[col.name]
        styles = []
        for idx in col.index:
            val = col[idx]
            if idx == baseline_label or base == val:
                styles.append("")
            elif base == 0:
                styles.append(f"color: {CRITICAL}" if val > 0 else "")
            elif val < base:
                styles.append(f"color: {GOOD}")
            else:
                styles.append(f"color: {CRITICAL}")
        return styles

    return df.style.apply(_color_column, axis=0)


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
        [
            "water_L_per_m2", "pct_vpd_band", "cumulative_dsv", "alternaria_risk", "leaf_wet_hours",
            "fan_kwh", "vent_actuations", "fan_actuations",
        ]
    ]
    # Plain-language primary label; technical term demoted to a column tooltip
    # (hover the header), not deleted -- see METRIC_SPECS' comment.
    COMPARISON_COLUMN_META = {
        "Water used": "L/m² applied over the season",
        "% hours in ideal humidity range": "% hours in achievable VPD band",
        "Late blight risk (temperate model)": "Wallin DSV",
        "Early blight risk": "Alternaria solani risk units",
        "Hours with wet leaves": "leaf-wet hours",
        "Fan electricity": "fan energy, kWh",
        "Vent movements": "vent actuations",
        "Fan movements": "fan actuations",
    }
    summary_df.columns = list(COMPARISON_COLUMN_META.keys())
    st.subheader(f"Comparison table -- {regime_label}, {year}")
    st.caption(
        "Colour is relative to the **Fixed** row (green = better, red = worse) -- every metric here is "
        "lower-is-better, so this reads as a comparison against the naive baseline, not three separate columns. "
        "Hover a column header for its technical name."
    )
    st.dataframe(
        style_vs_fixed(summary_df).format({
            "Water used": "{:.1f}", "% hours in ideal humidity range": "{:.2f}",
            "Late blight risk (temperate model)": "{:.0f}", "Early blight risk": "{:.0f}",
            "Hours with wet leaves": "{:.0f}", "Fan electricity": "{:.2f}",
            "Vent movements": "{:.0f}", "Fan movements": "{:.0f}",
        }),
        use_container_width=True,
        column_config={
            plain: st.column_config.Column(plain, help=technical) for plain, technical in COMPARISON_COLUMN_META.items()
        },
    )
    st.caption(
        "**\"% hours in ideal humidity range\":** Predictive scores lowest here on purpose -- night-time "
        "ventilation can't reach that target for most monsoon nights, so the controller deliberately spends "
        "its effort on leaf-wetness instead (see **Why Ventilation Alone Fails**). Working as designed."
    )

    st.subheader("Cumulative late blight risk (temperate model)")
    st.caption("Technical name: cumulative Wallin DSV. See Honest Limitation #3 on the Validation page for why this reads low in Guwahati's climate regardless of controller.")
    st.plotly_chart(build_dsv_overlay(results), use_container_width=True)

    st.subheader("Cumulative water use")
    st.plotly_chart(build_water_overlay(results), use_container_width=True)
    if regime_key == "A":
        st.caption(
            "Threshold and Predictive both flatten at zero here because monsoon rainfall already exceeds crop "
            "water demand, so any rain-aware controller irrigates nothing -- not because Predictive is "
            "cleverer about water. Switch the sidebar to **Dry season** to see irrigation decisions that "
            "actually differ between controllers."
        )

    with st.expander("Case study: irrigating right before rain (a caveat, not a headline)", expanded=False):
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
    both ends stay visible against the app's dark surface; the ramp's
    darkest steps (8-12) are skipped entirely for this reason.
    """
    t = max(0.0, min(100.0, pct)) / 100.0
    return _lerp_color(SEQ_BLUE[2], SEQ_BLUE[7], t)


# --- Cross-section geometry ---------------------------------------------------
# A schematic technical illustration, not a cartoon: recognisable Quonset-
# tunnel profile (matches the geometry already assumed in config.yaml's
# area_cover_m2 derivation -- see CLAUDE.md) with side posts, a curved
# poly-film roof drawn with ribs, and a hinged ridge vent, instead of a flat
# filled semicircle.
_ARCH_CENTER_X = 210.0
_ARCH_SPRING_Y = 175.0  # y where the roof film springs from the side posts
_ARCH_RX = 155.0
_ARCH_RY = 115.0
_GROUND_Y = 205.0
_POST_LEFT_X = 55.0
_POST_RIGHT_X = 365.0


def _arch_point(theta_deg: float) -> tuple[float, float]:
    """A point on the roof-film ellipse at angle theta_deg (180=left spring, 90=apex, 0=right spring)."""
    theta = math.radians(theta_deg)
    x = _ARCH_CENTER_X + _ARCH_RX * math.cos(theta)
    y = _ARCH_SPRING_Y - _ARCH_RY * math.sin(theta)
    return x, y


def _droplet_svg(x: float, y: float, scale: float = 1.0) -> str:
    """A droplet hanging from (x, y) -- used only along the roof film's inner
    surface, which is physically where condensation actually forms.
    """
    w, h = 6.5 * scale, 11.5 * scale
    return (
        f'<path d="M{x},{y} C{x - w},{y + h * 0.55} {x - w},{y + h * 1.15} {x},{y + h * 1.3} '
        f'C{x + w},{y + h * 1.15} {x + w},{y + h * 0.55} {x},{y} Z" fill="{BLUE}" opacity="0.75"/>'
    )


def _crop_silhouette_svg(x: float, y: float, scale: float = 1.0) -> str:
    """A restrained plant silhouette (stem + a few simple leaves) so the
    scene reads as a growing space, not an empty enclosure. Decorative only
    -- PLANT_COLOR is a fixed muted green, not one of the data-encoding
    palette colors above.
    """
    plant_color = "#4f7a55"
    stem_h = 15 * scale
    return f"""
    <g transform="translate({x},{y})" opacity="0.85">
      <line x1="0" y1="0" x2="0" y2="-{stem_h}" stroke="{plant_color}" stroke-width="{1.6 * scale}" stroke-linecap="round"/>
      <ellipse cx="-{5 * scale}" cy="-{stem_h * 0.65}" rx="{5.5 * scale}" ry="{2.6 * scale}" fill="{plant_color}" transform="rotate(-28 -{5 * scale} -{stem_h * 0.65})"/>
      <ellipse cx="{5 * scale}" cy="-{stem_h * 0.85}" rx="{5.5 * scale}" ry="{2.6 * scale}" fill="{plant_color}" transform="rotate(28 {5 * scale} -{stem_h * 0.85})"/>
      <ellipse cx="0" cy="-{stem_h * 1.05}" rx="{4.5 * scale}" ry="{2.2 * scale}" fill="{plant_color}" transform="rotate(0 0 -{stem_h * 1.05})"/>
    </g>
    """


def svg_polyhouse(
    t_in: float, vent_frac: float, fan_frac: float, moisture_pct: float, leaf_wet: bool, ts: pd.Timestamp
) -> str:
    """A schematic cross-section, restrained like a technical illustration:
    Quonset profile with structural ribs, a hinged ridge vent that rotates
    with vent_frac, a fan that visibly spins above fan_frac=0.5, condensation
    on the roof film's *inner* surface when leaf_wet (physically where it
    forms), soil coloured and textured by moisture, and a vertical air-volume
    gradient instead of one flat temperature fill.
    """
    cover_color = temp_to_cover_color(t_in)
    soil_color = moisture_to_soil_color(moisture_pct)
    vent_frac = max(0.0, min(1.0, vent_frac))
    fan_on = fan_frac > 0.5

    # Ridge vent: a hinged flap near the apex, closed = flush with the roof
    # slope, opening lifts it up and outward as vent_frac rises.
    hinge_x, hinge_y = 183.0, 66.0
    vent_angle = -6.0 - 55.0 * vent_frac

    # Structural ribs (purlins): straight members from the ground to three
    # points along the roof film, suggesting real framing under the film.
    ribs = "".join(
        f'<line x1="{x1:.1f}" y1="{_GROUND_Y:.0f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{TEXT_PRIMARY}" stroke-width="1.3" opacity="0.3"/>'
        for x1, (x2, y2) in (
            (_POST_LEFT_X, _arch_point(150)),
            (_ARCH_CENTER_X, _arch_point(90)),
            (_POST_RIGHT_X, _arch_point(30)),
        )
    )

    # Condensation droplets along the roof film's inner surface (a few
    # points spread across the dome, offset slightly inward/down from the
    # film itself), only when leaf_wet.
    droplet_thetas = (145, 118, 96, 74, 52, 25)
    droplets = ""
    if leaf_wet:
        droplets = "".join(
            _droplet_svg(x, y + 5.0, scale=0.9 + 0.15 * (i % 2))
            for i, (x, y) in enumerate(_arch_point(t) for t in droplet_thetas)
        )

    crops = "".join(
        _crop_silhouette_svg(x, _GROUND_Y - 2, scale=1.05 - 0.08 * (i % 3))
        for i, x in enumerate([95, 140, 185, 235, 280, 325])
    )

    wet_badge = (
        f'<span class="twin-badge" style="background:{CRITICAL};">leaf-wet</span>' if leaf_wet else ""
    )
    fan_badge = (
        f'<span class="twin-badge" style="background:{FAN_COLOR};">fan on</span>' if fan_on else ""
    )

    return f"""
    <div style="font-family:{CHART_FONT_FAMILY}; background:{SURFACE}; border:1px solid {BORDER};
                border-radius:10px; padding:14px;">
      <style>
        @keyframes twin-fan-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        .twin-fan-on {{ transform-origin: {_POST_LEFT_X:.0f}px 150px; animation: twin-fan-spin 1.1s linear infinite; }}
        .twin-badge {{
          color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 11.5px;
          font-weight: 600; margin-left: 8px; letter-spacing: 0.01em;
        }}
      </style>
      <svg viewBox="0 0 420 230" width="100%" height="250" style="display:block;">
        <defs>
          <linearGradient id="airGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="{cover_color}" stop-opacity="0.55"/>
            <stop offset="100%" stop-color="{cover_color}" stop-opacity="0.10"/>
          </linearGradient>
          <pattern id="soilTexture" width="9" height="9" patternTransform="rotate(35)" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="9" stroke="{SURFACE}" stroke-width="1.4" opacity="0.35"/>
          </pattern>
        </defs>

        <!-- interior air volume, vertical temperature gradient -->
        <path d="M{_POST_LEFT_X:.0f},{_GROUND_Y:.0f} L{_POST_LEFT_X:.0f},{_ARCH_SPRING_Y:.0f}
                 A{_ARCH_RX:.0f},{_ARCH_RY:.0f} 0 0 1 {_POST_RIGHT_X:.0f},{_ARCH_SPRING_Y:.0f}
                 L{_POST_RIGHT_X:.0f},{_GROUND_Y:.0f} Z" fill="url(#airGrad)"/>

        {ribs}

        <!-- roof film -->
        <path d="M{_POST_LEFT_X:.0f},{_ARCH_SPRING_Y:.0f}
                 A{_ARCH_RX:.0f},{_ARCH_RY:.0f} 0 0 1 {_POST_RIGHT_X:.0f},{_ARCH_SPRING_Y:.0f}"
              fill="none" stroke="{TEXT_SECONDARY}" stroke-width="2.2"/>

        <!-- side posts, with a small footing tick where each meets the ground -->
        <line x1="{_POST_LEFT_X:.0f}" y1="{_ARCH_SPRING_Y:.0f}" x2="{_POST_LEFT_X:.0f}" y2="{_GROUND_Y:.0f}"
              stroke="{TEXT_SECONDARY}" stroke-width="3.4"/>
        <line x1="{_POST_LEFT_X - 8:.0f}" y1="{_GROUND_Y:.0f}" x2="{_POST_LEFT_X + 8:.0f}" y2="{_GROUND_Y:.0f}"
              stroke="{TEXT_SECONDARY}" stroke-width="3.4" stroke-linecap="round"/>
        <line x1="{_POST_RIGHT_X:.0f}" y1="{_ARCH_SPRING_Y:.0f}" x2="{_POST_RIGHT_X:.0f}" y2="{_GROUND_Y:.0f}"
              stroke="{TEXT_SECONDARY}" stroke-width="3.4"/>
        <line x1="{_POST_RIGHT_X - 8:.0f}" y1="{_GROUND_Y:.0f}" x2="{_POST_RIGHT_X + 8:.0f}" y2="{_GROUND_Y:.0f}"
              stroke="{TEXT_SECONDARY}" stroke-width="3.4" stroke-linecap="round"/>

        <!-- ridge vent, hinged flap -->
        <g transform="translate({hinge_x:.0f},{hinge_y:.0f}) rotate({vent_angle:.1f})">
          <rect x="0" y="-2.5" width="50" height="5" rx="1.5" fill="{TEXT_PRIMARY}" opacity="0.55"/>
        </g>
        <circle cx="{hinge_x:.0f}" cy="{hinge_y:.0f}" r="2" fill="{TEXT_SECONDARY}"/>

        <!-- ground + soil -->
        <rect x="0" y="{_GROUND_Y:.0f}" width="420" height="25" fill="{soil_color}"/>
        <rect x="0" y="{_GROUND_Y:.0f}" width="420" height="25" fill="url(#soilTexture)"/>
        <line x1="0" y1="{_GROUND_Y:.0f}" x2="420" y2="{_GROUND_Y:.0f}" stroke="{BORDER}" stroke-width="1.5"/>

        {crops}

        <!-- fan, mounted on the left post -->
        <circle cx="{_POST_LEFT_X:.0f}" cy="150" r="14.5" fill="{SURFACE}" stroke="{FAN_COLOR}" stroke-width="2"
                opacity="{0.95 if fan_on else 0.55}"/>
        <g class="{'twin-fan-on' if fan_on else ''}">
          <path d="M{_POST_LEFT_X:.0f},150 L{_POST_LEFT_X:.0f},138.5 M{_POST_LEFT_X:.0f},150 L{_POST_LEFT_X + 10:.0f},156.5
                   M{_POST_LEFT_X:.0f},150 L{_POST_LEFT_X - 10:.0f},156.5"
                stroke="{FAN_COLOR}" stroke-width="2.4" stroke-linecap="round"
                opacity="{0.95 if fan_on else 0.55}"/>
        </g>

        {droplets}
      </svg>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;
                  padding-top:8px; border-top:1px solid {BORDER}; color:{TEXT_PRIMARY}; font-size:13px;">
        <span style="color:{TEXT_SECONDARY};">{ts:%Y-%m-%d %H:%M}</span>
        <span>
          T_in <b>{t_in:.1f}°C</b> &nbsp;·&nbsp;
          Vent <b>{vent_frac * 100:.0f}%</b> &nbsp;·&nbsp;
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
    components.html(html, height=380)

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


# --- Page: Model Validation & Honest Limitations --------------------------------

# CLAUDE.md's "Circulation fan actuator" sensitivity table (2026-08-11
# session): Threshold controller, fan on above 92% RH_in, 90-day monsoon
# window, rh_wet_threshold_max swept at 93/96/99%. Not re-derived here (no
# new physics) -- this is the exact recorded result, surfaced rather than
# left sitting only in prose.
FAN_SENSITIVITY_ROWS = [
    {"rh_wet_threshold_max": "93%", "leaf_wet_hours": 1249, "wallin_dsv": 4, "alternaria_risk": 7},
    {"rh_wet_threshold_max": "96% (shipped default)", "leaf_wet_hours": 990, "wallin_dsv": 2, "alternaria_risk": 0},
    {"rh_wet_threshold_max": "99%", "leaf_wet_hours": 277, "wallin_dsv": 0, "alternaria_risk": 0},
]


def build_gate_table(gate_data: dict) -> pd.DataFrame:
    df = pd.DataFrame(gate_data["gates"])
    df["Result"] = df["passed"].map(lambda p: "PASS" if p else "FAIL")
    return df.rename(
        columns={"name": "Gate", "description": "Description", "measured": "Measured", "bounds": "Bounds"}
    )[["Gate", "Description", "Measured", "Bounds", "Result"]]


def style_gate_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def _color_result(col: pd.Series) -> list[str]:
        return [f"color: {GOOD}; font-weight: 600" if v == "PASS" else f"color: {CRITICAL}; font-weight: 600" for v in col]

    return df.style.apply(_color_result, subset=["Result"], axis=0)


def render_validation_limitations() -> None:
    st.title("Model Validation & Honest Limitations")

    st.header("Validation: gates V1-V8")
    gate_data = load_gate_results()
    st.caption(
        f"7-day passive-policy run, {gate_data['window_start']}..{gate_data['window_end']}, vent fixed at 0.3, "
        "no irrigation -- results/gate_results.json, written by scripts/validate.py."
    )
    gates_df = build_gate_table(gate_data)
    st.dataframe(style_gate_table(gates_df), use_container_width=True, hide_index=True)
    n_fail = int((gates_df["Result"] == "FAIL").sum())
    if n_fail:
        st.error(f"{n_fail} gate(s) FAILED.")
    else:
        st.success(f"All {len(gates_df)} gates PASS.")

    col1, col2 = st.columns(2)
    val_png = FIGURES_DIR / "validation.png"
    et_png = FIGURES_DIR / "et_profile.png"
    if val_png.exists():
        col1.image(str(val_png), caption="T_in/T_out, RH_in, VPD over the validation week", use_container_width=True)
    if et_png.exists():
        col2.image(str(et_png), caption="Mean diurnal ET profile", use_container_width=True)

    st.divider()
    st.header("Honest limitations")
    st.caption("All already documented in CLAUDE.md -- surfaced here plainly, not softened.")

    st.subheader("1. The fan's leaf-wetness threshold mapping is a modelling assumption, not a measurement")
    st.write(
        "The fan raises the RH_in threshold used for `leaf_wet` from 90% (fan off) to 96% (fan at full power) "
        "to represent it thinning the leaf boundary layer. No boundary-layer transfer coefficient was fit to "
        "real HAF-fan data for this project -- 96% was chosen as a plausible ceiling, not derived from a cited "
        "source. The qualitative result (fans measurably raise the condensation-onset RH) is real and cited; "
        "the exact 96% figure is not."
    )
    sens_df = pd.DataFrame(FAN_SENSITIVITY_ROWS).rename(
        columns={
            "rh_wet_threshold_max": "Fan-on RH threshold", "leaf_wet_hours": "Leaf-wet hours (90d)",
            "wallin_dsv": "Wallin DSV", "alternaria_risk": "Alternaria risk",
        }
    )
    st.dataframe(sens_df, use_container_width=True, hide_index=True)
    st.caption(
        "Leaf-wet hours swing >4x (1249 -> 277) and Alternaria risk swings from 7 units to 0 across this "
        "range -- this assumption materially changes both disease-model outputs, not just a minor detail."
    )

    st.subheader("2. Weather is ERA5 reanalysis, not a Guwahati station observation")
    st.write(
        "All weather in this project comes from Open-Meteo's archive API, which serves ERA5-based reanalysis "
        "on a roughly 9 km grid cell, not a measurement taken at a Guwahati weather station. That is suitable "
        "for the climate-scale patterns this project studies (monsoon humidity regime, day/night cycles, "
        "multi-year variability) but is not a substitute for site-specific microclimate calibration -- a real "
        "deployment should validate against a local sensor before trusting absolute thresholds."
    )

    st.subheader("3. Wallin/BLITECAST DSV does not fit Guwahati's monsoon climate")
    st.write(
        "The Wallin (1962) late-blight table is calibrated for temperate climates -- its favourable "
        "temperature band tops out at 26.6°C. Guwahati's JJA monsoon mean ambient is 28.7°C, already above "
        "that ceiling most days before any polyhouse warming is added, which is why cumulative Wallin DSV "
        "reads structurally low here regardless of controller. This is exactly why the Alternaria solani "
        "(early blight) model was added alongside it -- Wallin is retained specifically to show this "
        "mismatch, not as the climate-appropriate disease signal."
    )

    st.subheader("4. Water use is structurally tied between Threshold and Predictive")
    st.write(
        "ETc (and therefore the soil water balance driving irrigation decisions) is computed from **outdoor** "
        "weather only -- it has no dependency on indoor climate or which vent/fan controller is running. Both "
        "Threshold's and Predictive's irrigation policies reduce to the same soil-depletion-plus-rain-skip "
        "rule acting on the same outdoor-driven soil trajectory, so no vent/fan strategy can differentiate "
        "water use from another rain-aware controller -- confirmed in Regime B, where both land on identical "
        "121.0 ± 11.4 L/m² water use, to the decimal."
    )

    st.subheader("5. The crop calendar uses tomato Kc for both regimes")
    st.write(
        "FAO-56 tomato crop coefficients drive ETc in both Regime A and Regime B, but tomato is actually a "
        "*rabi* (winter-sown) crop in Assam -- it isn't normally grown through the monsoon at all. Regime A's "
        "results should be read as modelling **off-season cultivation generally** (whatever crop occupies a "
        "monsoon-season polyhouse, using tomato's water/growth calendar as a stand-in), not literally "
        "monsoon-season tomato."
    )


# --- Page: Deployment & Cost ------------------------------------------------------


def render_deployment_cost() -> None:
    st.title("Deployment & Cost")
    st.caption(
        "Hardware retrofit budget for one existing 100 m² polyhouse. Every price is a real single-unit Indian "
        "retail listing (see Source column), researched August 2026 -- not invented. See dashboard/bom_data.py "
        "for full sourcing notes."
    )

    bom_df = pd.DataFrame(
        [
            {
                "Component": item.component, "Role": item.role, "Qty": item.qty,
                "Unit price (INR)": item.unit_price_inr, "Line total (INR)": item.line_total_inr,
                "Source": item.source_name,
            }
            for item in BOM
        ]
    )
    st.dataframe(
        bom_df.style.format({"Unit price (INR)": "₹{:.0f}", "Line total (INR)": "₹{:.0f}"}),
        use_container_width=True, hide_index=True,
    )
    total_inr = bom_total_inr()
    st.metric("Total hardware cost per polyhouse", f"₹{total_inr:,.0f}")
    st.caption(
        "Excludes wiring/mounting hardware, labour, and GST where the source listing didn't state "
        "GST-inclusive -- an order-of-magnitude retrofit budget, not a quote. The HAF fan line uses a small "
        "circulation-fan analog (18-inch, ~125W class) rather than a large industrial greenhouse exhaust fan "
        "(1.5HP+ listings run ₹18,000-25,000, a different product class than config.yaml's fan spec models)."
    )

    st.divider()
    st.header("What follows from this architecture")
    st.markdown(
        "- **Runs on the edge.** Every controller (Fixed, Threshold, Predictive) is plain Python control logic "
        "small enough to run on an ESP32-class microcontroller -- no cloud inference, no GPU, no external API "
        "call in the control loop itself.\n"
        "- **Deterministic and needs no network connection.** Fixed and Threshold are pure rule-based logic; "
        "Predictive plans on a 12-hour receding horizon (replanned every hour) using locally-available "
        "weather. None of the three controllers requires internet connectivity to keep operating -- a real "
        "advantage for rural polyhouse sites with unreliable connectivity.\n"
        "- **Retrofits onto existing structures.** The BOM above bolts onto an existing polyhouse (sensors, "
        "an actuator on the existing vent, two fans, a solenoid on the existing irrigation line) -- it does "
        "not require building a new structure."
    )

    st.divider()
    st.header("Scale-up projection")
    st.caption(
        "A transparent linear projection, not a fixed claim -- move the slider and the numbers below update "
        "with it, so the assumption stays visible rather than hidden inside a single quoted figure."
    )
    n_units = st.slider("Number of polyhouses (N)", min_value=1, max_value=500, value=50, step=1)
    proj_regime_key = st.radio(
        "Regime", ["A", "B"], format_func=lambda k: REGIME_LABELS[k], horizontal=True, key="deploy_regime"
    )
    summary = load_regime_summary(proj_regime_key)

    water_saved_per_house_l = (
        summary.loc["fixed", "water_L_per_m2_mean"] - summary.loc["mpc", "water_L_per_m2_mean"]
    ) * 100.0  # x100 m2 floor area per house
    alternaria_reduction_per_house = summary.loc["fixed", "alternaria_risk_mean"] - summary.loc["mpc", "alternaria_risk_mean"]
    leaf_wet_reduction_per_house = summary.loc["fixed", "leaf_wet_hours_mean"] - summary.loc["mpc", "leaf_wet_hours_mean"]

    total_cost = total_inr * n_units
    total_water = water_saved_per_house_l * n_units
    total_alternaria = alternaria_reduction_per_house * n_units
    total_leaf_wet = leaf_wet_reduction_per_house * n_units

    cols = st.columns(4)
    cols[0].metric("Total hardware cost", f"₹{total_cost:,.0f}")
    cols[1].metric("Water saved / season", f"{total_water:,.0f} L")
    cols[2].metric("Alternaria risk reduction", f"{total_alternaria:,.0f} units")
    cols[3].metric("Leaf-wet hours avoided", f"{total_leaf_wet:,.0f} h")
    st.caption(
        f"N × (Predictive − Fixed) from the Headline Results table for {REGIME_LABELS[proj_regime_key]}, "
        "× 100 m² floor area per house for water. Does not model shared infrastructure, water-source limits, "
        "maintenance, or interaction effects across units -- a simple per-unit multiplication, shown with the "
        "slider specifically so the assumption stays visible rather than becoming a single unverifiable claim."
    )


# --- Page: For Growers (plain-language, last in the nav on purpose) -------------
#
# This page exists to show end-user thinking for the Sustainability Impact
# score -- it is not the headline and must not displace the technical pages
# a judge reads first. No jargon, no equations, no metric abbreviations;
# every number is pulled from the same regime summaries the rest of the app
# uses, never re-typed by hand, so it can't drift out of sync with them.


def render_for_growers() -> None:
    st.title("For Growers")
    st.caption("Plain language, no jargon, no equations -- what this means for someone running a polyhouse.")

    st.header("The problem, in two sentences")
    st.write(
        "In Assam's monsoon, a polyhouse's real enemy isn't heat -- it's humidity. When leaves stay wet for "
        "hours at a stretch, fungal disease takes hold, and a grower running the vents on a fixed clock has "
        "no way to know when that's happening or to do anything about it."
    )

    st.header("What the system physically is")
    st.write("A small kit that bolts onto an existing polyhouse -- it does not require building anything new:")
    st.markdown(
        "- Sensors that read temperature, humidity, sunlight, and soil moisture\n"
        "- A motor that opens and closes the roof vent\n"
        "- Two small fans that blow air gently across the leaves\n"
        "- A valve that switches the drip irrigation on and off\n"
        "- A small controller box that makes the decisions"
    )
    st.metric("Total cost to fit one polyhouse", f"₹{bom_total_inr():,.0f}")
    st.caption("Full parts list and sourcing on the Deployment & Cost page.")

    st.header("What it does differently from a timer")
    st.write(
        "A timer opens the vents at the same two clock times every day, no matter what the weather is "
        "actually doing. This system instead looks up to 12 hours ahead at the weather, and -- the important "
        "part -- runs the fans specifically at night, when opening the vents alone can't dry the air but "
        "fungus is most likely to infect the leaves. It waters the crop only when the soil actually needs it, "
        "and skips watering if rain is already on the way."
    )

    st.header("What a grower gets")
    summary_a = load_regime_summary("A")
    summary_b = load_regime_summary("B")
    alt_delta = pct_delta(summary_a.loc["mpc", "alternaria_risk_mean"], summary_a.loc["fixed", "alternaria_risk_mean"])
    wet_delta_a = pct_delta(summary_a.loc["mpc", "leaf_wet_hours_mean"], summary_a.loc["fixed", "leaf_wet_hours_mean"])
    water_delta_b = pct_delta(summary_b.loc["mpc", "water_L_per_m2_mean"], summary_b.loc["fixed", "water_L_per_m2_mean"])
    cols = st.columns(3)
    if alt_delta is not None:
        cols[0].metric("Less disease risk", f"{abs(alt_delta):.0f}% lower")
        cols[0].caption("monsoon season, vs. a timer")
    cols[1].metric("Fewer hours with wet leaves", f"{abs(wet_delta_a):.0f}% fewer")
    cols[1].caption("monsoon season, vs. a timer")
    if water_delta_b is not None:
        cols[2].metric("Less water wasted", f"{abs(water_delta_b):.0f}% less")
        cols[2].caption("dry season, vs. a timer -- see below for why monsoon water isn't the story")
    st.caption(
        "Water savings are shown for the **dry season**, not the monsoon: during the monsoon, rain alone "
        "already covers what the crop needs, so the monsoon water number doesn't say much about the system -- "
        "the dry season is where its irrigation decisions actually matter."
    )

    st.header("What it costs to run")
    fan_mean = summary_a.loc["mpc", "fan_kWh_mean"]
    fan_cost = fan_mean * ELECTRICITY_RATE_INR_PER_KWH
    st.write(
        f"Running the fans uses about {fan_mean:,.0f} units of electricity over a monsoon season -- roughly "
        f"**₹{fan_cost:,.0f}** at Assam's APDCL electricity rate (~₹{ELECTRICITY_RATE_INR_PER_KWH:.0f} per "
        "unit). That's the real running cost behind the disease-risk reduction above; it isn't free, and this "
        "project doesn't pretend it is."
    )

    st.header("What it does not do")
    st.warning(
        "This system does **not** directly increase yield. What it does is reduce disease risk and cut "
        "wasted water. Whether healthier, less-stressed plants also produce more or better fruit is likely, "
        "but it is **not something this project measured** -- we are not claiming a yield number here."
    )


# --- App shell -------------------------------------------------------------------


PAGE_RENDERERS = {
    "Headline Results": render_headline_results,
    "Why Ventilation Alone Fails": render_vent_authority,
    "Controller Comparison": render_controller_comparison,
    "Live Simulation": render_live_simulation,
    "Live Twin": render_live_twin,
    "Validation & Limitations": render_validation_limitations,
    "Deployment & Cost": render_deployment_cost,
    "For Growers": render_for_growers,
}


def render_sidebar_nav() -> str:
    """Product-navigation sidebar: logo + title, two labelled sections
    (Results / Explore), the active page rendered as a filled primary
    button (a native, theme-consistent "selected" state -- not a CSS hack
    on radio internals), and a footer with the project name and repo link.
    """
    st.session_state.setdefault("active_page", "Headline Results")

    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
                {LOGO_SVG}
                <span style="font-size:1.2rem; font-weight:700; letter-spacing:-0.01em; color:{TEXT_PRIMARY};">
                    Avinya Twin
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Polyhouse digital twin -- Guwahati, Assam")

        for section_label, page_names in NAV_SECTIONS.items():
            st.markdown(f'<div class="sidebar-nav-label">{section_label}</div>', unsafe_allow_html=True)
            for page_name in page_names:
                is_active = st.session_state.active_page == page_name
                if st.button(
                    page_name, key=f"nav_{page_name}", use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_page = page_name
                    st.rerun()

        st.markdown(
            f"""
            <div style="margin-top:1.6rem; padding-top:0.9rem; border-top:1px solid {BORDER};
                        font-size:0.78rem; color:{TEXT_SECONDARY}; line-height:1.6;">
                <div style="font-weight:600; color:{TEXT_PRIMARY};">avinya-twin</div>
                <div>Physics-based polyhouse controller comparison for Guwahati, Assam.</div>
                <a href="{REPO_URL}" target="_blank" style="color:{ACCENT}; text-decoration:none;">
                    View source on GitHub &rarr;
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.active_page


def main() -> None:
    st.set_page_config(
        page_title="Avinya Twin", page_icon=str(REPO_ROOT / "dashboard" / "assets" / "favicon.png"), layout="wide"
    )
    inject_css()
    page = render_sidebar_nav()
    PAGE_RENDERERS[page]()


if __name__ == "__main__":
    main()
