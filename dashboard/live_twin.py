"""Interactive HTML5 canvas Live Twin: an operator-console view over one
already-computed run.

**This module is a viewer, not a model.** It contains no physics, no control
logic and no derived numbers. Every value it draws or prints is read straight
out of the precomputed Parquet record that `dashboard/app.py` already loads
for the Live Twin page, and every actuator reason string is produced by the
existing rule-based explanation layer (`app.explain_controller_action`),
passed in as a callable so there is exactly one copy of that text in the
codebase.

Two pieces live here:

``build_twin_payload``
    Turns one (controller, regime, year) result frame into a compact JSON
    payload: hourly series rounded to the precision actually displayed, plus
    an interned string table of the per-hour rule explanations. Timestamps
    are not serialised per row; every cached window is strictly hourly and
    contiguous (asserted at build time), so the component derives them from
    a single start timestamp.

``build_component_html``
    Assembles the self-contained page handed to ``st.components.v1.html``:
    one inline stylesheet, one inline script, vanilla canvas drawing only.
    No CDN scripts, no webfonts, no images, no fetch/XHR/WebSocket. That is
    deliberate: the app claims elsewhere that it runs with no network access,
    and a component that quietly pulled a font would make that claim false.

The scene is a shallow oblique (cavalier) projection of a Quonset tunnel:
model space is (u across the house, v along its length, h up), projected with
``sx = u + v*DX``, ``sy = -h - v*DY`` so the depth axis recedes up and to the
right. The geometry is schematic. It matches the tunnel shape assumed in
config.yaml's ``area_cover_m2`` derivation in character, not in dimensions,
and nothing drawn here feeds back into the model.
"""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd

# Fixed iframe height for st.components.v1.html. The internal layout is a
# flex column sized to exactly this, with the scene taking the slack, so no
# inner scrollbar appears and nothing is cut off. Keep in sync with the
# per-block heights in _CSS if those change.
COMPONENT_HEIGHT = 900

# Columns the payload needs. Anything not in this list is not shown.
REQUIRED_COLUMNS = (
    "T_in", "T_out", "RH_in", "vent", "fan", "soil_pct", "leaf_wet", "irrigation", "rain",
)


def _series(res: pd.DataFrame, col: str, nd: int) -> list[float]:
    """One column rounded to the precision it is actually displayed at."""
    return [round(float(v), nd) for v in res[col].to_numpy()]


def _intern(lines: list[str]) -> dict:
    """String table plus index array.

    The rule layer repeats itself heavily over a 2160 hour window (the Fixed
    controller's irrigation line has exactly two variants for the whole run,
    and long quiet stretches produce byte-identical climate lines), so
    interning is worth roughly a third of the payload.
    """
    table: list[str] = []
    seen: dict[str, int] = {}
    index: list[int] = []
    for s in lines:
        i = seen.get(s)
        if i is None:
            i = len(table)
            seen[s] = i
            table.append(s)
        index.append(i)
    return {"table": table, "index": index}


def build_twin_payload(
    res: pd.DataFrame,
    *,
    controller_label: str,
    regime_label: str,
    year: int,
    explain_fn: Callable[[pd.DataFrame, int, str, float, float, float], list[str]],
    vpd_low: float,
    vpd_high: float,
    raw_pct: float,
    rh_wet_min: float,
) -> dict:
    """Serialise one cached run for the component.

    `explain_fn` is `app.explain_controller_action`, injected rather than
    imported so this module cannot accidentally grow a second copy of the
    reason text (and so app.py keeps ownership of that layer).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in res.columns]
    if missing:
        raise KeyError(f"cached run is missing columns needed by the Live Twin viewer: {missing}")

    n = len(res)
    if n < 2:
        raise ValueError("cached run is too short to animate")

    step = res.index.to_series().diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta("1h"):
        raise ValueError("cached run is not a contiguous hourly series; cannot derive timestamps")

    climate: list[str] = []
    water: list[str] = []
    for i in range(n):
        lines = explain_fn(res, i, controller_label, vpd_low, vpd_high, raw_pct)
        climate.append(lines[0] if len(lines) > 0 else "")
        water.append(lines[1] if len(lines) > 1 else "")

    leaf_wet = "".join("1" if bool(v) else "0" for v in res["leaf_wet"].to_numpy())

    return {
        "t0": res.index[0].isoformat(),
        "n": n,
        "meta": {
            "controller": controller_label,
            "regime": regime_label,
            "year": int(year),
            "vpdBand": [round(float(vpd_low), 2), round(float(vpd_high), 2)],
        },
        "thresholds": {"rawPct": round(float(raw_pct), 1), "rhWetMin": round(float(rh_wet_min), 1)},
        "series": {
            "T_in": _series(res, "T_in", 1),
            "T_out": _series(res, "T_out", 1),
            "RH_in": _series(res, "RH_in", 1),
            "vent": _series(res, "vent", 2),
            "fan": _series(res, "fan", 2),
            "soil_pct": _series(res, "soil_pct", 1),
            "irrigation": _series(res, "irrigation", 2),
            "rain": _series(res, "rain", 1),
            "leaf_wet": leaf_wet,
        },
        "reasons": {"climate": _intern(climate), "water": _intern(water)},
    }


def payload_json(payload: dict) -> str:
    """Compact JSON, with `<` escaped so no string can close the inline
    script tag it gets embedded in.
    """
    return json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")


def payload_size_kb(payload_js: str) -> float:
    return len(payload_js.encode("utf-8")) / 1024.0


# --- Page assembly -----------------------------------------------------------


def build_component_html(payload_js: str, palette: dict[str, str], fallback_html: str) -> str:
    """The full self-contained document body for st.components.v1.html."""
    css_vars = "".join(f"--tw-{k}:{v};" for k, v in palette.items())
    palette_js = json.dumps(palette, separators=(",", ":")).replace("<", "\\u003c")
    js = _JS.replace("__PAYLOAD__", payload_js).replace("__PALETTE__", palette_js)
    return (
        f"<style>:root{{{css_vars}}}{_CSS}</style>"
        f"{_BODY_OPEN}"
        f'<div id="tw-fallback" hidden><div class="tw-fallback-note">'
        "Interactive viewer could not start in this browser. Showing the static cross section "
        "for the first hour of the run instead."
        f"</div>{fallback_html}</div>"
        f"{_BODY_CLOSE}"
        f"<script>{js}</script>"
    )


_CSS = """
*,*::before,*::after{box-sizing:border-box;}
html,body{margin:0;padding:0;height:100%;background:var(--tw-surface);}
#tw-root{
  height:100%;padding:12px 14px;display:flex;flex-direction:column;gap:10px;
  font-family:var(--tw-font);color:var(--tw-text);background:var(--tw-surface);
  border:1px solid var(--tw-border);border-radius:10px;outline:none;
}
#tw-root:focus-visible{border-color:var(--tw-accent);}
#tw-head{display:flex;align-items:baseline;justify-content:space-between;gap:14px;height:38px;flex:0 0 auto;}
.tw-run{font-size:13px;font-weight:600;letter-spacing:-0.01em;}
.tw-run span{color:var(--tw-muted);font-weight:500;}
.tw-clock{display:flex;align-items:baseline;gap:14px;font-variant-numeric:tabular-nums;}
#tw-ts{font-size:16px;font-weight:600;letter-spacing:-0.01em;}
#tw-hour{font-size:12px;color:var(--tw-muted);}
#tw-focus{font-size:11px;color:var(--tw-muted);}
#tw-scene-wrap{position:relative;flex:1 1 auto;min-height:330px;border:1px solid var(--tw-border);
  border-radius:8px;overflow:hidden;background:var(--tw-surface);}
#tw-scene{display:block;width:100%;height:100%;}
#tw-tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .08s linear;
  background:var(--tw-surface2);border:1px solid var(--tw-border);border-radius:6px;
  padding:6px 9px;font-size:11.5px;line-height:1.45;color:var(--tw-text);max-width:230px;
  box-shadow:0 6px 18px rgba(0,0,0,.45);z-index:4;}
#tw-tip b{font-weight:600;}
#tw-act{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;height:136px;flex:0 0 auto;}
.tw-tile{border:1px solid var(--tw-border);border-radius:8px;background:var(--tw-surface2);
  padding:9px 11px;display:flex;flex-direction:column;gap:5px;min-width:0;}
.tw-tile-head{display:flex;align-items:center;justify-content:space-between;gap:8px;}
.tw-tile-name{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--tw-muted);font-weight:600;}
.tw-tile-val{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-0.01em;}
.tw-state{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:600;}
.tw-dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto;}
.tw-why{font-size:11.5px;line-height:1.45;color:var(--tw-muted);overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;}
#tw-charts-wrap{height:232px;flex:0 0 auto;border:1px solid var(--tw-border);border-radius:8px;
  overflow:hidden;background:var(--tw-surface);}
#tw-charts{display:block;width:100%;height:100%;cursor:crosshair;}
#tw-controls{display:flex;align-items:center;gap:10px;height:36px;flex:0 0 auto;}
.tw-btn{font:inherit;font-size:12px;font-weight:600;color:var(--tw-text);background:var(--tw-surface2);
  border:1px solid var(--tw-border);border-radius:6px;padding:6px 11px;cursor:pointer;line-height:1;
  white-space:nowrap;}
.tw-btn:hover{border-color:var(--tw-accent);}
.tw-btn.tw-on{background:var(--tw-accent);border-color:var(--tw-accent);color:#0d1512;}
#tw-play{min-width:64px;}
.tw-speed{display:flex;gap:4px;align-items:center;}
.tw-speed-label{font-size:11px;color:var(--tw-muted);margin-right:2px;}
#tw-scrub{flex:1 1 auto;-webkit-appearance:none;appearance:none;height:5px;border-radius:3px;
  background:var(--tw-border);outline:none;cursor:pointer;min-width:80px;}
#tw-scrub::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--tw-accent);border:2px solid var(--tw-surface);cursor:pointer;}
#tw-scrub::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--tw-accent);
  border:2px solid var(--tw-surface);cursor:pointer;}
#tw-fallback{padding:10px;}
.tw-fallback-note{font-size:12px;color:var(--tw-muted);margin-bottom:10px;}
"""

_BODY_OPEN = """
<div id="tw-root" tabindex="0">
  <div id="tw-head">
    <div class="tw-run" id="tw-run"></div>
    <div class="tw-clock">
      <span id="tw-focus">Click the panel to enable space and arrow keys</span>
      <span id="tw-hour"></span>
      <span id="tw-ts"></span>
    </div>
  </div>
  <div id="tw-scene-wrap"><canvas id="tw-scene"></canvas><div id="tw-tip"></div></div>
  <div id="tw-act">
    <div class="tw-tile" id="tile-vent">
      <div class="tw-tile-head"><span class="tw-tile-name">Roof vent</span>
        <span class="tw-tile-val" id="val-vent"></span></div>
      <div class="tw-state" id="state-vent"></div>
      <div class="tw-why" id="why-vent"></div>
    </div>
    <div class="tw-tile" id="tile-fan">
      <div class="tw-tile-head"><span class="tw-tile-name">Circulation fan</span>
        <span class="tw-tile-val" id="val-fan"></span></div>
      <div class="tw-state" id="state-fan"></div>
      <div class="tw-why" id="why-fan"></div>
    </div>
    <div class="tw-tile" id="tile-irr">
      <div class="tw-tile-head"><span class="tw-tile-name">Irrigation valve</span>
        <span class="tw-tile-val" id="val-irr"></span></div>
      <div class="tw-state" id="state-irr"></div>
      <div class="tw-why" id="why-irr"></div>
    </div>
  </div>
  <div id="tw-charts-wrap"><canvas id="tw-charts"></canvas></div>
  <div id="tw-controls">
    <button class="tw-btn" id="tw-back" title="Step back one hour">Back</button>
    <button class="tw-btn" id="tw-play">Play</button>
    <button class="tw-btn" id="tw-fwd" title="Step forward one hour">Forward</button>
    <span class="tw-speed">
      <span class="tw-speed-label">Speed</span>
      <button class="tw-btn tw-sp" data-hps="1">1x</button>
      <button class="tw-btn tw-sp" data-hps="5">5x</button>
      <button class="tw-btn tw-sp" data-hps="20">20x</button>
    </span>
    <input type="range" id="tw-scrub" min="0" step="1" value="0">
  </div>
</div>
"""

_BODY_CLOSE = ""


# The component script. Written as a plain template (no f-string) so the
# braces do not need escaping; __PAYLOAD__ and __PALETTE__ are substituted in
# build_component_html.
_JS = r"""
(function(){
'use strict';

var DATA = __PAYLOAD__;
var C = __PALETTE__;

var root = document.getElementById('tw-root');
var fallbackEl = document.getElementById('tw-fallback');
var fellBack = false;
var booted = false;

// Idempotent and callable at any point in the panel's life, not just during
// boot: a throw inside the per-frame tick() (see below) calls this exactly
// the same way a boot-time failure or the watchdog timeout does.
function showFallback(err){
  if (fellBack) return;
  fellBack = true;
  // style.display, not the hidden attribute: #tw-root's own CSS rule sets
  // display:flex with higher specificity than the UA [hidden] rule, so the
  // attribute alone silently loses and leaves the broken panel on screen
  // underneath the fallback note.
  try {
    if (root) root.style.display = 'none';
    if (fallbackEl) fallbackEl.hidden = false;
  } catch (e) {}
  if (err && window.console) console.error('[LiveTwin] falling back to static cross section:', err);
}
var watchdog = setTimeout(function(){
  if (!booted) showFallback(new Error('initialisation timed out'));
}, 2500);

try {

// ---------------------------------------------------------------- helpers --
function clamp(v, a, b){ return v < a ? a : (v > b ? b : v); }
function hexToRgb(h){
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
function rgba(hex, a){
  var c = hexToRgb(hex);
  return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
}
function lerpHex(a, b, t){
  t = clamp(t, 0, 1);
  var x = hexToRgb(a), y = hexToRgb(b), o = '#';
  for (var i = 0; i < 3; i++){
    var v = Math.round(x[i] + (y[i] - x[i]) * t).toString(16);
    o += v.length < 2 ? '0' + v : v;
  }
  return o;
}
// Deterministic per-index jitter so plants, droplets and rain streaks are
// uneven but identical on every redraw.
function rnd(i){
  var x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}
function fmt(v, nd){
  var s = v.toFixed(nd);
  return s === '-0' || s === '-0.0' || s === '-0.00' ? s.slice(1) : s;
}
var PAD2 = function(v){ return v < 10 ? '0' + v : '' + v; };

// The same diverging map dashboard/app.py's temp_to_cover_color uses, so the
// interior tint reads as the same product as the static illustration. Not a
// new number: the anchors are copied from that function.
function tempColor(t){
  var COLD = 20.0, MID = 32.5, HOT = 45.0;
  var v = clamp(t, COLD, HOT);
  if (v <= MID) return lerpHex(C.blue, C.neutralMid, (v - COLD) / (MID - COLD));
  return lerpHex(C.neutralMid, C.hot, (v - MID) / (HOT - MID));
}
function soilColor(pct){ return lerpHex(C.soilDry, C.soilWet, clamp(pct, 0, 100) / 100); }

// ------------------------------------------------------------------ data --
var S = DATA.series;
var N = DATA.n;
var WET = S.leaf_wet;
var RCLIM = DATA.reasons.climate, RWATER = DATA.reasons.water;
// The cached timestamps are naive local wall-clock stamps for Guwahati, so
// they are parsed as UTC and read back with the getUTC* accessors. Letting
// Date parse them in the browser's own zone would shift every hour label by
// the viewer's offset, which would silently misplace the day/night wash.
var T0 = (function(){
  var m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/.exec(DATA.t0);
  if (!m) throw new Error('unparseable start timestamp: ' + DATA.t0);
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]));
})();
var MS_H = 3600000;

function at(i){
  return {
    i: i,
    t: new Date(T0.getTime() + i * MS_H),
    tin: S.T_in[i], tout: S.T_out[i], rh: S.RH_in[i],
    vent: S.vent[i], fan: S.fan[i], soil: S.soil_pct[i],
    wet: WET.charAt(i) === '1', irr: S.irrigation[i], rain: S.rain[i]
  };
}
function stampOf(d){
  return d.getUTCFullYear() + '-' + PAD2(d.getUTCMonth() + 1) + '-' + PAD2(d.getUTCDate())
       + '  ' + PAD2(d.getUTCHours()) + ':00';
}

// ----------------------------------------------------------------- state --
var st = { idxF: 0, idx: 0, playing: false, hps: 5, fanAngle: 0, t: 0, rainPhase: 0, dash: 0 };

// --------------------------------------------------------------- geometry --
// Model space: u across the house, v along its length, h up.
var MW = 100, MD = 150, WALL = 34, RIDGE = 74;
var DX = 0.52, DY = 0.30;          // depth axis: right and up on screen
var BOX = { x0: -58, x1: 238, y0: -128, y1: 22 };
var VENT_U0 = 0.50 * MW, VENT_U1 = 0.74 * MW, VENT_V0 = 0.26 * MD, VENT_V1 = 0.76 * MD;
var FAN_V = 0.27 * MD, FAN_H = 25, FAN_R = 15;
var BED_U0 = 0.05 * MW, BED_U1 = 0.95 * MW, BED_V0 = 0.02 * MD, BED_V1 = 0.98 * MD;
var POST_V = [0, 0.25 * MD, 0.5 * MD, 0.75 * MD, MD];
var HOOP_V = [0, 0.25 * MD, 0.5 * MD, 0.75 * MD, MD];
var PURLIN_U = [0, 0.14, 0.32, 0.5, 0.68, 0.86, 1].map(function(f){ return f * MW; });

function archH(u){
  var r = (u - MW / 2) / (MW / 2);
  var s = 1 - r * r;
  return WALL + (RIDGE - WALL) * Math.sqrt(s > 0 ? s : 0);
}

var view = { s: 1, ox: 0, oy: 0 };
function px(u, v){ return view.ox + (u + v * DX) * view.s; }
function py(v, h){ return view.oy + (-h - v * DY) * view.s; }
function P(u, v, h){ return [px(u, v), py(v, h)]; }

function fitView(w, hgt){
  var s = Math.min(w / (BOX.x1 - BOX.x0), hgt / (BOX.y1 - BOX.y0));
  view.s = s;
  view.ox = (w - (BOX.x1 - BOX.x0) * s) / 2 - BOX.x0 * s;
  view.oy = (hgt - (BOX.y1 - BOX.y0) * s) / 2 - BOX.y0 * s;
}

// Silhouette of the swept tunnel, as the union (nonzero winding) of a stack
// of translated gable outlines. Rebuilt only on resize.
var SIL = null;
function buildSilhouette(){
  var p = new Path2D(), K = 14, k, j, u, a;
  for (k = 0; k <= K; k++){
    var v = MD * k / K;
    p.moveTo(px(0, v), py(v, 0));
    p.lineTo(px(0, v), py(v, WALL));
    for (j = 0; j <= 28; j++){
      u = MW * j / 28;
      a = P(u, v, archH(u));
      p.lineTo(a[0], a[1]);
    }
    p.lineTo(px(MW, v), py(v, 0));
    p.closePath();
  }
  SIL = p;
}

// ---------------------------------------------------------------- canvases --
var sceneWrap = document.getElementById('tw-scene-wrap');
var scene = document.getElementById('tw-scene');
var sctx = scene.getContext('2d');
var chartsWrap = document.getElementById('tw-charts-wrap');
var charts = document.getElementById('tw-charts');
var cctx = charts.getContext('2d');
var chartBase = document.createElement('canvas');
var bctx = chartBase.getContext('2d');
var sceneW = 0, sceneH = 0, chW = 0, chH = 0;

function sizeCanvas(cv, ctx, w, h){
  var dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(w * dpr));
  cv.height = Math.max(1, Math.round(h * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function layout(){
  var r = sceneWrap.getBoundingClientRect();
  sceneW = Math.max(1, r.width); sceneH = Math.max(1, r.height);
  sizeCanvas(scene, sctx, sceneW, sceneH);
  fitView(sceneW, sceneH);
  buildSilhouette();
  var q = chartsWrap.getBoundingClientRect();
  chW = Math.max(1, q.width); chH = Math.max(1, q.height);
  sizeCanvas(charts, cctx, chW, chH);
  sizeCanvas(chartBase, bctx, chW, chH);
  drawChartBase();
}

// ------------------------------------------------------------ scene layers --
var hits = [];
function hit(x, y, w, h, title, body){ hits.push({ x: x, y: y, w: w, h: h, title: title, body: body }); }

function dayFactor(hour){
  var d = Math.sin((hour - 6) / 12 * Math.PI);
  return clamp(d, 0, 1);
}

function drawSky(g, d){
  var hour = d.t.getUTCHours();
  var day = dayFactor(hour);
  var top = lerpHex(C.skyNight, C.skyDay, day);
  var bot = lerpHex(C.skyNightLow, C.skyDayLow, day);
  var grad = g.createLinearGradient(0, 0, 0, sceneH);
  grad.addColorStop(0, top); grad.addColorStop(1, bot);
  g.fillStyle = grad; g.fillRect(0, 0, sceneW, sceneH);

  // Sun or moon, small and low contrast, arcing with the hour so the
  // night-time part of the leaf wetness story is felt, not just labelled.
  var isDay = day > 0.02;
  var arc = clamp(isDay ? ((hour + 24 - 6) % 24) / 12 : ((hour + 24 - 18) % 24) / 12, 0, 1);
  var cx = sceneW * (0.08 + 0.84 * arc);
  var cy = sceneH * (0.26 - 0.17 * Math.sin(arc * Math.PI));
  var r = Math.max(5, sceneH * 0.019);
  var col = isDay ? C.sun : C.moon;
  var glow = g.createRadialGradient(cx, cy, r * 0.6, cx, cy, r * 5);
  glow.addColorStop(0, rgba(col, isDay ? 0.26 : 0.16));
  glow.addColorStop(1, rgba(col, 0));
  g.fillStyle = glow;
  g.beginPath(); g.arc(cx, cy, r * 5, 0, Math.PI * 2); g.fill();
  g.fillStyle = rgba(col, isDay ? 0.7 : 0.45);
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fill();
  hit(cx - r * 2, cy - r * 2, r * 4, r * 4, isDay ? 'Daylight' : 'Night',
      'Hour ' + PAD2(hour) + ':00 local');
}

function drawGround(g, d){
  var pts = [P(-34, -46, 0), P(MW + 34, -46, 0), P(MW + 34, MD + 46, 0), P(-34, MD + 46, 0)];
  g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
  for (var i = 1; i < 4; i++) g.lineTo(pts[i][0], pts[i][1]);
  g.closePath();
  g.fillStyle = C.ground; g.fill();
  g.strokeStyle = rgba(C.border, 0.9); g.lineWidth = 1; g.stroke();
}

function drawSoilBed(g, d){
  var col = soilColor(d.soil);
  var pts = [P(BED_U0, BED_V0, 0), P(BED_U1, BED_V0, 0), P(BED_U1, BED_V1, 0), P(BED_U0, BED_V1, 0)];
  g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
  for (var i = 1; i < 4; i++) g.lineTo(pts[i][0], pts[i][1]);
  g.closePath();
  g.fillStyle = col; g.fill();
  g.strokeStyle = rgba(C.soilWet, 0.9); g.lineWidth = 1; g.stroke();

  // Raised bed edge: a thin extruded face along the near long side.
  var a = P(BED_U0, BED_V0, 0), b = P(BED_U1, BED_V0, 0);
  var lip = 2.2 * view.s;
  g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
  g.lineTo(b[0], b[1] + lip); g.lineTo(a[0], a[1] + lip); g.closePath();
  g.fillStyle = lerpHex(col, '#000000', 0.35); g.fill();

  var xs = [pts[0][0], pts[1][0], pts[2][0], pts[3][0]];
  var ys = [pts[0][1], pts[1][1], pts[2][1], pts[3][1]];
  hit(Math.min.apply(null, xs), Math.min.apply(null, ys),
      Math.max.apply(null, xs) - Math.min.apply(null, xs),
      Math.max.apply(null, ys) - Math.min.apply(null, ys),
      'Soil bed', 'Moisture ' + fmt(d.soil, 0) + '% of field capacity');
}

var PLANTS = (function(){
  var out = [], row, k;
  for (row = 0; row < 2; row++){
    var u = row === 0 ? 0.33 * MW : 0.65 * MW;
    for (k = 0; k < 9; k++){
      var seed = row * 31 + k;
      var v = BED_V0 + 6 + (BED_V1 - BED_V0 - 12) * (k / 8);
      out.push({
        u: u + (rnd(seed) - 0.5) * 5,
        v: v + (rnd(seed + 7) - 0.5) * 5,
        h: 11 + rnd(seed + 13) * 6,
        tilt: (rnd(seed + 19) - 0.5) * 0.5,
        leaves: 4 + Math.floor(rnd(seed + 23) * 3),
        tone: rnd(seed + 29)
      });
    }
  }
  out.sort(function(a, b){ return b.v - a.v; });
  return out;
})();

function drawCrops(g, d){
  var s = view.s, i, j;
  for (i = 0; i < PLANTS.length; i++){
    var p = PLANTS[i];
    var base = P(p.u, p.v, 0);
    var topP = P(p.u + p.tilt * 3, p.v, p.h);
    var green = lerpHex(C.plantDark, C.plantLight, p.tone);
    g.strokeStyle = green; g.lineWidth = Math.max(1, 1.4 * s); g.lineCap = 'round';
    g.beginPath(); g.moveTo(base[0], base[1]); g.lineTo(topP[0], topP[1]); g.stroke();
    for (j = 0; j < p.leaves; j++){
      var f = 0.30 + 0.66 * (j / Math.max(1, p.leaves - 1));
      var side = (j % 2 === 0) ? 1 : -1;
      var lu = p.u + side * (2.6 + rnd(i * 17 + j) * 2.0);
      var lh = p.h * f;
      var c = P(lu, p.v + (rnd(i * 5 + j) - 0.5) * 2.0, lh);
      g.save();
      g.translate(c[0], c[1]);
      g.rotate(side * -0.42);
      g.fillStyle = green; g.globalAlpha = 0.92;
      g.beginPath();
      g.ellipse(0, 0, Math.max(1.5, 3.4 * s), Math.max(0.8, 1.5 * s), 0, 0, Math.PI * 2);
      g.fill();
      g.restore();
    }
  }
  var a = P(BED_U0, BED_V0, 0), b = P(BED_U1, BED_V1, 18);
  hit(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]),
      'Crop row', 'Leaf wetness ' + (d.wet ? 'wet' : 'dry'));
}

function drawInteriorAir(g, d){
  g.save();
  g.clip(SIL);
  var col = tempColor(d.tin);
  var top = py(MD, RIDGE), bot = py(0, 0);
  var grad = g.createLinearGradient(0, top, 0, bot);
  grad.addColorStop(0, rgba(col, 0.62));
  grad.addColorStop(0.55, rgba(col, 0.34));
  grad.addColorStop(1, rgba(col, 0.12));
  g.fillStyle = grad;
  g.fillRect(0, 0, sceneW, sceneH);
  g.restore();
  var a = P(0, 0, 0), b = P(MW, MD, RIDGE);
  hit(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]),
      'Interior air', 'T_in ' + fmt(d.tin, 1) + ' °C, RH_in ' + fmt(d.rh, 1) + '%');
}

function strokeArch(g, v){
  g.beginPath();
  var a = P(0, v, WALL); g.moveTo(a[0], a[1]);
  for (var j = 1; j <= 36; j++){
    var u = MW * j / 36, q = P(u, v, archH(u));
    g.lineTo(q[0], q[1]);
  }
  g.stroke();
}

function drawFrame(g){
  var s = view.s, i, j;
  // Far frame first, at reduced weight, then the near frame over the film.
  g.lineCap = 'round';
  for (i = 0; i < POST_V.length; i++){
    var v = POST_V[i];
    var near = (v === 0);
    [0, MW].forEach(function(u){
      var a = P(u, v, 0), b = P(u, v, WALL);
      g.strokeStyle = rgba(C.frame, near ? 0.95 : (u === MW ? 0.8 : 0.42));
      g.lineWidth = Math.max(1.2, (near ? 2.6 : 2.0) * s);
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      // footing tick
      g.lineWidth = Math.max(1, 1.6 * s);
      g.beginPath(); g.moveTo(a[0] - 4 * s, a[1]); g.lineTo(a[0] + 4 * s, a[1]); g.stroke();
    });
  }
  for (i = 0; i < HOOP_V.length; i++){
    var vv = HOOP_V[i];
    var edge = (vv === 0 || vv === MD);
    g.strokeStyle = rgba(C.frame, edge ? 0.9 : 0.34);
    g.lineWidth = Math.max(1, (edge ? 2.0 : 1.3) * s);
    strokeArch(g, vv);
  }
  for (i = 0; i < PURLIN_U.length; i++){
    var u2 = PURLIN_U[i];
    var a2 = P(u2, 0, archH(u2)), b2 = P(u2, MD, archH(u2));
    g.strokeStyle = rgba(C.frame, 0.30);
    g.lineWidth = Math.max(0.8, 1.1 * s);
    g.beginPath(); g.moveTo(a2[0], a2[1]); g.lineTo(b2[0], b2[1]); g.stroke();
  }
}

function drawFilm(g){
  // Panel grid between hoops and purlins, translucent, with a soft specular
  // band across the upper left of the roof so the enclosure reads as covered.
  var i, j;
  for (i = 0; i < HOOP_V.length - 1; i++){
    for (j = 0; j < PURLIN_U.length - 1; j++){
      var v0 = HOOP_V[i], v1 = HOOP_V[i + 1], u0 = PURLIN_U[j], u1 = PURLIN_U[j + 1];
      var q = [P(u0, v0, archH(u0)), P(u1, v0, archH(u1)), P(u1, v1, archH(u1)), P(u0, v1, archH(u0))];
      g.beginPath(); g.moveTo(q[0][0], q[0][1]);
      for (var k = 1; k < 4; k++) g.lineTo(q[k][0], q[k][1]);
      g.closePath();
      var mid = (u0 + u1) / 2 / MW;
      var spec = Math.exp(-Math.pow((mid - 0.27) / 0.20, 2));
      g.fillStyle = rgba(C.film, 0.05 + 0.09 * spec);
      g.fill();
      g.strokeStyle = rgba(C.film, 0.07);
      g.lineWidth = 1; g.stroke();
    }
  }
  // Gable films, kept very faint so the interior stays legible. The back one
  // is a touch stronger so the tunnel reads as closed at the far end.
  [[MD, 0.075], [0, 0.045]].forEach(function(pair){
    var v = pair[0], alpha = pair[1];
    g.beginPath();
    var a = P(0, v, 0); g.moveTo(a[0], a[1]);
    var b = P(0, v, WALL); g.lineTo(b[0], b[1]);
    for (var m = 0; m <= 30; m++){ var u = MW * m / 30, p2 = P(u, v, archH(u)); g.lineTo(p2[0], p2[1]); }
    var c = P(MW, v, 0); g.lineTo(c[0], c[1]);
    g.closePath();
    g.fillStyle = rgba(C.film, alpha); g.fill();
  });
  var q0 = P(0, 0, 0), q1 = P(MW, MD, RIDGE);
  hit(Math.min(q0[0], q1[0]), Math.min(q0[1], q1[1]), Math.abs(q1[0] - q0[0]), Math.abs(q1[1] - q0[1]) * 0.55,
      'Poly film', 'Translucent cover over the frame');
}

function ventGeometry(vf){
  var uA = VENT_U0, hA = archH(uA);
  var du = VENT_U1 - uA, dh = archH(VENT_U1) - hA;
  var th = clamp(vf, 0, 1) * 1.08;                  // up to about 62 degrees
  var cu = du * Math.cos(th) - dh * Math.sin(th);
  var ch = du * Math.sin(th) + dh * Math.cos(th);
  return { uA: uA, hA: hA, uF: uA + cu, hF: hA + ch };
}

function drawVent(g, d){
  var vg = ventGeometry(d.vent), s = view.s;
  // Aperture under the panel, only visible once it lifts.
  if (d.vent > 0.01){
    g.beginPath();
    var a0 = P(vg.uA, VENT_V0, vg.hA); g.moveTo(a0[0], a0[1]);
    for (var j = 1; j <= 10; j++){
      var u = vg.uA + (VENT_U1 - vg.uA) * j / 10, q = P(u, VENT_V0, archH(u));
      g.lineTo(q[0], q[1]);
    }
    for (var k = 10; k >= 0; k--){
      var u2 = vg.uA + (VENT_U1 - vg.uA) * k / 10, q2 = P(u2, VENT_V1, archH(u2));
      g.lineTo(q2[0], q2[1]);
    }
    g.closePath();
    g.fillStyle = rgba(C.aperture, 0.30 + 0.45 * d.vent); g.fill();
  }
  var pts = [P(vg.uA, VENT_V0, vg.hA), P(vg.uF, VENT_V0, vg.hF), P(vg.uF, VENT_V1, vg.hF), P(vg.uA, VENT_V1, vg.hA)];
  g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
  for (var m = 1; m < 4; m++) g.lineTo(pts[m][0], pts[m][1]);
  g.closePath();
  g.fillStyle = rgba(C.film, 0.06 + 0.16 * d.vent); g.fill();
  g.strokeStyle = rgba(C.blue, 0.22 + 0.62 * d.vent);
  g.lineWidth = Math.max(1, 1.5 * s); g.stroke();
  // Hinge line, always drawn so the panel reads as hinged even when shut.
  g.strokeStyle = rgba(C.frame, 0.75); g.lineWidth = Math.max(1, 1.8 * s);
  g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); g.lineTo(pts[3][0], pts[3][1]); g.stroke();

  var xs = pts.map(function(p){ return p[0]; }), ys = pts.map(function(p){ return p[1]; });
  hit(Math.min.apply(null, xs), Math.min.apply(null, ys),
      Math.max.apply(null, xs) - Math.min.apply(null, xs),
      Math.max.apply(null, ys) - Math.min.apply(null, ys),
      'Roof vent', d.vent > 0 ? fmt(d.vent * 100, 0) + '% open' : 'Closed');
  return vg;
}

function drawFan(g, d){
  var s = view.s, i;
  var hub = P(2.5, FAN_V, FAN_H);
  var mount = P(0, FAN_V, FAN_H);
  var stay = P(0, FAN_V, FAN_H + 8);
  g.strokeStyle = rgba(C.frame, 0.85); g.lineWidth = Math.max(1, 1.6 * s);
  g.beginPath(); g.moveTo(mount[0], mount[1]); g.lineTo(hub[0], hub[1]);
  g.moveTo(stay[0], stay[1]); g.lineTo(hub[0], hub[1]); g.stroke();

  function ring(r, k){
    g.beginPath();
    for (i = 0; i <= 40; i++){
      var a = i / 40 * Math.PI * 2;
      var q = P(2.5, FAN_V + r * Math.cos(a), FAN_H + r * Math.sin(a) * k);
      if (i === 0) g.moveTo(q[0], q[1]); else g.lineTo(q[0], q[1]);
    }
    g.closePath();
  }
  ring(FAN_R, 1);
  g.fillStyle = rgba(C.surface, 0.55); g.fill();
  g.strokeStyle = rgba(C.violet, d.fan > 0 ? 0.95 : 0.5);
  g.lineWidth = Math.max(1, 1.7 * s); g.stroke();
  ring(FAN_R * 0.66, 1);
  g.strokeStyle = rgba(C.violet, d.fan > 0 ? 0.45 : 0.25);
  g.lineWidth = Math.max(0.8, 1 * s); g.stroke();

  g.strokeStyle = rgba(C.violet, d.fan > 0 ? 0.95 : 0.55);
  g.lineWidth = Math.max(1.4, 2.2 * s); g.lineCap = 'round';
  for (i = 0; i < 3; i++){
    var a2 = st.fanAngle + i * Math.PI * 2 / 3;
    var tip = P(2.5, FAN_V + FAN_R * 0.82 * Math.cos(a2), FAN_H + FAN_R * 0.82 * Math.sin(a2));
    g.beginPath(); g.moveTo(hub[0], hub[1]); g.lineTo(tip[0], tip[1]); g.stroke();
  }
  g.fillStyle = rgba(C.violet, 0.9);
  g.beginPath(); g.arc(hub[0], hub[1], Math.max(1.5, 2.2 * s), 0, Math.PI * 2); g.fill();

  var r2 = FAN_R * view.s * 1.2;
  hit(hub[0] - r2, hub[1] - r2 * 1.3, r2 * 2, r2 * 2.6,
      'Circulation fan', d.fan > 0 ? fmt(d.fan * 100, 0) + '% power' : 'Off');
}

var DROPS = (function(){
  var out = [];
  for (var i = 0; i < 15; i++){
    out.push({ u: MW * (0.06 + 0.88 * rnd(i * 3 + 1)), v: MD * (0.04 + 0.9 * rnd(i * 3 + 2)),
               r: 1.3 + rnd(i * 3 + 5) * 1.4 });
  }
  return out;
})();

function drawDroplets(g){
  var s = view.s, i;
  g.save();
  for (i = 0; i < DROPS.length; i++){
    var dp = DROPS[i];
    var q = P(dp.u, dp.v, archH(dp.u) - 1.2);
    g.fillStyle = rgba(C.drop, 0.55);
    g.beginPath();
    g.ellipse(q[0], q[1], Math.max(1, dp.r * s), Math.max(1.2, dp.r * 1.35 * s), 0, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = rgba('#ffffff', 0.30);
    g.beginPath();
    g.arc(q[0] - dp.r * 0.3 * s, q[1] - dp.r * 0.4 * s, Math.max(0.5, dp.r * 0.28 * s), 0, Math.PI * 2);
    g.fill();
  }
  g.restore();
}

function arrow(g, a, b, c, alpha, col){
  g.save();
  g.strokeStyle = rgba(col, alpha);
  g.lineWidth = Math.max(1, 1.5 * view.s);
  g.setLineDash([6 * view.s, 5 * view.s]);
  g.lineDashOffset = -st.dash;
  g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(b[0], b[1], c[0], c[1]); g.stroke();
  g.setLineDash([]);
  var ang = Math.atan2(c[1] - b[1], c[0] - b[0]);
  var hl = 6 * view.s;
  g.fillStyle = rgba(col, alpha);
  g.beginPath();
  g.moveTo(c[0], c[1]);
  g.lineTo(c[0] - hl * Math.cos(ang - 0.42), c[1] - hl * Math.sin(ang - 0.42));
  g.lineTo(c[0] - hl * Math.cos(ang + 0.42), c[1] - hl * Math.sin(ang + 0.42));
  g.closePath(); g.fill();
  g.restore();
}

function drawAirflow(g, d, vg){
  if (d.vent > 0.005){
    var a = 0.22 + 0.55 * d.vent;
    arrow(g, P(0.42 * MW, 0.5 * MD, 22), P(0.48 * MW, 0.5 * MD, 58), P(vg.uF - 3, 0.5 * MD, vg.hF + 12), a, C.blue);
    arrow(g, P(0.20 * MW, 0.3 * MD, 12), P(0.30 * MW, 0.3 * MD, 46), P(0.44 * MW, 0.3 * MD, 66), a * 0.7, C.blue);
  }
  if (d.fan > 0.005){
    var f = 0.22 + 0.58 * d.fan;
    arrow(g, P(9, FAN_V, FAN_H), P(0.42 * MW, FAN_V, FAN_H + 6), P(0.74 * MW, FAN_V, FAN_H - 2), f, C.violet);
    arrow(g, P(0.78 * MW, 0.62 * MD, 9), P(0.44 * MW, 0.62 * MD, 5), P(0.16 * MW, 0.62 * MD, 11), f * 0.65, C.violet);
  }
}

function drawRain(g, d){
  if (!(d.rain > 0)) return;
  var count = Math.round(clamp(18 + d.rain * 9, 18, 88));
  g.save();
  var outside = new Path2D();
  outside.rect(0, 0, sceneW, sceneH);
  outside.addPath(SIL);
  g.clip(outside, 'evenodd');
  g.strokeStyle = rgba(C.rain, clamp(0.16 + d.rain * 0.035, 0.16, 0.42));
  g.lineWidth = Math.max(0.8, 1.0 * view.s);
  var len = 12 + clamp(d.rain, 0, 8) * 1.6;
  for (var i = 0; i < count; i++){
    var x = rnd(i * 2 + 1) * (sceneW + 120) - 60;
    var y = (rnd(i * 2 + 2) * sceneH + st.rainPhase * (120 + rnd(i) * 90)) % (sceneH + 60) - 30;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x - len * 0.34, y + len); g.stroke();
  }
  g.restore();
  hit(0, 0, sceneW, sceneH * 0.30, 'Rain', fmt(d.rain, 1) + ' mm this hour');
}

// ------------------------------------------------------------ annotations --
function labelFont(){ return Math.round(clamp(11 * view.s * 2.1, 9, 12)); }

function leader(g, anchor, tx, ty, text, align){
  var fs = labelFont();
  g.save();
  g.strokeStyle = rgba(C.muted, 0.5);
  g.lineWidth = 1;
  var elbowX = align === 'right' ? tx + 8 : tx - 8;
  g.beginPath();
  g.moveTo(anchor[0], anchor[1]);
  g.lineTo(elbowX, ty);
  g.lineTo(align === 'right' ? tx + 2 : tx - 2, ty);
  g.stroke();
  g.fillStyle = rgba(C.muted, 0.72);
  g.beginPath(); g.arc(anchor[0], anchor[1], 2, 0, Math.PI * 2); g.fill();
  g.font = '500 ' + fs + 'px ' + C.font;
  g.textAlign = align === 'right' ? 'left' : 'right';
  g.textBaseline = 'middle';
  g.fillStyle = rgba(C.muted, 0.92);
  g.fillText(text, tx, ty);
  g.restore();
}

function drawLabels(g, vg){
  if (sceneW < 460) return;
  var midV = (VENT_V0 + VENT_V1) / 2;
  leader(g, P((vg.uA + vg.uF) / 2, midV, (vg.hA + vg.hF) / 2 + 1), px(196, 0), py(0, 108), 'Roof vent', 'right');
  leader(g, P(0.20 * MW, 0.62 * MD, archH(0.20 * MW)), px(-50, 0), py(0, 100), 'Poly film', 'left');
  leader(g, P(2.5, FAN_V, FAN_H), px(-50, 0), py(0, 40), 'Circulation fan', 'left');
  leader(g, P(BED_U0 + 4, BED_V0 + 6, 0), px(-50, 0), py(0, -6), 'Soil bed', 'left');
  leader(g, P(0.65 * MW, 0.80 * MD, 12), px(196, 0), py(0, 24), 'Crop row', 'right');
}

function chip(g, anchor, dx, dy, id, value, dotColor){
  var fs = clamp(Math.round(10 * view.s * 2.1), 9, 11);
  g.save();
  g.font = '600 ' + fs + 'px ' + C.font;
  var label = id + '  ' + value;
  var w = g.measureText(label).width + 22, h = fs + 11;
  var x = anchor[0] + dx, y = anchor[1] + dy - h / 2;
  x = clamp(x, 2, sceneW - w - 2); y = clamp(y, 2, sceneH - h - 2);
  g.strokeStyle = rgba(C.muted, 0.45); g.lineWidth = 1;
  g.beginPath(); g.moveTo(anchor[0], anchor[1]); g.lineTo(x + (dx < 0 ? w : 0), y + h / 2); g.stroke();
  g.fillStyle = rgba(C.surface2, 0.94);
  g.beginPath();
  if (g.roundRect) { g.roundRect(x, y, w, h, 4); } else { g.rect(x, y, w, h); }
  g.fill();
  g.strokeStyle = rgba(C.border, 1); g.stroke();
  g.fillStyle = dotColor;
  g.beginPath(); g.arc(x + 8, y + h / 2, 3, 0, Math.PI * 2); g.fill();
  g.fillStyle = C.text;
  g.textAlign = 'left'; g.textBaseline = 'middle';
  g.fillText(label, x + 15, y + h / 2 + 0.5);
  g.restore();
  hit(x, y, w, h, id, value);
}

function drawSensors(g, d){
  if (sceneW < 420) return;
  var rawPct = DATA.thresholds.rawPct, rhMin = DATA.thresholds.rhWetMin;
  // Air node on a post, soil probe in the bed, leaf wetness at canopy height.
  // These three are the only sensed quantities the cached run actually
  // carries, so they are the only nodes drawn.
  chip(g, P(MW, 0.10 * MD, 27), 16, -10, 'AT-01',
       fmt(d.tin, 1) + ' °C  ' + fmt(d.rh, 0) + '%',
       d.rh >= rhMin ? C.critical : C.good);
  chip(g, P(0.28 * MW, 0.58 * MD, 0), -22, 30, 'SM-01',
       fmt(d.soil, 0) + '%',
       d.soil <= rawPct ? C.critical : C.good);
  chip(g, P(0.60 * MW, 0.34 * MD, 15), -18, -26, 'LW-01',
       d.wet ? 'wet' : 'dry',
       d.wet ? C.critical : C.good);
}

// --------------------------------------------------------------- the scene --
function drawScene(d){
  var g = sctx;
  hits = [];
  g.clearRect(0, 0, sceneW, sceneH);
  drawSky(g, d);
  drawGround(g, d);
  drawInteriorAir(g, d);
  drawSoilBed(g, d);
  drawCrops(g, d);
  drawFan(g, d);
  drawFrame(g);
  drawFilm(g);
  var vg = drawVent(g, d);
  if (d.wet) drawDroplets(g);
  drawAirflow(g, d, vg);
  drawRain(g, d);
  drawSensors(g, d);
  drawLabels(g, vg);
}

// -------------------------------------------------------------- the charts --
var PANELS = [
  { key: 'T_in', extra: 'T_out', color: C.blue, extraColor: C.orange,
    title: 'Interior temperature (°C)', auto: true, wet: false },
  { key: 'RH_in', color: C.blue, title: 'Humidity (%), shaded where the leaves are wet',
    lo: 0, hi: 100, wet: true },
  { key: 'soil_pct', color: C.aqua, title: 'Soil moisture (%)', lo: 0, hi: 100, wet: false }
];
var PAD_L = 8, PAD_R = 46, PAD_T = 8, PAD_B = 6, GAP = 8;

function panelRect(i){
  var h = (chH - PAD_T - PAD_B - GAP * (PANELS.length - 1)) / PANELS.length;
  return { x: PAD_L, y: PAD_T + i * (h + GAP), w: chW - PAD_L - PAD_R, h: h };
}

function xFor(i, r){ return r.x + (N <= 1 ? 0 : (i / (N - 1)) * r.w); }
function idxForX(x, r){ return Math.round(clamp((x - r.x) / Math.max(1, r.w), 0, 1) * (N - 1)); }

function drawChartBase(){
  var g = bctx, i, j;
  g.clearRect(0, 0, chW, chH);
  g.fillStyle = C.surface; g.fillRect(0, 0, chW, chH);
  for (i = 0; i < PANELS.length; i++){
    var p = PANELS[i], r = panelRect(i), arr = S[p.key];
    var lo, hi;
    if (p.auto){
      lo = Infinity; hi = -Infinity;
      for (j = 0; j < N; j++){ if (arr[j] < lo) lo = arr[j]; if (arr[j] > hi) hi = arr[j]; }
      if (p.extra){
        var e = S[p.extra];
        for (j = 0; j < N; j++){ if (e[j] < lo) lo = e[j]; if (e[j] > hi) hi = e[j]; }
      }
      var padv = (hi - lo) * 0.08 || 1;
      lo -= padv; hi += padv;
    } else { lo = p.lo; hi = p.hi; }
    p._lo = lo; p._hi = hi; p._r = r;
    var yFor = function(v){ return r.y + r.h - ((v - lo) / Math.max(1e-9, hi - lo)) * r.h; };
    p._yFor = yFor;

    if (p.wet){
      g.fillStyle = rgba(C.critical, 0.16);
      var runStart = -1;
      for (j = 0; j <= N; j++){
        var on = j < N && WET.charAt(j) === '1';
        if (on && runStart < 0) runStart = j;
        if (!on && runStart >= 0){
          var x0 = xFor(runStart, r), x1 = xFor(j - 1, r);
          g.fillRect(x0, r.y, Math.max(1, x1 - x0), r.h);
          runStart = -1;
        }
      }
    }
    g.strokeStyle = rgba(C.grid, 1); g.lineWidth = 1;
    g.beginPath(); g.moveTo(r.x, r.y + r.h + 0.5); g.lineTo(r.x + r.w, r.y + r.h + 0.5); g.stroke();
    g.beginPath(); g.moveTo(r.x, r.y + 0.5); g.lineTo(r.x + r.w, r.y + 0.5); g.stroke();

    if (p.extra){
      g.strokeStyle = rgba(p.extraColor, 0.55); g.lineWidth = 1;
      g.beginPath();
      for (j = 0; j < N; j++){
        var xx = xFor(j, r), yy = yFor(S[p.extra][j]);
        if (j === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy);
      }
      g.stroke();
    }
    g.strokeStyle = p.color; g.lineWidth = 1.4;
    g.beginPath();
    for (j = 0; j < N; j++){
      var x2 = xFor(j, r), y2 = yFor(arr[j]);
      if (j === 0) g.moveTo(x2, y2); else g.lineTo(x2, y2);
    }
    g.stroke();

    g.font = '600 10.5px ' + C.font;
    g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillStyle = rgba(C.muted, 0.95);
    g.fillText(p.title, r.x + 4, r.y + 3);
    g.textAlign = 'left'; g.textBaseline = 'middle';
    g.font = '500 10px ' + C.font;
    g.fillStyle = rgba(C.muted, 0.8);
    g.fillText(fmt(hi, hi >= 100 ? 0 : 1), r.x + r.w + 6, r.y + 5);
    g.fillText(fmt(lo, lo >= 100 ? 0 : 1), r.x + r.w + 6, r.y + r.h - 5);
  }
}

function drawCharts(d){
  var g = cctx, i;
  g.clearRect(0, 0, chW, chH);
  g.drawImage(chartBase, 0, 0, chW, chH);
  for (i = 0; i < PANELS.length; i++){
    var p = PANELS[i], r = p._r;
    if (!r) continue;
    var x = xFor(d.i, r);
    g.strokeStyle = rgba(C.text, 0.55); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x + 0.5, r.y); g.lineTo(x + 0.5, r.y + r.h); g.stroke();
    var v = S[p.key][d.i];
    g.fillStyle = p.color;
    g.beginPath(); g.arc(x, p._yFor(v), 2.8, 0, Math.PI * 2); g.fill();
    // Current value pinned to the panel's top right, clear of both the title
    // on the left and the axis range labels in the gutter.
    g.font = '600 11px ' + C.font;
    g.textAlign = 'right';
    g.textBaseline = 'top';
    g.fillStyle = C.text;
    var txt = fmt(v, 1);
    if (p.extra) txt += '   ' + fmt(S[p.extra][d.i], 1) + ' out';
    g.fillText(txt, r.x + r.w - 4, r.y + 3);
  }
}

// ------------------------------------------------------------- DOM readout --
var elRun = document.getElementById('tw-run');
var elTs = document.getElementById('tw-ts');
var elHour = document.getElementById('tw-hour');
var elScrub = document.getElementById('tw-scrub');
var elPlay = document.getElementById('tw-play');
var elTip = document.getElementById('tw-tip');

elRun.innerHTML = DATA.meta.controller + ' controller <span>' + DATA.meta.regime + ', ' + DATA.meta.year + '</span>';
elScrub.max = String(N - 1);

function stateChip(id, on, text, color){
  var el = document.getElementById(id);
  el.innerHTML = '<span class="tw-dot" style="background:' + (on ? color : C.muted) + '"></span>'
               + '<span style="color:' + (on ? C.text : C.muted) + '">' + text + '</span>';
}

var lastDom = -1;
function updateDom(d){
  if (d.i === lastDom) return;
  lastDom = d.i;
  elTs.textContent = stampOf(d.t);
  elHour.textContent = 'hour ' + (d.i + 1) + ' of ' + N;
  if (document.activeElement !== elScrub) elScrub.value = String(d.i);

  document.getElementById('val-vent').textContent = fmt(d.vent * 100, 0) + '%';
  stateChip('state-vent', d.vent > 0, d.vent > 0 ? 'Open' : 'Closed', C.blue);
  document.getElementById('val-fan').textContent = fmt(d.fan * 100, 0) + '%';
  stateChip('state-fan', d.fan > 0, d.fan > 0 ? 'Running' : 'Off', C.violet);
  document.getElementById('val-irr').textContent = fmt(d.irr, 1) + ' mm';
  stateChip('state-irr', d.irr > 0, d.irr > 0 ? 'Applying' : 'Closed', C.aqua);

  var climate = RCLIM.table[RCLIM.index[d.i]];
  var water = RWATER.table[RWATER.index[d.i]];
  var wv = document.getElementById('why-vent'), wf = document.getElementById('why-fan'),
      wi = document.getElementById('why-irr');
  wv.textContent = climate; wv.title = climate;
  wf.textContent = climate; wf.title = climate;
  wi.textContent = water; wi.title = water;
}

// ------------------------------------------------------------ interaction --
function setIdx(i, stopPlay){
  st.idxF = ((i % N) + N) % N;
  st.idx = Math.floor(st.idxF);
  if (stopPlay) setPlaying(false);
}
function setPlaying(v){
  st.playing = v;
  elPlay.textContent = v ? 'Pause' : 'Play';
  elPlay.classList.toggle('tw-on', v);
}
function setSpeed(h){
  st.hps = h;
  var b = document.querySelectorAll('.tw-sp');
  for (var i = 0; i < b.length; i++) b[i].classList.toggle('tw-on', Number(b[i].dataset.hps) === h);
}

elPlay.addEventListener('click', function(){ setPlaying(!st.playing); root.focus(); });
document.getElementById('tw-back').addEventListener('click', function(){ setIdx(st.idx - 1, true); root.focus(); });
document.getElementById('tw-fwd').addEventListener('click', function(){ setIdx(st.idx + 1, true); root.focus(); });
var spBtns = document.querySelectorAll('.tw-sp');
for (var bi = 0; bi < spBtns.length; bi++){
  spBtns[bi].addEventListener('click', function(e){ setSpeed(Number(e.currentTarget.dataset.hps)); root.focus(); });
}
elScrub.addEventListener('input', function(){ setIdx(Number(elScrub.value), true); });

// Keys are bound on the panel, not the document, and only act while the
// panel itself holds focus, so Streamlit's own shortcuts stay untouched.
root.addEventListener('keydown', function(e){
  if (!root.contains(document.activeElement)) return;
  if (e.key === ' ' || e.code === 'Space'){ e.preventDefault(); setPlaying(!st.playing); }
  else if (e.key === 'ArrowLeft'){ e.preventDefault(); setIdx(st.idx - 1, true); }
  else if (e.key === 'ArrowRight'){ e.preventDefault(); setIdx(st.idx + 1, true); }
});
var elFocus = document.getElementById('tw-focus');
root.addEventListener('focusin', function(){ elFocus.textContent = 'Space plays, arrows step by hour'; });
root.addEventListener('focusout', function(){
  if (!root.contains(document.activeElement)) elFocus.textContent = 'Click the panel to enable space and arrow keys';
});

function chartSeek(e){
  var rect = charts.getBoundingClientRect();
  var x = e.clientX - rect.left;
  var r = panelRect(0);
  setIdx(idxForX(x, r), true);
}
charts.addEventListener('mousedown', function(e){ chartSeek(e); charts._drag = true; root.focus(); });
window.addEventListener('mouseup', function(){ charts._drag = false; });
charts.addEventListener('mousemove', function(e){ if (charts._drag) chartSeek(e); });

scene.addEventListener('mousemove', function(e){
  var rect = scene.getBoundingClientRect();
  var x = e.clientX - rect.left, y = e.clientY - rect.top, found = null;
  for (var i = hits.length - 1; i >= 0; i--){
    var h = hits[i];
    if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h){ found = h; break; }
  }
  if (!found){ elTip.style.opacity = '0'; return; }
  elTip.innerHTML = '<b>' + found.title + '</b><br>' + found.body;
  elTip.style.opacity = '1';
  var tw = elTip.offsetWidth, th = elTip.offsetHeight;
  elTip.style.left = clamp(x + 14, 4, Math.max(4, sceneW - tw - 4)) + 'px';
  elTip.style.top = clamp(y - th - 10, 4, Math.max(4, sceneH - th - 4)) + 'px';
});
scene.addEventListener('mouseleave', function(){ elTip.style.opacity = '0'; });

// ------------------------------------------------------------- frame loop --
var visible = true, onScreen = true, last = 0, frameMs = 0, frames = 0, lastReport = 0;

document.addEventListener('visibilitychange', function(){
  visible = !document.hidden;
  if (visible) { last = 0; requestAnimationFrame(tick); }
});
if (window.IntersectionObserver){
  new IntersectionObserver(function(entries){
    onScreen = entries[0].isIntersecting;
    if (onScreen) { last = 0; requestAnimationFrame(tick); }
  }, { threshold: 0 }).observe(root);
}
if (window.ResizeObserver){
  var ro = new ResizeObserver(function(){ layout(); lastDom = -1; });
  ro.observe(sceneWrap); ro.observe(chartsWrap);
} else {
  window.addEventListener('resize', function(){ layout(); lastDom = -1; });
}

function tick(now){
  if (!visible || !onScreen) return;
  // Guarded per-frame, not just at boot: a throw here would otherwise kill
  // the rAF loop silently (requestAnimationFrame callbacks run outside the
  // boot try/catch below), leaving a half-drawn, unresponsive panel instead
  // of falling back to the static cross section.
  try {
    var dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
    last = now;
    st.t += dt;

    if (st.playing){
      st.idxF = (st.idxF + dt * st.hps) % N;
      st.idx = Math.floor(st.idxF);
    }
    var d = at(st.idx);
    st.fanAngle += dt * d.fan * 9.0;
    st.rainPhase += dt * 0.55;
    st.dash += dt * (28 * Math.max(d.vent, d.fan));

    var t0 = performance.now();
    drawScene(d);
    drawCharts(d);
    updateDom(d);
    frameMs += performance.now() - t0;
    frames++;
    if (now - lastReport > 2000){
      window.__twinFrameMs = frames ? frameMs / frames : 0;
      if (window.console && window.console.log){
        console.log('[LiveTwin] draw ' + window.__twinFrameMs.toFixed(2) + ' ms/frame over ' + frames + ' frames');
      }
      frameMs = 0; frames = 0; lastReport = now;
    }
  } catch (err) {
    showFallback(err);
    return;
  }
  requestAnimationFrame(tick);
}

// A small read/write hook so the panel can be driven from a test harness
// (jump to a specific hour, read the measured frame time) without clicking.
// It only moves the playhead; it cannot change a single simulated value.
window.__twin = {
  seek: function(i){ setIdx(i, true); },
  play: setPlaying,
  speed: setSpeed,
  state: function(){ return { idx: st.idx, playing: st.playing, hps: st.hps, n: N,
                              frameMs: window.__twinFrameMs || 0 }; }
};

// ------------------------------------------------------------------- boot --
layout();
setSpeed(5);
setPlaying(false);
setIdx(0, false);
requestAnimationFrame(tick);
booted = true;
clearTimeout(watchdog);

} catch (err) {
  clearTimeout(watchdog);
  showFallback(err);
}
})();
"""
