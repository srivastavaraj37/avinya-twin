# avinya-twin

A physics-based digital twin of a 100 m² plastic polyhouse growing tomato in
Guwahati, Assam, India (26.19°N, 91.69°E), used to compare vent/fan/irrigation
controllers on the same simulated plant. The central engineering finding: in
Guwahati's monsoon, **ventilation alone has almost no authority over indoor
humidity at night** -- during JJA monsoon nights, 82.7% of hours have less
than 0.2 kPa of achievable VPD range between vents fully closed and fully
open, because outdoor air is itself already near-saturated (mean RH_out
85.9%). Opening vents at night mostly just swaps saturated indoor air for
near-saturated outdoor air. That's why a second actuator -- a circulation fan
that thins the leaf boundary layer instead of exchanging bulk air -- exists in
this project, and why a predictive controller that uses it beats a naive
timer on leaf-wetness even though it can't move VPD any more than the naive
controller can.

**Live app**: https://guwahati-polyhouse.streamlit.app

No machine learning, no curve-fitting. Every physical constant is sourced
(FAO-56, Tetens, ASHRAE, Wallin/BLITECAST, FAST/TOMCAST) and every function is
unit-annotated. Simulator dependencies: numpy, pandas, requests, matplotlib,
pytest (the `config.yaml` loader is a small hand-rolled parser rather than a
PyYAML dependency; see `sim/config.py`). The dashboard adds streamlit, plotly,
and pyarrow on top -- see `requirements.txt`.

## Headline results (5-year mean ± sd, Predictive vs. Fixed)

Fixed is the naive timer schedule most smallholders actually run today
(fixed vent window, fixed irrigation clock, no fan). Full methodology,
every controller's logic, and the per-year numbers behind these means are
in the app's **Headline Results** and **Validation & Limitations** pages,
and in `results/regime_A_summary.csv` / `results/regime_B_summary.csv`.

### Regime A -- Monsoon (Jun 1 - Aug 29)

| metric | Fixed | Threshold | **Predictive (MPC)** |
|---|---|---|---|
| Water (L/m²) | 720.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Leaf-wet hours | 1453.0 ± 55.9 | 1037.6 ± 57.5 | **785.8 ± 52.3** |
| Alternaria (early blight) risk | 28.6 ± 10.3 | 1.2 ± 1.3 | **0.4 ± 0.5** |
| Wallin (late blight) DSV | 23.4 ± 11.1 | 4.4 ± 2.1 | **2.0 ± 0.7** |
| % hours in achievable VPD band | 12.31 ± 1.71 | 8.07 ± 1.00 | 1.82 ± 1.12 |
| Vent actuations | 180.0 ± 0.0 | 199.4 ± 13.9 | 299.4 ± 39.4 |
| Fan (kWh) | 0.00 ± 0.00 | 316.35 ± 12.87 | 506.85 ± 17.28 |

### Regime B -- Dry season (Nov 1 - Jan 29)

| metric | Fixed | Threshold | **Predictive (MPC)** |
|---|---|---|---|
| Water (L/m²) | 720.0 ± 0.0 | 121.0 ± 11.4 | 121.0 ± 11.4 |
| Leaf-wet hours | 1337.8 ± 31.5 | 1110.6 ± 48.8 | **595.6 ± 88.1** |
| Alternaria (early blight) risk | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Wallin (late blight) DSV | 31.6 ± 13.4 | 0.4 ± 0.5 | **0.0 ± 0.0** |
| % hours in achievable VPD band | 12.19 ± 1.14 | 3.17 ± 0.57 | 1.61 ± 0.56 |
| Vent actuations | 180.0 ± 0.0 | 585.6 ± 53.8 | **132.0 ± 39.5** |
| Fan (kWh) | 0.00 ± 0.00 | 323.5 ± 14.05 | 535.7 ± 2.74 |

Predictive is not a strict win on every metric -- it scores *lowest* on "%
hours in achievable VPD band" and costs more fan energy than Threshold. Both
are stated plainly, with the reasoning, on the app's Headline Results and
Controller Comparison pages (search "working as designed" and "what this
costs to run").

## Honest limitations (summary)

1. The fan's leaf-wetness threshold mapping (90%→96% RH_in as fan_frac goes
   0→1) is a modelling assumption, not a measured boundary-layer coefficient.
2. Weather is ERA5 reanalysis on a ~9 km grid cell, not a Guwahati station
   observation.
3. The Wallin DSV late-blight model is temperate-climate-calibrated and
   reads structurally low in Guwahati's monsoon regardless of controller --
   retained specifically to show that mismatch, not as the climate-appropriate
   signal (Alternaria was added for that reason).
4. Irrigation demand (ETc) is computed from outdoor weather only, so no
   vent/fan controller can differentiate water use from another rain-aware
   one -- confirmed structurally, not a tuning gap.
5. The crop calendar uses tomato's FAO-56 coefficients for both regimes, even
   though tomato is a *rabi* (winter-sown) crop in Assam; Regime A should be
   read as modelling off-season cultivation generally.

Full derivation of each point, plus the V1-V8 physical-realism gate table,
is on the app's **Validation & Limitations** page and in `CLAUDE.md`.

## What this models

- **Weather**: real hourly Open-Meteo ERA5 reanalysis for Guwahati, 2021
  through the latest complete day, one CSV per year (`data/fetch_weather.py`).
- **Psychrometrics**: saturation vapour pressure, absolute humidity, VPD,
  dewpoint (`sim/psychro.py`).
- **Reference evapotranspiration**: FAO-56 Penman-Monteith, hourly form
  (`sim/et0.py`).
- **Soil water**: FAO-56 single-layer depletion bucket (`sim/soil.py`).
- **Polyhouse**: single-zone lumped energy + vapour balance, coupling
  outdoor weather to indoor temperature/humidity via solar gain,
  conduction, ventilation, and crop transpiration (`sim/polyhouse.py`).
- **Disease**: Wallin (1962) late-blight Disease Severity Value, as used in
  BLITECAST (`sim/disease.py`).
- **Engine**: hourly simulation loop coupling all of the above, driven by
  injected vent/irrigation/fan policy callables (`sim/engine.py`).
- **Controllers**: Fixed (timer), Threshold (reactive bang-bang), and
  Predictive (12-hour receding-horizon MPC over a joint vent+fan candidate
  search) -- all in `controllers/`, all compared against the identical
  physical plant above. See the Headline Results section above and the
  app's Controller Comparison page.

## Equations

### Psychrometrics (`sim/psychro.py`)

Saturation vapour pressure (Tetens 1930, as adopted by FAO-56 Eq. 11):

```
e_s(T) = 0.6108 * exp(17.27*T / (T + 237.3))      [kPa, T in degC]
```

Vapour pressure deficit: `VPD = e_s(T) * (1 - RH/100)`.

Absolute humidity via the ideal gas law for water vapour
(R_v = 461.5 J/(kg·K), from R_universal / M_water):

```
chi = e * 1000 / (R_v * (T + 273.15))             [kg/m3]
```

Dewpoint is the Tetens formula inverted at the actual vapour pressure.

### Reference ET (`sim/et0.py`)

FAO-56 Penman-Monteith, hourly form (Allen et al. 1998, Eq. 53):

```
ET0 = [0.408*Delta*(Rn-G) + gamma*(37/(T+273))*u2*(e_s-e_a)]
      / [Delta + gamma*(1 + 0.34*u2)]
```

- `Delta` = slope of the saturation vapour pressure curve (Eq. 13)
- `gamma` = psychrometric constant from elevation-derived pressure (Eq. 7-8)
- `u2` = wind speed adjusted from 10 m to the 2 m reference height (Eq. 47)
- `Rn = (1-albedo)*Rs - Rnl`, `Rs` converted from W/m2 to MJ/m2/hr (×0.0036)
- `Rnl` uses the FAO-56 hourly Stefan-Boltzmann formulation, but derives its
  cloudiness correction directly from Open-Meteo's `cloud_cover` column
  instead of the Rs/Rso clear-sky ratio (unstable at night); see the
  docstring in `sim/et0.py` for the full justification.
- `G` (soil heat flux) is FAO-56's day/night step fraction (0.1×Rn day,
  0.5×Rn night), smoothly blended across a narrow twilight band instead of
  a hard step, so ET0(I_solar) stays continuous (and monotonic) through
  sunrise/sunset.

Crop coefficients follow FAO-56 Table 12 for tomato: `Kc` = 0.6 (initial,
days 0-29), 0.9 (development, 30-69), 1.15 (mid-season, 70-119), 0.80
(late, 120+). `ETc = Kc * ET0`.

### Soil water balance (`sim/soil.py`)

FAO-56 Ch. 8 bucket model:

```
TAW = (FC - WP) * Zr * 1000     [mm, total available water]
RAW = p * TAW                    [mm, readily available water]
```

Each step adds irrigation+rain (excess beyond field capacity becomes
tracked runoff), then subtracts ETc (capped at the water actually
available, so depletion `D` never exceeds `TAW`).

### Polyhouse energy + vapour balance (`sim/polyhouse.py`)

```
C_eff * dT/dt = Q_solar + Q_conduction + Q_ventilation - Q_latent
V * dchi/dt   = E - Q_vol * (chi_in - chi_out)
```

- `Q_solar = tau * I_solar * A_floor`
- `Q_cond  = U * A_cover * (T_out - T_in)`
- `Q_vent  = m_dot * cp_air * (T_out - T_in)`, `m_dot = rho_air * V * ACH/3600`
- `ACH = ACH_min + (ACH_max - ACH_min) * vent_frac`,
  `ACH_max = ACH_max_base * (1 + wind_coeff * wind)`
- Transpiration (Stanghellini-type, two terms):
  `E = A_floor * (coeff_radiation * I_in/lambda_vap + coeff_vpd * LAI * max(VPD_in, 0))`
- `chi` is clamped to `chi_sat(T_in)` each step; the excess condenses on
  the cover and accumulates in `condensed_kg`.

**Integration note**: both balances are linear relaxations of the form
`dy/dt = k*(y_eq - y)` once weather/vent-fraction are frozen for a
sub-step. They are solved *exactly* over each 300 s sub-step
(`y_eq + (y0-y_eq)*exp(-k*dt)`) rather than with a plain forward-Euler
difference. This is still fully explicit (no iteration), but is
unconditionally stable -- plain forward Euler is **not** stable here: the
vapour-exchange rate constant routinely exceeds the forward-Euler stability
limit (`dt*k > 2`) at high ventilation, which was verified to make RH_in
oscillate between 0% and 100% every sub-step before this fix. See the
`sim/polyhouse.py` module docstring for the full derivation.

### Disease: Wallin DSV and Alternaria risk (`sim/disease.py`)

A continuous leaf-wetness period (RH_in > a fan-dependent threshold, 90% at
fan_frac=0 rising to 96% at fan_frac=1 -- see the polyhouse fan section
above) is closed out at day boundaries or when wetness ends; its mean
temperature and duration are scored by **two** independent models, run
side by side and reported separately (never pooled):

- **Wallin (1962) DSV** -- the late-blight (*Phytophthora infestans*) model
  used by BLITECAST, temperate-climate-calibrated (favourable temperature
  band tops out at 26.6 degC).
- **Alternaria risk** -- an early-blight (*Alternaria solani*) model with a
  warmer favourable band (24-29 degC), added 2026-08-11.

**Which model is valid for which climate, and why this matters:**

Guwahati's JJA monsoon ambient T_out averages 28.7 degC, already above
Wallin's 26.6 degC favourable ceiling on 89 of 92 monsoon days before any
polyhouse warming is added (see CLAUDE.md's Gate 6b finding). Applying
Wallin here without checking its calibration range would have been an
error: cumulative DSV over a 90-day monsoon run in this climate is
structurally near-zero (0-11 across controllers, see `results/`), not
because disease pressure is actually low, but because the *table itself*
assumes late blight can't progress in heat that Guwahati's monsoon nights
routinely exceed. Read literally, that near-zero DSV would (wrongly) say
"disease risk isn't a real problem here" -- the opposite of the true
picture for a warm, humid growing region. Wallin is retained and reported
*because* this mismatch is itself the finding worth showing, not because
it's the right tool for this job.

Alternaria solani (early blight) is the disease actually calibrated for
warm-humid tomato-growing conditions: established early-blight forecasting
systems (FAST, Madden/Pennypacker/MacNab 1978; TOMCAST, Pitblado 1992) use
a favourable temperature window during the leaf-wetness period that extends
into the high 20s degC, matching Guwahati's regime instead of excluding it.
The two-tier rule implemented here (24-29 degC, >=10 continuous wet hours =
1 risk unit, >=16 hours = 2 units, accumulated seasonally) is a simplified
scheme built for this project in that same spirit, not a literal
reproduction of the published FAST/TOMCAST tables -- see `sim/disease.py`'s
module docstring for the exact citations and the scope of what is and isn't
reproduced from them.

**The general lesson**: a disease-forecast model is only as good as its
calibration range. Check a model's stated favourable conditions against the
target climate *before* trusting its output, especially when porting a
model between climates it wasn't built for (temperate -> tropical here) --
a structurally-near-zero result can look like "no problem" when it's
actually "wrong tool."

## Parameters (`config.yaml`)

| Parameter | Value | Unit | Source |
|---|---|---|---|
| `location.latitude/longitude` | 26.19 / 91.69 | deg | Guwahati, Assam |
| `polyhouse.area_floor_m2` | 100.0 | m² | project spec |
| `polyhouse.area_cover_m2` | 203.0 | m² | derived: Quonset/tunnel geometry, see `config.yaml` comment |
| `polyhouse.height_m` | 3.0 | m | assumed mean internal height |
| `polyhouse.cover_tau` | 0.65 | - | typical PE film shortwave transmissivity |
| `polyhouse.cover_U_Wm2K` | 6.0 | W/m²/K | ASHRAE single-layer PE film range (5-7) |
| `polyhouse.C_eff_JK_per_m2` | 30000 | J/K/m² | effective thermal mass (crop+soil+structure) |
| `polyhouse.albedo` | 0.23 | - | FAO-56 reference crop albedo |
| `ventilation.ach_min` | 0.5 | 1/hr | infiltration baseline |
| `ventilation.ach_max_base` | 30.0 | 1/hr | vents fully open, zero wind |
| `ventilation.wind_coeff` | 0.15 | 1/(m/s) | wind-driven ACH boost |
| `fan.power_w` | 250.0 | W | two 125W HAF circulation fans, typical commercial spec for a 100 m² house |
| `fan.rh_wet_threshold_min` | 90.0 | % | leaf-wet RH_in threshold at fan_frac=0 |
| `fan.rh_wet_threshold_max` | 96.0 | % | leaf-wet RH_in threshold at fan_frac=1 -- **modelling assumption, not measured**; see CLAUDE.md |
| `transpiration.lai` | 3.0 | m²/m² | mature tomato canopy |
| `transpiration.coeff_radiation` | 0.4 | - | Stanghellini-type radiation term |
| `transpiration.coeff_vpd` | 2.0e-5 | kg/s/m²/kPa per LAI | Stanghellini-type VPD term |
| `soil.fc / wp` | 0.30 / 0.13 | m³/m³ | FAO-56 typical loam |
| `soil.zr_m` | 0.4 | m | FAO-56 Table 22 (young/managed crop) |
| `soil.p` | 0.40 | - | FAO-56 Table 22 tomato depletion fraction |
| `crop.kc_*` | 0.6/0.9/1.15/0.80 | - | FAO-56 Table 12 tomato |
| `disease.spray_threshold_dsv` | 18 | - | BLITECAST action threshold |
| `simulation.dt_seconds` | 300 | s | sub-step |

Full derivations and comments live in `config.yaml` itself.

## How to run

```bash
pip install -e .[dev]                 # numpy, pandas, requests, matplotlib, pytest
python data/fetch_weather.py          # fetch + cache Guwahati weather, 2021 through the latest complete day, one CSV per year
python -m pytest tests/ -v            # full test suite (gates 1-8)
python scripts/validate.py            # gate table + figures/*.png
python scripts/diagnose_vent_authority.py  # quantifies vent-only VPD authority during monsoon (see CLAUDE.md)
python scripts/sweep_mpc_weights.py   # coarse grid sweep -> results/weight_sweep.csv, results/weight_sweep_pareto.csv
python scripts/experiment.py          # multi-year Regime A/B controller comparison -> results/regime_A_summary.csv, results/regime_B_summary.csv
```

### Interactive dashboard

```bash
pip install -e .[dashboard]                     # adds streamlit + plotly + pyarrow
python scripts/precompute_dashboard_data.py     # writes results/precomputed/*.parquet (~5-10 min, one-time)
streamlit run dashboard/app.py
```

Three pages: **Live Simulation** (inspect one controller over a regime/year
window), **Controller Comparison** (fixed vs threshold vs predictive/MPC over
a chosen regime and year, including a "failure moment" view of Threshold
irrigating right before rain -- shown for Regime B/dry season, the only
regime where that comparison is real; see CLAUDE.md), and **Live Twin** (an
animated polyhouse cross-section scrubbed through a run). See
`dashboard/app.py`.

**Precomputed by default.** The Predictive (MPC) controller takes 2-4
minutes per 90-day window, too slow to run on every page load -- especially
on a free-tier server. By default every page reads
`results/precomputed/{fixed,threshold,mpc}.parquet` (written once by
`scripts/precompute_dashboard_data.py`, ~2MB total, committed to the repo)
instead of simulating. Live Simulation has an explicit **"Run live
simulation (slow)"** toggle (off by default, warns about runtime) for
running the physics on demand over an arbitrary date range.

#### Deploying to Streamlit Community Cloud

1. Run `python data/fetch_weather.py` and `python scripts/precompute_dashboard_data.py`
   locally at least once, and commit their outputs (`data/guwahati_*.csv`,
   `results/precomputed/*.parquet`) -- the deployed container has no way to
   fetch weather or run MPC itself, and reads these files only.
2. Push to a GitHub repo. Streamlit Cloud reads `requirements.txt` (not
   `pyproject.toml` extras) and `runtime.txt` from the repo root, and
   `.streamlit/config.toml` for theme/server settings -- all already present.
3. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   the repo with main file path `dashboard/app.py`.
4. Never commit `venv/` (469MB locally; `.gitignore` covers it -- verified
   with `git check-ignore -v venv/`). Streamlit Cloud builds its own
   environment from `requirements.txt`.

## Current gate results

| Gate | What it protects against | Result |
|---|---|---|
| 1 (weather ingestion) | Wrong location/parsing, silently corrupted monsoon signature | **PASS** -- 4416/4416 rows, JJA mean RH 85.96% (band 78-95), night I_solar=0, June daily max 531-899 W/m2 |
| 2 (psychrometrics) | Wrong Tetens/ideal-gas constants | **PASS** -- all 4 checks |
| 3 (soil bucket) | Water created/destroyed, D out of [0,TAW] | **PASS** -- TAW=68.0mm, balance closes to 1e-6 over 1000 random steps |
| 4 (reference ET0) | Wrong radiation/wind coupling in Penman-Monteith | **PASS** -- clear-day ET0=6.42 mm/day, night ET0 ~0, monotonic in I_solar |
| 6a (DSV table) | Wrong Wallin lookup logic | **PASS** -- reference values match exactly |
| 6b (90-day closed-vs-open DSV) | Humidity/temperature-disease coupling | **XFAIL (documented)** -- see "Known limitations" in CLAUDE.md: the Wallin table's temperate-climate favourable band (7.2-26.6°C) is incompatible with Guwahati's monsoon ambient (JJA mean 28.7°C) once a sealed polyhouse's air saturates 24/7, a provable (not a bug) result of the specified physics |
| V1 (daytime greenhouse effect) | Broken energy balance | **PASS** -- 4.51°C (band 4-16) |
| V2 (pre-dawn humidity) | Broken vapour balance / the core monsoon signature | **PASS** -- 97.93% (>92%) |
| V3 (peak hourly ET) | Wrong transpiration magnitude | **PASS** -- 0.610 mm/hr (band 0.25-0.70) |
| V4 (daily total ET) | Wrong transpiration magnitude | **PASS** -- 3.05-4.47 mm/day (band 2.5-6.0) |
| V5 (T_in plausibility) | Runaway/negative temperature | **PASS** -- [25.07, 40.52]°C (band 5-55) |
| V6 (night leaf-wetness) | Missing the Assam disease signature | **PASS** -- 1.00 (>0.5) |
| V7 (monotonic vent response) | Non-physical ventilation coupling | **PASS** -- T_in and RH_in strictly decreasing across 5 vent levels |
| V8 (no-solar steady state) | Energy balance not conserving / not relaxing correctly | **PASS** -- \|T_in-T_out\| = 0.332°C (<0.5) |

Performance: a 90-day hourly run (2160 hours × 12 sub-steps) completes in
~3.3 s, well under the 60 s budget.

See `CLAUDE.md` for architecture decisions, the full Gate 6b diagnosis, and
known limitations.
