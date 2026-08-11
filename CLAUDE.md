# CLAUDE.md -- notes for future sessions

This file records architecture decisions, what each gate protects against,
known limitations, and next steps. Read this before touching `sim/`.

## Architecture decisions

**Config, not constants.** `config.yaml` holds every tunable physical
parameter; `sim/config.py` is a small hand-rolled loader for a YAML subset
(nested `key: value`, comments, scalars) because the project's dependency
list excludes PyYAML. If you need real YAML features (lists, anchors,
multi-line strings), you've outgrown the loader -- extend it deliberately,
don't silently widen scope.

**`area_cover_m2` is the one free geometric parameter.** The task spec
fixes `area_floor_m2`, `cover_tau`, `cover_U_Wm2K`, `ACH_min`,
`ACH_max_base`, `wind_coeff`, `C_eff_JK_per_m2`, and both transpiration
coefficients exactly. It never specifies the polyhouse's enclosing surface
area (walls+roof) or its height. Both are derived in `config.yaml` from an
explicit Quonset/tunnel-shape geometric assumption (semicircular arch,
`height_m=3.0` as the *mean* internal height) -- not tuned by trial and
error to pass gates, though the derivation was chosen carefully because
it's the only lever available to balance two competing constraints: enough
conduction *loss* to keep V1 (daytime greenhouse effect, needs `A_cover`
small enough for warming) and V8 (no-solar steady state) both satisfiable
at the same time. See the exhaustive search in git history / the
conversation that built this if you need to re-derive it after changing
`area_floor_m2`.

**Exponential (integrating-factor) sub-stepping, not plain forward Euler.**
Both the temperature and vapour balances are, once weather/vent_frac are
frozen for a 300 s sub-step, linear relaxations `dy/dt = k*(y_eq - y)`.
They are integrated by solving that ODE *exactly* over the sub-step
(`y_eq + (y0-y_eq)*exp(-k*dt)`), not by a first-order Euler difference. This
was **not** a style preference -- plain forward Euler at `dt=300s` is
provably unstable for the vapour balance at high ventilation
(`dt * ACH/3600 > 2`, the standard forward-Euler stability limit for a
linear decay, is crossed whenever `ACH > 24/hr`, well within the normal
`vent_frac` range). Verified by running the naive difference: RH_in
oscillated between 0% and 100% every 5-minute sub-step under sustained high
ventilation. The exponential update is still fully explicit (single
closed-form evaluation, no iteration/matrix solve) and is the same
technique used by established greenhouse models (e.g. Vanthoor 2011
"GreenLight") for the same reason. See `sim/polyhouse.py` module docstring.

**Wind speed unit bug caught early.** Open-Meteo's archive API defaults
`wind_speed_10m` to km/h, not m/s. `data/fetch_weather.py` explicitly
requests `wind_speed_unit=ms`. Before this was caught, fetched "wind"
values were ~3.6x too high (max ~33.5 "m/s" = a cyclone, for routine hourly
Guwahati data -- the tell). This mattered a lot: wind feeds `ACH_max`
multiplicatively, so the bug was silently making the ventilation term
dominate the energy balance under ordinary conditions and suppressing the
daytime greenhouse-effect signal (V1). If you ever re-derive weather
fetching from scratch, sanity-check wind magnitude against physical
intuition (Guwahati mean wind is ~1.7 m/s) before trusting anything
downstream of it.

**Disease periods close at day boundaries, not just on RH drop.**
`DiseaseModel.step(..., end_of_day=...)` forces a period close at the last
hour of each calendar day, in addition to closing on `leaf_wet=False`. The
literal Wallin/BLITECAST description just says "track continuous wetness
periods," but taken completely literally that means an unbroken multi-week
saturated stretch (which happens under near-zero ventilation, see below)
would count as a *single* period -- and since the table saturates at DSV=4
for any duration past ~19-28 hours, an arbitrarily long streak would
contribute exactly one DSV=4 total, silently hiding weeks of sustained risk
instead of accumulating it. Daily boundaries match how BLITECAST is
actually run in practice (Krause et al. 1975 evaluate per calendar day).

## What each gate protects against

- **Gate 1 (weather)**: wrong lat/lon, unit confusion, or a parsing bug
  that would silently destroy the monsoon-humidity signature everything
  else depends on.
- **Gate 2 (psychrometrics)**: wrong Tetens constants or unit conversions
  in the humidity math that every other module calls.
- **Gate 3 (soil)**: water being created or destroyed by the bucket model,
  or depletion escaping its physical [0, TAW] bounds.
- **Gate 4 (ET0)**: a broken FAO-56 Penman-Monteith term (radiation,
  aerodynamic, or the day/night soil-heat-flux switch) producing physically
  implausible reference ET.
- **Gate 6a (DSV table)**: transcription errors in the Wallin lookup table.
- **Gate 6b (documented XFAIL)**: see "Known limitations" below.
- **V1**: the energy balance actually produces a greenhouse effect (not
  just numerically stable nonsense).
- **V2**: the vapour balance actually produces the pre-dawn saturation that
  IS the regional problem this whole project exists to study.
- **V3/V4**: transpiration magnitude is in a believable range (not an
  order-of-magnitude unit error).
- **V5**: no runaway/negative temperature under the validation policy.
- **V6**: the "Assam signature" -- sustained night leaf wetness -- actually
  shows up, since that's the mechanistic link to disease pressure.
- **V7**: ventilation actually cools and dries monotonically (a
  non-monotonic response would make any controller comparison built on top
  of this meaningless).
- **V8**: the coupled ODEs conserve energy correctly and relax to the
  right fixed point with no forcing -- a basic sanity check on the
  integration scheme independent of weather data.

## Known limitations

1. **Single-zone, lumped-parameter.** No spatial gradients (no distinction
   between near-vent and far-from-vent zones, no vertical stratification).
   A real polyhouse has meaningfully different microclimates near open
   vents vs. the center.
2. **No CO2 balance.** Photosynthesis/respiration CO2 dynamics are not
   modeled; only water and energy.
3. **No crop growth feedback.** LAI is fixed at 3.0 for the whole run; a
   real crop's LAI (and therefore transpiration/shading) changes through
   the season. Kc changes by calendar day (FAO-56 stages) but that's a
   scheduled table lookup, not a growth model.
4. **Explicit (exponential-Euler) integration.** First-order accurate.
   Adequate given the deliberately-stable sub-stepping (see above), but not
   a substitute for a proper adaptive/implicit solver if someone later
   wants tighter accuracy guarantees.
5. **Gate 6b: the Wallin/BLITECAST table is temperate-climate-calibrated
   and mismatches tropical Guwahati.** This is the significant finding from
   this build and is worth understanding in full before extending the
   disease model:

   - The Wallin (1962) table's three temperature bands top out at 26.6°C
     (its highest band, 15.1-26.6°C, is the one relevant to Guwahati).
     Above that, every duration maps to DSV=0 -- the table assumes late
     blight simply doesn't progress in heat, which is true for *Phytophthora
     infestans* in its original temperate (US Northeast potato belt)
     calibration context.
   - Guwahati JJA (monsoon) ambient T_out averages **28.7°C**, already
     above that ceiling on 89 of 92 days, before any polyhouse warming is
     added.
   - Given the specified transpiration formula (`0.4*I_in/lambda_vap` term,
     uncoupled from ambient RH) and `ACH_min=0.5/hr` (both fixed by spec),
     a polyhouse with `vent_frac=0` has its transpiration source overwhelm
     the minimal infiltration removal at any realistic floor area/volume:
     `chi` is driven to `chi_sat(T_in)` continuously, so RH_in=100% (and
     leaf_wet=True) essentially all day, every day, for the whole run --
     verified numerically, not assumed. This was checked across
     `area_cover_m2` from 203 up to an unrealistic 800 m² (8x the floor
     area) and the conclusion doesn't change.
   - Because the wetness period therefore spans essentially the entire
     day, each day's DSV is gated by *that day's mean* T_in, not just its
     coolest hours -- and daily mean T_in for a closed polyhouse is always
     >= T_out's own daily mean (28.7°C), so it almost never re-enters the
     table's favourable band. Result: **cumulative DSV over a 90-day
     closed-vent monsoon run is ~0**, not >40.
   - The **open**-vent run, counter-intuitively, scores a modest *nonzero*
     DSV (~12 over 90 days) specifically because ventilation restricts
     leaf-wetness to cooler overnight/pre-dawn hours (mean ~26.8°C, right
     at the table's ceiling) instead of including the hot midday. This is
     the mechanism by which more ventilation can produce *higher* measured
     DSV than a sealed structure under this exact table+climate
     combination -- the reverse of gate 6's original expectation.
   - This is tracked as `tests/test_engine_disease.py::test_gate6_closed_vents_produce_high_cumulative_dsv`,
     marked `xfail(strict=True)` with the full derivation in its docstring,
     per explicit user direction (2026-08-11): report as a documented
     finding rather than loosen the underlying physics or the >40
     threshold. Do not "fix" this by changing the Wallin table's
     thresholds, the transpiration formula, or `ACH_min` -- all three are
     fixed by the project spec. If a future session wants a location-aware
     disease signal for tropical climates, the honest fix is a different
     (documented, sourced) disease model calibrated for warm/humid
     conditions -- e.g. a bacterial-wilt- or *Alternaria*-style
     temperature-humidity index -- run alongside Wallin rather than instead
     of it.

## Controllers (added 2026-08-11, mpc.py + experiment.py superseded later same day -- see below)

`controllers/` has three, all exposing `vent_policy`/`irrigation_policy`/
`fan_policy` bound methods matching `sim.engine.run`'s callable signature
(the fan actuator and its `fan_policy` were added later the same day; see
"Circulation fan actuator" below):

- **`fixed.py`**: timer baseline, no feedback -- fixed day/night vent
  window, two fixed-clock-time irrigation doses/day, fan always off.
- **`threshold.py`**: reactive bang-bang with vent hysteresis (separate
  open/close thresholds on T_in and RH_in to avoid chattering), rain-skip
  irrigation gated on `soil.stressed`, and fan on at full power above 92%
  RH_in.
- **`mpc.py`**: receding-horizon controller, **retuned** the same day it was
  first written (see "MPC objective retune" below for the full mechanism
  and weight-selection story) -- the description immediately below is the
  *original* version and is kept for history, not as current behavior. The
  original: rolled ~7 candidate 12-hour vent *sequences* forward through a
  scratch `Polyhouse` copy using the run's own future weather as a perfect
  short-horizon forecast, scored each on fixed-band VPD tracking + a
  leaf-wet-hours DSV proxy + a hard temperature-safety penalty (added after
  an early version let T_in spike to 62°C), applied only the winning
  candidate's first hour. The retuned version additionally searches a fan
  dimension jointly, clips the VPD target into the per-hour *achievable*
  range instead of a fixed band, and adds an explicit actuation-switching
  penalty. Both versions use a 1-hour internal rollout step instead of the
  usual 300s sub-step -- legitimate because the exponential integrator (see
  above) is unconditionally stable at any dt.

`scripts/experiment.py` originally ran all three once over a single 90-day
2025-06-01..2025-08-29 window. It was **restructured** the same day (see
"Diagnosed 2026-08-11" and the regime/multi-year sections below) to run two
separate climate regimes across every available year (2021-2025) instead,
reporting mean +/- sd rather than a single-year snapshot -- the paragraphs
immediately below describe that original single-year run and are kept for
history; see `results/regime_A_summary.csv` / `results/regime_B_summary.csv`
for the current numbers.

**Notable result from the original single-year run, not a bug**:
`threshold` and `mpc` showed **zero** irrigation over that 90 days (`fixed`
applied 720 L/m2). JJA 2025 Guwahati rainfall in the fetched data totals
~1397mm over the window (~15.5mm/day mean), far exceeding tomato ETc
(~3-6mm/day), so `soil.stressed` never fires for a rain-aware controller --
the naive fixed schedule is irrigating into fields that don't need it. This
irrigation-policy logic is unchanged by the later fan/MPC retune, so this
finding still holds in the current multi-year Regime A results.

Cumulative DSV in that original run ended low (10-11) and comparable across
all three controllers, consistent with the Gate 6b finding above: this
climate + the literal Wallin table rarely produces high DSV regardless of
ventilation strategy. That specific numeric comparison is superseded by the
current multi-year results (which also add the Alternaria model
specifically because of this Wallin-in-the-wrong-climate problem -- see
"Alternaria risk model" / README), but the underlying diagnosis (don't trust
raw Wallin DSV as the primary cross-controller signal in this climate) still
stands.

## Diagnosed 2026-08-11: ventilation has almost no authority over VPD at night in monsoon

The original `results/comparison_summary.csv` showed Fixed / Threshold /
Predictive landing on nearly identical numbers -- 13.70 / 13.29 / 13.80 %
hours-in-VPD-band, DSV 10 / 10 / 11 -- with the MPC actually *worse* on DSV
and on actuations (330 vs. 180). This looked like the controllers weren't
doing anything, and they largely weren't, for a physical reason rather than
a bug in any of them.

**Hypothesis**: during JJA monsoon, outdoor RH_out averages 85-95%. Opening
the vents doesn't exchange saturated indoor air for *dry* outdoor air, the
way it would in an arid climate -- it exchanges saturated indoor air for
*near-saturated* outdoor air. Ventilation still has real authority over
temperature (the greenhouse effect is driven by solar gain vs. conductive +
advective loss, both of which vents control directly -- see V1/V7), but very
little authority over vapour content, because the vapour driving force
`(chi_in - chi_out)` the vent term acts on is small almost by construction
once RH_out is already high.

**Quantified** (`scripts/diagnose_vent_authority.py`, 90-day monsoon window,
2025-06-01..2025-08-29, 2160 hours, mean JJA RH_out=85.90%): ran the engine
twice, vent_frac pinned at 0.0 and at 1.0, and computed the achievable VPD
range `|VPD(vent=1) - VPD(vent=0)|` at every hour --

| | value |
|---|---|
| Mean achievable VPD range | 0.467 kPa |
| Median achievable VPD range | 0.262 kPa |
| Fraction of all hours with range < 0.2 kPa | 44.1% |
| -- daytime (06-18h) only | 5.5% |
| -- night (18-06h) only | **82.7%** |

The day/night split is the important number. During the day, solar gain
gives vents real thermal (and therefore VPD) leverage: only 5.5% of daytime
hours are "stuck" below a 0.2 kPa achievable range. At night, with no solar
term, 82.7% of hours are stuck -- no vent setting, fully closed to fully
open, can move indoor VPD by more than 0.2 kPa. Night is exactly when
leaf-wetness accrues (V6, V2) and DSV/disease pressure is decided, so this
is the mechanistic explanation for why all three controllers converged:
**they were all fighting the same night-time ceiling, which ventilation
alone cannot lift.** No amount of retuning a vent-only controller's
thresholds or MPC weights would have changed that -- the actuator itself
lacked authority over the metric that matters most. This motivated adding a
second actuator (a circulation fan, see below) that acts on the leaf
boundary layer instead of bulk air exchange, and retuning the MPC objective
around what vents can actually still do (temperature safety, VPD band
*where achievable*) rather than penalizing it for a target it never had the
authority to reach.

## Circulation fan actuator (added 2026-08-11)

`sim/polyhouse.py`'s `Polyhouse.step` takes a third actuator, `fan_frac` in
[0, 1], modelling a horizontal-airflow (HAF) circulation fan (config: `fan.*`,
default 250 W = two 125 W fans for a 100 m2 house). Unlike the vent, the fan
does **not** exchange air with outside -- it thins the leaf boundary layer
(Stanghellini 1987; Monteith & Unsworth boundary-layer-resistance theory),
which suppresses condensation on the leaf at a given *bulk* RH_in. We don't
model the boundary layer as a state variable (would need a leaf-energy-balance
sub-model this project doesn't otherwise carry); instead we model its net
effect directly as a rise in the bulk-RH threshold used for `leaf_wet`:

```
rh_wet_threshold(fan_frac) = rh_wet_threshold_min + (rh_wet_threshold_max - rh_wet_threshold_min) * fan_frac
                            = 90% at fan_frac=0  ->  96% at fan_frac=1
```

**The 90->96 mapping is a modelling assumption, not a measured value.** No
boundary-layer transfer coefficient was fit to real HAF-fan data for this
project; the two endpoints were chosen so that fan_frac=0 exactly reproduces
the project's existing, already-validated 90% threshold (used everywhere
else, including V2/V6/Gate 6b), and fan_frac=1 was picked as a plausible
"meaningfully harder to condense on" ceiling, not derived from a cited
source. Treat any downstream result that leans on the fan's leaf-wet
suppression as *directionally* trustworthy (fans do measurably raise the
condensation-onset RH -- that mechanism is real and cited) but *not*
quantitatively precise at the specific 96% figure.

**Sensitivity, quantified**: reran the Threshold controller (fan on above
92% RH_in, same as everywhere else) over the 90-day monsoon window with
`rh_wet_threshold_max` at 93%, 96% (the shipped default), and 99%:

| rh_wet_threshold_max | leaf_wet_hours (90d) | Wallin DSV | Alternaria risk |
|---|---|---|---|
| 93% | 1249 | 4 | 7 |
| 96% (default) | 990 | 2 | 0 |
| 99% | 277 | 0 | 0 |

Leaf-wet hours swing by >4x (1249 -> 277) across this range, and the
Alternaria risk score swings from 7 units down to 0 -- **this assumption is
not a minor detail; it materially changes both disease-model outputs.** Any
claim of the form "the fan cuts leaf-wetness by X%" should be read as
conditional on the 96% figure, and revisited if a real HAF-fan condensation
study becomes available to calibrate it properly. The qualitative result
(fan-equipped controllers show measurably fewer leaf-wet hours than
vent-only ones) is robust across this whole tested range; the exact
magnitude is not.

The fan also adds a small sensible heat load (`fan_power_w * fan_frac`,
into the same energy balance as solar gain) and its energy use is tracked
as an explicit cost (`fan_kWh` in engine output), not treated as free --
see the MPC retune below, which prices it into the objective.

## MPC objective retune (added 2026-08-11)

The original MPC (see "Controllers" above) lost to Threshold on both DSV
(11 vs. 10) and vent actuations (330 vs. 180) in the original 90-day
comparison, for the structural reason diagnosed above: it was penalizing
VPD tracking against a fixed 0.8-1.2 kPa band during hours ventilation had
no authority to reach it, with no term discouraging chatter. `controllers/mpc.py`
was retuned along four lines (see its module docstring for the full
mechanism):

1. **Achievable-VPD clipping.** Two cheap boundary rollouts per decision
   hour (vent pinned at 0 and at 1, fan off) bound the achievable VPD range;
   the fixed 0.8-1.2 kPa target is clipped into that range before scoring,
   so a structurally-unreachable hour costs zero VPD penalty instead of a
   misleading gradient.
2. **Fan added as a real search dimension.** The candidate search is now a
   cross product of 7 vent trajectories x 3 fan trajectories (off / on /
   reactive-at-92%RH), each pair rolled out jointly, so the objective can
   actually trade vent and fan against each other rather than optimizing
   vent alone.
3. **Explicit actuation-switching penalty**, `w_switch * |action -
   previous_action|`, applied to both vent and fan, including the
   transition from the real plant's last-committed action into the
   candidate's first hour.
4. **Leaf-wet-hours weighted explicitly** (`leaf_wet_weight`, not folded in
   as an implicit DSV proxy) since that's the quantity the fan gives the
   controller real authority over, per the Problem-1 diagnosis. A small fan
   energy cost (`fan_energy_weight * fan_frac`) prices the fan's `fan_kWh`
   rather than treating it as free.

**Weight selection**: `scripts/sweep_mpc_weights.py` ran a 3x3x2x2 = 36-combo
coarse grid (`vpd_weight` in {0.5,1,2}, `leaf_wet_weight` in {0.04,0.08,0.16},
`w_switch` in {0.05,0.15}, `fan_energy_weight` in {0,0.02}) on one year of
Regime A (2025-06-01..2025-08-29), 10 combos in parallel. Raw results:
`results/weight_sweep.csv`; Pareto front (water / leaf-wet-hours /
actuations, all minimized): `results/weight_sweep_pareto.csv`.

Threshold's baseline on this same window (fan on above 92% RH_in, the same
rule everywhere): 990 leaf-wet hours, 424 total actuations (223 vent + 201
fan), 304 fan_kWh.

Only **2 of 36** combinations landed on the strict Pareto front, and both
have `fan_energy_weight=0`:

| vpd_weight | leaf_wet_weight | w_switch | fan_energy_weight | leaf_wet_hours | total_actuations | fan_kWh |
|---|---|---|---|---|---|---|
| 0.5 | 0.04 | 0.15 | 0.0 | 769 | 288 | 528.0 |
| 0.5 | 0.04 | 0.15 | 0.02 | 1227 | 199 | 12.0 |

Reading the full 36-row table (not just the strict Pareto pair) surfaces the
actual finding: **every `fan_energy_weight=0.02` combination either lost
badly on actuations (character of the table's middle rows: 440-970
actuations) or overcorrected the fan off hard enough that leaf-wet hours
climbed back above Threshold's 990** (the `leaf_wet_weight=0.04,
fan_energy_weight=0.02` rows land at 1170-1246 wet hours -- *worse* than
doing nothing). At this grid's scale, `0.02` was simply too large a fan
price relative to the `leaf_wet_weight` range tested (0.04-0.16): it makes
the fan "not worth it" almost everywhere, defeating the entire reason the
fan actuator was added (Problem 1's diagnosis: the fan is the *only*
lever with real authority over night-time leaf-wetness). A finer grid
between 0 and 0.02 might find a real middle ground; that's future work, not
done here (would cost another sweep run) -- see "What to do next".

**Chosen: `vpd_weight=2.0, leaf_wet_weight=0.04, w_switch=0.15,
fan_energy_weight=0.0`** (now `MPCController`'s class default, and
`scripts/experiment.py`'s `MPC_WEIGHTS`). This isn't the strict-Pareto row
above -- it's the best-on-fan_kWh member of the small cluster of combos that
share `(leaf_wet_weight=0.04, w_switch=0.15, fan_energy_weight=0.0)` and
vary only `vpd_weight`:

| vpd_weight | leaf_wet_hours | total_actuations | fan_kWh |
|---|---|---|---|
| 0.5 | 769 | 288 | 528.0 |
| 1.0 | 783 | 327 | 506.25 |
| **2.0 (chosen)** | **791** | **396** | **480.75** |

All three beat Threshold on both leaf-wet hours (769-791 vs. 990) and total
actuations (288-396 vs. 424) by a comfortable margin; `vpd_weight=2.0` gives
up only 22 leaf-wet hours (2.9%) relative to the `0.5` extreme in exchange
for the lowest fan_kWh of the three and the largest actuation margin below
Threshold. **Honestly disclosed cost**: fan_kWh (480.75) is still ~58%
*higher* than Threshold's own fan use (304) -- beating Threshold on
leaf-wetness and actuation count did not come for free, it cost more
electricity. `fan_kWh` is tracked and reported in every output table
specifically so this tradeoff is never hidden.

The same weights are used for both regimes and every year -- per explicit
direction, weights are not re-tuned per-regime; one controller configuration
has to handle both.

## Acceptance: final multi-year, two-regime comparison (2026-08-11)

`python scripts/experiment.py`, 5 years each regime (2021-2025 for Regime A;
Nov-2021..Jan-2022 through Nov-2025..Jan-2026 for Regime B -- 2026 excluded
from both, insufficient cached data past 2026-08-01), mean +/- sd across
years. Full tables: `results/regime_A_summary.csv`, `results/regime_B_summary.csv`
(and `_raw.csv` for the per-year numbers behind each mean); figure:
`figures/regime_comparison.png`.

### Regime A -- Monsoon (Jun 1 - Aug 29)

| metric | Fixed | Threshold | **Predictive (MPC)** |
|---|---|---|---|
| Water (L/m2) | 720.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Leaf-wet hours | 1453.0 ± 55.9 | 1037.6 ± 57.5 | **785.8 ± 52.3** |
| Alternaria risk (units) | 28.6 ± 10.3 | 1.2 ± 1.3 | **0.4 ± 0.5** |
| Wallin DSV | 23.4 ± 11.1 | 4.4 ± 2.1 | **2.0 ± 0.7** |
| % hours achievable-VPD-band | 12.31 ± 1.71 | 8.07 ± 1.00 | 1.82 ± 1.12 |
| Vent actuations | 180.0 ± 0.0 | 199.4 ± 13.9 | 299.4 ± 39.4 |
| Fan (kWh) | 0.00 ± 0.00 | 316.35 ± 12.87 | 506.85 ± 17.28 |

### Regime B -- Dry season (Nov 1 - Jan 29)

| metric | Fixed | Threshold | **Predictive (MPC)** |
|---|---|---|---|
| Water (L/m2) | 720.0 ± 0.0 | 121.0 ± 11.4 | 121.0 ± 11.4 |
| Leaf-wet hours | 1337.8 ± 31.5 | 1110.6 ± 48.8 | **595.6 ± 88.1** |
| Alternaria risk (units) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Wallin DSV | 31.6 ± 13.4 | 0.4 ± 0.5 | **0.0 ± 0.0** |
| % hours achievable-VPD-band | 12.19 ± 1.14 | 3.17 ± 0.57 | 1.61 ± 0.56 |
| Vent actuations | 180.0 ± 0.0 | 585.6 ± 53.8 | **132.0 ± 39.5** |
| Fan (kWh) | 0.00 ± 0.00 | 323.5 ± 14.05 | 535.7 ± 2.74 |

(Bold = Predictive's best result among the three on that metric, where it
actually wins.)

### Stated plainly: where Predictive wins, where it loses

**Wins, clearly and consistently:**
- **Leaf-wet hours, both regimes.** 785.8 vs. Threshold's 1037.6 in Regime A
  (24% fewer) and 595.6 vs. 1110.6 in Regime B (46% fewer). This is the
  metric the whole fan-actuator-plus-retune exercise was aimed at (Problem
  1's diagnosis: vent-only control has almost no night-time VPD authority,
  so leaf-wetness needed a second actuator to move at all) and it is a real,
  multi-year-confirmed win, not a single-year fluke.
- **Wallin DSV and Alternaria risk, both regimes** (directly downstream of
  leaf-wet hours). Marginal in absolute terms in Regime A (Alternaria 0.4 vs
  1.2; both climates keep raw Wallin low per the Gate 6b finding), zero-zero
  in Regime B for Alternaria (too cool for its 24-29 degC band most of the
  dry season) but Predictive still edges out Wallin DSV there too.
- **Vent actuations, Regime B only.** 132.0 vs. Threshold's 585.6 -- a large
  win, because Threshold's hysteresis deadband chatters heavily in the dry
  season's sharper day/night VPD swings, and the actuation-switching penalty
  suppresses that for Predictive.

**Losses, stated directly, not buried:**
- **Vent actuations, Regime A -- Predictive LOSES to Threshold here**: 299.4
  vs. 199.4, i.e. Predictive uses *50% more* vent actuations than Threshold
  in the monsoon regime, the opposite of Regime B. The weight sweep (see
  above) selected weights that beat Threshold on *total* (vent+fan)
  actuations in its single-year check (396 vs. 424), by trading vent
  chatter for fan stability -- but the acceptance table asks for vent
  actuations specifically, and on that specific metric alone, Predictive is
  worse than Threshold in Regime A. Reporting the total-actuations framing
  without this caveat would have been presenting a favourable subset;
  stating it here instead.
- **% hours in the achievable VPD band, both regimes -- Predictive has the
  LOWEST of all three controllers**, not the highest: 1.82% (Regime A) and
  1.61% (Regime B), below both Threshold (8.07% / 3.17%) and even Fixed
  (12.31% / 12.19%, the best of the three on this metric in both regimes).
  This is a real loss, not a metric artifact -- Predictive's objective
  explicitly deprioritizes VPD tracking relative to leaf-wet hours (see
  weight selection above: `leaf_wet_weight` dominates the objective, VPD is
  only a soft, achievability-clipped penalty), so it is *working as
  designed*, but "working as designed" is not the same as "winning," and the
  table should say so plainly.
- **Fan energy, both regimes -- Predictive costs more than Threshold**:
  506.85 vs. 316.35 kWh (Regime A, +60%) and 535.7 vs. 323.5 kWh (Regime B,
  +66%). Beating Threshold on leaf-wetness and (in Regime B) vent actuations
  did not come for free.

**Tied, and why (a structural finding, not a coincidence):** Water use is
*identical* between Threshold and Predictive in Regime B (121.0 +/- 11.4 for
both, to the decimal) and in Regime A (0.0 for both). This is not a
near-tie -- check `sim/engine.py` and `sim/et0.py`: `ETc` (and therefore
`soil.step`'s water balance) is computed from **outdoor** weather
(`T_out`/`RH_out`/`wind`/`I_solar`/`cloud` via `et0_hourly`), never from
indoor state. Irrigation demand in this model is structurally decoupled
from the vent/fan controller entirely -- both Threshold's and Predictive's
irrigation policies reduce to the same soil-depletion-plus-rain-skip rule
acting on the same outdoor-driven soil trajectory, so no vent/fan strategy
can differentiate water use from another rain-aware controller. **Per the
acceptance question "does it beat Threshold on water in Regime B" -- no, it
does not beat it, they are tied, and the reason is a genuine model
structure (irrigation demand has no dependency on the climate controller),
not a tuning failure.** A future controller that wanted to win on water
specifically would need either a different irrigation policy (not part of
this retune's scope) or a model where indoor climate feeds back into crop
water demand (it currently doesn't -- ETc uses ambient weather only, a
known simplification, not something this session changed).

## What to do next

Controllers exist and are compared; plausible next steps:

- A location-appropriate disease signal (see Gate 6b note above) so
  disease-pressure comparisons between controllers carry more signal.
- Sensitivity analysis on MPC's cost weights (`dsv_weight`, `t_safety_c`)
  and candidate library -- currently hand-picked, not tuned.
- A real weather-forecast source for MPC instead of perfect foresight over
  the historical record, if this ever needs to run prospectively rather
  than as a comparison study.

- Vent and irrigation policies are already injected as plain callables
  `(state, soil, weather_row, timestamp) -> float` into `sim/engine.run()`
  -- no engine changes should be needed to plug in a rule-based, PID, or
  MPC controller.
- A natural first controller-comparison experiment: rule-based
  humidity-triggered venting (open when RH_in crosses some threshold) vs.
  the passive `vent=0.3` baseline used for validation, scored on cumulative
  DSV (acknowledging the Gate 6b caveat above -- a fairer score than raw
  DSV magnitude for this climate is probably cumulative *night* leaf-wet
  hours, or time spent with VPD outside the 0.8-1.2 kPa optimal band shown
  in `figures/validation.png`) and on total ET/irrigation demand.
- If you add controllers that make decisions off `state["day_index"]` or
  other engine-provided state, check `sim/engine.py`'s `state` dict
  construction -- it's deliberately built from the *previous* hour's
  observed output so a policy can't see the future.
- Before trusting any controller comparison, re-run
  `python scripts/validate.py` -- it's cheap (~1s for the 7-day window)
  and will catch it immediately if a controller change to `sim/` broke one
  of the physical-realism gates.
