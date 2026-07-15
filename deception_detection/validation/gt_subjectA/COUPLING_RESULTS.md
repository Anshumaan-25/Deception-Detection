# Coupling model (graph-line v2) evaluation — SubjectA (2026-07-10): FALSIFIED by Bar 4

The predictive cross-modal coupling model (`Documentation/COUPLING_MODEL_DESIGN.md`) was fit on
the A/V-synced baseline clip and evaluated against its four pre-registered success bars on the
ELAN-annotated interviews (`coupling_evaluate.py`, run on the desktop against
`REC_SUBJECTA_SYNCED_*`). **It fails the go/no-go bar — the same Bar 4 that killed v1** — and per
its own §6 decision rule the graph line is **closed for the n=1 era**. The marginal per-channel
z-score attribution remains the shippable method.

This is the SECOND pre-registered graph formulation falsified by the same small-baseline
brittleness (v1 = reconstruction, `STGAE_RESULTS.md`; v2 = predictive coupling, here). Two
independent objectives, one failure mode → the obstacle is **N=1**, not the architecture.

## The fit is healthy and the idea has a real spark — but the go/no-go gate fails
- **Bar 0 (fit health): PASS.** 12 255 params, 206 baseline windows (164 train / 42 val),
  val prediction 0.904 vs predict-zero 1.128 → **ratio 0.801 (< 0.90 gate)**. The subject's
  channels genuinely predict each other: most-predictable-from-neighbours are `hand_left` (0.479),
  `body` (0.530), `hand_right` (0.585); least are `au_mid` (1.193), `au_upper` (1.023),
  `voice` (1.011), `head_pose` (1.010) — ratio ≥ 1 means neighbours carry ~nothing for that node.
- **Bar 1 (Sensitivity ≥ 0.69): PASS — and this is new vs v1.** Best coupling channel within-clip-06
  AUC = **0.754** (feature `postural_stillness`), *above* the marginal path's best (~0.70). v1
  flunked this bar at 0.605. So "body freezing relative to what the other channels predict" does
  carry within-clip Lie/Truth contrast.
- **Bar 2 (Interpretability): FAIL.** Top-3 nodes by within-06 AUC = `[voice, au_upper, head_pose]`
  — does not include `au_mouth` or a hand node (where ground truth says the signal lives).
- **Bar 3 (Relational, gaze decoupling ≥ 0.55): FAIL.** `gaze` coupling-z within-06 AUC = **0.295**
  (the freeze does not surface as a coupling anomaly; if anything gaze-decoupling is *higher* on
  truth here).
- **Bar 4 (Holdout Truth Stability — the go/no-go): FAIL.** Global coupling-z median baseline
  −0.63 / **truth 39.69** / lie 22.23; flag-rate (> p95 baseline) baseline 6% / **truth 93%** /
  lie 91%. Truth spikes *harder* than lie — the exact small-baseline brittleness Bar 4 guards
  against, identical in character to v1 (truth 81% there).

## Per-node and per-feature detail (within-clip-06, Lie > Truth)

Pure windows: pooled Lie 510 / Truth 138; within-06 Lie 190 / Truth 50.

| node | within-06 AUC | pooled (within-clip pct) |
|---|---|---|
| voice | 0.634 | 0.547 |
| au_upper | 0.553 | 0.516 |
| head_pose | 0.509 | 0.469 |
| congruence | 0.442 | 0.451 |
| au_mouth | 0.395 | 0.453 |
| blink | 0.376 | 0.435 |
| au_mid | 0.354 | 0.478 |
| gaze | 0.295 | 0.451 |
| body | 0.237 | 0.425 |
| hand_right | 0.178 | 0.411 |
| hand_left | 0.093 | 0.353 |
| GLOBAL (sum) | **0.146** | — |
| MAX-node | 0.152 | — |

Top per-feature (within-06): `postural_stillness` 0.754, `head_roll` 0.729, `head_pitch` 0.684,
`frame_acoustic_energy_rms` 0.651, `AU12` 0.622, `AU1` 0.620, `right_wrist_velocity` 0.614,
`AU1_velocity` 0.610, `frame_wavlm_latent_4` 0.598, `frame_wavlm_latent_3` 0.584.

## Why it fails (root cause) — the conditional-transfer bet holds locally, breaks in aggregate
The design's central claim was that *conditional* couplings transfer across the baseline↔interview
domain gap better than *marginal* deviations. The evidence splits:

1. **At the per-feature within-clip level, the bet partly holds** — Bar 1 passes, several coupling
   features (postural_stillness, head_roll/pitch tremor, AU12/AU1) separate Lie/Truth within clip
   06, where rank-AUC cancels the domain offset.
2. **At the aggregate level, the bet fails** — the node-summed global coupling-z is *still* a
   distance-from-baseline meter. Every interview window (truthful or not) is far from the neutral
   baseline, so the aggregate spikes on truth (median 39.69, flag 93%) exactly as v1's
   reconstruction error did. The node-level GLOBAL(sum) AUC is **0.146 — inverted**, the same sign
   inversion as the scalar `deviation_magnitude` (0.384): truth deviates *more* in aggregate for
   this subject (body/motion higher during truth), so any node-summed score inverts.

The go/no-go Bar 4 reads the aggregate on purpose (it is the false-positive gate). It fails. Per
the pre-registered rule, a within-clip Bar-1 pass does **not** rescue a Bar-4 failure — the
instrument would flag 93% of truthful windows in production, where there are no ELAN labels to
score within-clip.

## Decision & forward path (pre-registered — not moved after seeing the numbers)
- **Discard (this version):** the predictive coupling model as a shipped deception channel. Bar 4
  fails; Bars 2–3 fail; only Bar 1 passes.
- **Graph line CLOSED for the n=1 era.** Two pre-registered graph objectives, both falsified by the
  same small-baseline brittleness → the blocker is N=1, not the model. Revisit only with N>1
  subjects and a **supervised** attribution head (the honest v2+ direction; the coupling substrate
  — `coupling_{model,fit,attribute}.py`, all tested — is reusable for it).
- **Keep:** the marginal per-channel z-score attribution (RESULTS_PRODUCTION.md, within-06
  0.68–0.70, direction-aware 11-node table). Unchanged as the shippable instrument.
- **Note for N>1:** `postural_stillness` coupling (0.754) and the head-tremor coupling features are
  the coupling channels most worth re-checking *once multiple subjects exist* — if they replicate
  across subjects with a stable sign, a supervised head could legitimately weight them. At N=1 they
  are one subject's within-one-clip contrast and prove nothing on their own.

## Provenance
Fit + attribution: `pipeline_system_outputs/REC_SUBJECTA/coupling_fit/` +
`coupling_attribution.csv` (761 windows, file_index 2–7; clip 1 = B04C002 has no `.eaf`; baseline
clip 0 scored separately as the Bar-4 reference). Data: `REC_SUBJECTA_SYNCED_*` (A/V-synced,
canonicalizer 80 ms fix). Code: `stgae/coupling_*.py`; tests `tests/verify_coupling.py` (21 checks,
green); evaluation `validation/gt_subjectA/coupling_evaluate.py`. Fit is deterministic (fixed seed;
CuBLAS-nondeterminism warnings are benign for the rank-based bars).
