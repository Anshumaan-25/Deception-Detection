# Predictive Cross-Modal Coupling Model — design + pre-registered bars (v2 of the graph line)

**Status:** DESIGN LOCKED 2026-07-10, pre-registered before any real-data run.
**Predecessor:** the reconstruction ST-GAE (Documentation/ST_GAE_DESIGN.md) — implemented,
evaluated, and **falsified** by its own pre-registered Bar 4 on 2026-07-09 (§10 there;
full evidence in `deception_detection/validation/gt_subjectA/STGAE_RESULTS.md`).
This document exists because that process worked: bars are written down *first*.

## 1. Why a second graph attempt at all

The v1 post-mortem identified two **structural** failure modes of reconstruction error:

1. **Domain gap dominates.** Reconstruction error measures distance-from-baseline; every
   interview window is far from the neutral baseline, truthful or not. "Interview ≠ baseline"
   swamped "lie ≠ truth" (truth clip 04 most 'anomalous', all-lie clip 07 nearly least).
2. **Node aggregation dilutes.** Node-summed error is dominated by bulk non-discriminative
   variance; the validated needles (AU12 velocity dynamics, hand↔face) are buried.

The coupling model is not a tweak of v1 — it changes the **question**. Instead of
*"can I reconstruct this window?"* it asks, per node: *"hide this node entirely — can the
other 10 predict what it was doing?"* The per-node prediction residual then reads:
**"this channel stopped moving the way this person's other channels say it should"** — a
decoupling detector, aimed at the validated freeze+leakage signature (bipolar 11-node
table, RESULTS_PRODUCTION.md).

Why this attacks both failure modes rather than working around them:

- **Domain gap:** the interview shift is mostly *marginal* (everything more animated).
  Conditional relationships *between* channels are closer to physiology and should
  transfer far better; residuals are additionally scored **within-clip** (§5), which
  cancels any residual global shift by construction.
- **Dilution:** residuals are kept **per-feature first** and aggregated to node level
  only for display; the feature-level columns are first-class outputs.

## 2. Objective: masked-node prediction (leave-one-node-out imputation)

For a window tensor `X [T, N=11, D]` (baseline-z features, dataset.py unchanged):

- pick a target node `n`; **replace its encoder stream with a learned per-node mask
  token before any message passing**;
- run the spatio-temporal graph network; decode all nodes;
- loss = feature-count-normalized, validity-masked, confidence-weighted MSE **on the
  target node's features only** (`masked_node_error` reused, restricted to `n`), with
  frames where the target has zero valid features carrying zero weight.

**Leakage guarantee (the critical invariant).** With message passing, information flows
multi-hop (n→m→n in two layers), so down-weighting a self-edge is NOT safe. The only
safe construction is that the target's input never exists in the forward pass: the mask
token replaces `h_n` at the encoder output, so the entire network output is invariant to
`x_n` — at any depth, bit-for-bit. `verify_coupling.py` asserts this with `torch.equal`.

**No bottleneck.** v1's temporal/channel bottleneck existed solely to prevent the
autoencoder from copying its input. A masked node cannot be copied, so the bottleneck
(and its ~0.5 lossy-compression floor) is removed. 2 ST-blocks (2-hop coverage of the
25-edge prior graph), embed 16, ≈15k params.

## 3. What is reused from v1 unchanged

- `graph_spec.py` — the LOCKED 11-node spec, feature slices, mask policy
  (voice ⇐ is_audio_active, face ⇐ face_confidence), functional-prior adjacency.
- `dataset.py` — frame CSV → `[T,N,D]` baseline-z tensors + validity + confidence.
- The anti-overfit stack shape: temporal 80/20 split, early stop on val, denoising
  augmentation (noise + feature dropout on *visible* features), fixed seed, loud
  degenerate gate.
- The falsified v1 modules stay in place as the record; v2 lives alongside them
  (`coupling_model.py`, `coupling_fit.py`, `coupling_attribute.py`).

## 4. Fit protocol and the degenerate gate

- Training: each window in a batch gets one uniformly-sampled target node per step.
- Validation: deterministic rotation over **all 11 targets** per val window.
- Reference: **predict-zero** (ẑ=0 = the subject's baseline mean for the hidden node).
  `ratio = val_prediction_error / val_predict_zero_error` now has a direct reading:
  *do this person's channels carry information about each other at all?*
  - On structureless noise the neighbors carry nothing → ratio ≈ 1.0.
  - **Gate (pre-registered): HEALTHY iff ratio < 0.90.** At ≥ 0.90 the fit prints the
    loud degenerate warning and attributions are suspect.
- Per-node ratios are reported too: a node with baseline ratio ≈ 1 is individually
  unpredictable for this subject and its coupling-z should be read with suspicion.

## 5. Scoring: coupling-z + within-clip percentile, per-feature first

Attribution on the 2 s / 1 s grid (same as the marginal path and the ELAN scorer):

- **coupling error** per frame per node = target-node residual averaged over the
  overlapping 90-frame model windows covering that frame, **weighted by target
  validity** (a non-speaking voice frame contributes nothing, rather than a fake 0).
- **coupling-z** = (window error − baseline window error mean) / baseline std, per node
  AND per feature (61 feature columns emitted alongside the 11 node columns).
- **within-clip percentile** of coupling-z is computed at evaluation time; the truth/lie
  contrast is always read within-interview, where any residual domain shift cancels.

## 6. PRE-REGISTERED SUCCESS BARS (fixed before any real-data run)

Evaluated by `validation/gt_subjectA/coupling_evaluate.py` on REC_SUBJECTA_SYNCED,
fit on baseline clip 000 only, ELAN labels for scoring only. Bars 1–4 mirror
ST_GAE_DESIGN §6 for comparability.

- **Bar 0 — capability gate (before real data):** the synthetic suite must be green:
  no-self-leakage (bitwise), coupling recovery (a planted A→B coupling is learned;
  breaking it spikes B's residual and only B's), domain-shift robustness (scaling all
  marginals with couplings intact must NOT spike residuals — the exact v1 failure,
  simulated), overfit-on-noise → gate fires. Plus fit health on the real baseline:
  ratio < 0.90.
- **Bar 1 — Sensitivity:** best coupling channel (node- or feature-level) within-clip-06
  AUC ≥ **0.69** — it must at least match the marginal z path to justify existing.
- **Bar 2 — Interpretability:** top-3 nodes by within-06 AUC must include `au_mouth` or
  a hand node (the channels ground truth says carry signal).
- **Bar 3 — Relational:** the freeze pattern must surface as a coupling anomaly:
  `gaze` coupling-z within-06 AUC ≥ **0.55** (marginal path shows gaze *inverse* ≈ 0.16
  = freeze; a direction-agnostic decoupling detector must flag it above chance).
- **Bar 4 — Holdout Truth Stability (the go/no-go):** identical formula to v1:
  |truth_median − baseline_median| ≤ max(1.0, 0.5·|lie_median − baseline_median|) AND
  truth flag-rate (> p95 of baseline windows) ≤ baseline flag-rate + 0.15.
  v1 failed this at truth=81% vs baseline=6%.

**Decision rule.** Pass all → the coupling channels ship as a *complement* to the
marginal z table (never a replacement; attribution doctrine unchanged). Fail Bar 1 or
Bar 4 → v2 is discarded like v1, the marginal per-channel z-score attribution remains
the shipped instrument, and the graph line is **closed for the n=1 era** (revisit only
with N>1 subjects and a supervised head).

## 7. RTX 6000 Ada execution profile (the production box)

The model is tiny; the speed traps are transfer and Python overhead, not FLOPs:

- **Whole-dataset-resident-on-GPU fit:** all baseline window tensors (~10 MB) are moved
  to VRAM once; every epoch is pure GPU compute, zero per-step host↔device transfer.
- **Vectorized masking:** the 11 target-node passes are folded into the batch dimension
  (`repeat_interleave` ×11 + per-replica mask index) — one batched forward, not 11
  sequential ones. Equivalence with sequential passes is unit-tested.
- **TF32 matmuls enabled** on CUDA (Ada tensor cores; rank-based bars are immune to the
  tiny CPU↔GPU numeric differences). Full fp16 AMP deliberately NOT used — at 15k params
  kernel-launch overhead dominates, not arithmetic.
- Device auto-fallback (`cuda` if available else CPU) keeps the identical code path
  runnable for the synthetic CPU test suite on the dev laptop.

## 8. Deliverables

- `deception_detection/stgae/coupling_model.py` / `coupling_fit.py` / `coupling_attribute.py`
- `deception_detection/tests/verify_coupling.py` (synthetic, CPU, no real footage)
- `deception_detection/validation/gt_subjectA/coupling_evaluate.py` (desktop; prints the
  bar-by-bar verdict and writes `coupling_attribution.csv`)
- MASTER_REFERENCE §12/§14 + changelog sync.

## 9. Empirical outcome — FALSIFIED 2026-07-10 (Bar 4)

Run on the desktop against `REC_SUBJECTA_SYNCED_*` (full write-up:
`deception_detection/validation/gt_subjectA/COUPLING_RESULTS.md`). Verdict per the §6
decision rule: **v2 is discarded; the graph line is closed for the n=1 era.**

| Bar | Threshold | Result | |
|---|---|---|---|
| 0 fit health | ratio < 0.90 | 0.801 | PASS |
| 1 sensitivity | best within-06 AUC ≥ 0.69 | **0.754** (feature `postural_stillness`) | PASS |
| 2 interpretability | top-3 nodes include au_mouth/hand | `[voice, au_upper, head_pose]` | FAIL |
| 3 relational | gaze decoupling AUC ≥ 0.55 | 0.295 | FAIL |
| 4 holdout truth stability (go/no-go) | truth flag-rate ≤ baseline+15% | baseline 6% → **truth 93%**, lie 91% | **FAIL** |

The central bet — conditional couplings transfer across the baseline↔interview domain gap where
marginals don't — **holds per-feature within-clip (Bar 1 passes, beating the marginal path's
~0.70) but breaks in aggregate**: the node-summed global coupling-z is still a
distance-from-baseline meter (truth median 39.69 > lie 22.23; GLOBAL(sum) AUC 0.146, *inverted*,
same signature as scalar `deviation_magnitude`). Bar 4 reads the aggregate on purpose (it is the
production false-positive gate) and fails identically in character to v1 (truth 81% there).

**This is the SECOND pre-registered graph formulation falsified by the same small-baseline
brittleness** (v1 reconstruction, v2 predictive coupling). Two independent objectives, one failure
mode → the blocker is **N=1**, not the architecture. Per §6, revisit the graph line only with N>1
subjects and a supervised head; the coupling substrate (all tested) is reusable for it. The
marginal per-channel z-score attribution remains the shipped instrument.
