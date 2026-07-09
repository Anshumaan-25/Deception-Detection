# ST-GAE evaluation — SubjectA (2026-07-09): v1 FALSIFIED by the pre-registered bars

The reconstruction-based ST-GAE (Documentation/ST_GAE_DESIGN.md) was implemented in full,
fit on the A/V-synced baseline clip, and evaluated against the four pre-registered success bars
on the ELAN-annotated interviews. **It fails — decisively and informatively.** Per the design's
own §6 Bar-4 falsification clause, the reconstruction ST-GAE is discarded and the marginal
z-score attribution path (which works: within-06 AUCs 0.68–0.70) remains the method.

This is the methodology functioning as intended: Bar 4 (Holdout Truth Stability) was added
specifically to catch small-baseline overfitting, and it did.

## The fit is healthy — the *attribution* is not
- Fit on 206 baseline windows (164 train / 42 val), 23 107 params, val recon 3.37 vs
  predict-baseline 12.33 → **ratio 0.274 (HEALTHY)**, best epoch 364. The model learns and
  generalizes the baseline clip's structure well.
- But applied to the interviews, reconstruction error explodes on **all** of them, truthful
  or deceptive:

| clip | ELAN content | global recon-z median |
|---|---|---|
| baseline (00) | Neutral | ≈ 0 (by construction) |
| clip 3 | Lie+Truth | 7.9 |
| clip 5 | Lie×2 | 9.3 |
| clip 7 | **Lie×1** | **7.2** (nearly lowest) |
| clip 6 | Truth→Lie | 34.1 |
| clip 2 | Lie×2 | 36.1 |
| clip 4 | **Truth×2** | **39.8** (highest) |

The most truthful clip (04) is the *most* "anomalous"; the all-lie clip (07) is nearly the
*least*. The reconstruction error tracks **clip identity / distance from the neutral baseline**,
not deception.

## Bar-by-bar verdict
- **Bar 1 (Sensitivity ≥0.69):** best within-06 node recon-z AUC = **0.605 (voice)** → **FAIL.**
  Every other node sits at/below chance (au_mouth 0.35, hand_left 0.17, gaze 0.36). The
  reconstruction signal does not carry the truth/lie contrast even within a single clip (where
  rank-AUC cancels the domain offset).
- **Bar 2 (Interpretability):** top nodes `[voice, head_pose, au_upper]` — does **not** recover
  the known structure (au_mouth/hands should lead). → **FAIL.**
- **Bar 3 (Relational):** congruence recon-z 0.437; gaze recon-z 0.361 — the freeze pattern is
  not surfaced as anomaly. → **FAIL.**
- **Bar 4 (Holdout Truth Stability):** global recon-z median baseline −0.7 / **truth 47.2** /
  lie 13.3; flag-rate (>p95 baseline) baseline 6% / **truth 81%** / lie 65%. Truth spikes harder
  than lie. → **FAIL — the exact small-baseline brittleness Bar 4 guards against.**

## Why it fails (root cause)
1. **Domain gap dominates.** A 105 s neutral baseline and the interview clips are globally
   different distributions. The AE, trained on baseline z (|z|≈1), is fragile out-of-domain:
   the interviews' modestly larger deviations (marginal median 11–20 vs baseline 9.85, ~1.5–2×)
   produce a *super-linear* reconstruction blow-up (~40× in recon-z). "Interview ≠ baseline"
   swamps "lie ≠ truth."
2. **Aggregation dilutes the signal.** Even within a clip (domain-invariant), node-level
   reconstruction error is dominated by the bulk, non-discriminative feature variance; the small
   discriminative part (e.g. AU12 velocity tremor) that the marginal |z| path isolates is buried.

Neither is a fixable hyperparameter — they are structural to *reconstruction error over an
aggregated graph fit on one short baseline*. The marginal z-score path avoids both by comparing
each feature to its own baseline and reading truth-vs-lie **within** the interview.

## Decision & forward path
- **Keep:** the marginal per-channel z-score attribution (validated: RESULTS_PRODUCTION.md,
  within-06 0.68–0.70, direction-aware 11-node table). It is the shippable instrument.
- **Discard (this version):** reconstruction-error ST-GAE as the deception signal.
- **VideoMAE:** stays deferred — its re-entry criterion (an AU blind-spot in a *working* ST-GAE)
  is not even reached.
- **Future (honest, not this session):** if the graph idea is revisited, the objective must
  change from reconstruction to something that isolates the discriminative signal —
  a **predictive** objective (predict a channel from its cross-modal neighbours, flag broken
  couplings), **per-feature** (not node-aggregated) residual targeting, or a **supervised**
  attribution head once N>1 subjects exist. The implemented package (graph_spec/dataset/model/
  fit/attribute, all tested) is a reusable substrate for that.

## Provenance
Fit + attribution: `pipeline_system_outputs/REC_SUBJECTA/stgae_fit/` +
`stgae_attribution.csv`. Data: `REC_SUBJECTA_SYNCED_*` (A/V-synced, canonicalizer 80 ms fix).
Code: `deception_detection/stgae/`; tests `tests/verify_stgae.py` (16 checks, green);
evaluation `validation/gt_subjectA/stgae_evaluate.py`.
