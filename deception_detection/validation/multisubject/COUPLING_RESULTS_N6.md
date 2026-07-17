# Coupling model — N=6 per-subject 4-bar re-eval (2026-07-17)

**Headline: the predictive cross-modal coupling model is FALSIFIED by its pre-registered go/no-go
gate (Bar 4, Holdout Truth Stability) on every subject where it can be tested.** Extending the
earlier falsifications (SubjectA ×2, SubjectB ×1), the four new subjects give **Bar 4 FAIL on C, D,
and E**; SubjectF is inconclusive (no interview clip has ≥20 pure Lie *and* Truth windows, so the
within-clip FOCUS bars can't run). Net: **Bar 4 fails on all 5 evaluable subjects (A, B, C, D, E).**
The graph line stays closed — now confirmed across the N=6 corpus, not just N=1.

## Method
`validation/multisubject/coupling_evaluate_generic.py` — manifest-driven mirror of the SubjectA/B
scripts, IDENTICAL pre-registered thresholds. Per subject: fit the coupling model on the baseline
clip ONLY (labels never touch the fit), attribute every interview clip, auto-select the richest
bilabel FOCUS clip (≥20 pure Lie AND Truth), run Bars 0–4. Bar 4 is the go/no-go gate: held-out
TRUTH windows must not spike coupling-z vs baseline (truth median shift ≤ max(1, ½·lie shift) AND
truth 95th-pct flag-rate ≤ baseline + 15 pp).

## Results (C/D/E/F this session; A/B from prior records)

| subject | fit ratio (Bar 0) | FOCUS | best chan (Bar 1) | gaze (Bar 3) | **Bar 4** | base FP | truth FP | lie FP |
|---|---|---|---|---|---|---|---|---|
| A (prior) | 0.274 ✓ | — | 0.75 ✓ | ✓ | **FAIL** | 6% | 81–93% | — |
| B (prior) | ✓ | 1 | 0.754 ✓ | ✓ | **FAIL** | 6% | 93% | — |
| **C** | 0.842 ✓ | 4 | 0.797 ✓ | 0.457 ✗ | **FAIL** | 5% | **46%** | 32% |
| **D** | 0.756 ✓ | 7 | 0.918 ✓ | 0.547 ✗ | **FAIL** | 5% | 6% | 6% |
| **E** | 0.778 ✓ | 2 | 0.832 ✓ | 0.560 ✓ | **FAIL** | 5% | **77%** | 92% |
| **F** | 0.627 ✓ | — | — | — | **N/A** | — | — | — |

(FP = fraction of windows above the baseline 95th-percentile coupling-z threshold.)

## Why each subject fails Bar 4 (same failure mode, three flavours)
- **SubjectC** — truth spikes: truthful windows flag at **46%** vs 5% baseline (truth median 5.87 vs
  base −1.07). Coupling-z is a distance-from-baseline meter; interview-truth ≠ neutral baseline.
- **SubjectD** — the *quiet* failure: flag-rates look stable (truth 6% ≈ base 5%), but the median
  condition fails — truth median shifts **+2.71** from baseline vs the allowed **1.77** (= ½·lie
  shift of 3.53). Truth drifts from baseline nearly as far as lie does. No truth-vs-lie separation.
- **SubjectE** — the *instructive* failure: it **PASSES Bars 1, 2, AND 3** (best channel 0.832,
  interpretable top node `hand_left`, gaze decoupled 0.560) — the model looks excellent on every
  signal metric — and Bar 4 still kills it: truthful windows flag at **77%** (base 5%), lie 92%.
  Exactly the SubjectA pattern: the signal bars are fooled, the go/no-go gate is not.

## Interpretation
Across six subjects the coupling model's aggregate coupling-z never becomes a *deception* meter — it
stays a **distance-from-baseline** meter. Truthful interview windows are as far from the neutral
baseline as deceptive ones, so the pre-registered false-positive gate rejects it every time it can be
run. This is the same conclusion the marginal scorecard and the LOSO head reached from the other
direction: **there is no transferable deception construct here beyond per-subject, per-channel
attribution.** The graph line (ST-GAE recon v1 + coupling v2) is closed at N=6 as it was at N=1.

## Caveat / scope
SubjectF is genuinely under-powered for the within-clip coupling test (no interview clip reaches the
≥20-per-class floor); its fit is healthy (ratio 0.627) but the FOCUS bars are undefined — recorded as
N/A, not PASS/FAIL. The Bar 4 thresholds are exactly those pre-registered for SubjectA/B (no tuning).

## Provenance
`coupling_evaluate_generic.py` (this dir); per-subject `pipeline_system_outputs/REC_SUBJECT{C,D,E}/
coupling_attribution.csv` + `coupling_fit/`; run log `pipeline_system_outputs/coupling_N6.log`.
Design + bars: `Documentation/COUPLING_MODEL_DESIGN.md`; N=1 evidence
`validation/gt_subjectA/COUPLING_RESULTS.md`.
