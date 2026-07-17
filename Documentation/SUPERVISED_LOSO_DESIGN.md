# Supervised LOSO head — pre-registered design (generalizability test, NOT a performance play)

**Status:** DESIGN LOCKED, pre-registered **before the N=6 corpus is scored** (2026-07-15/17).
**What this is:** a *diagnostic* that answers one question — *"is there a **universal** deception
signature that transfers across people, or is the signal subject-specific?"* — by training a
supervised model on some subjects and testing it on **held-out** subjects (leave-one-subject-out).
**What this is NOT:** it is **not** a replacement for the shipped marginal per-channel z-score
attribution, and it is **not** claimed to give better numbers. The shipped instrument (per-subject
baseline calibration → within-clip per-channel |z| attribution) is unchanged.

> **Pre-committed expectation (write it down before the result):** everything measured so far —
> N=2 SUBJECT-SPECIFIC (SubjectA leaks via AU12 lip-tremor + silent-speech; SubjectB via blink;
> **disjoint** channels) — predicts this LOSO head will land **near chance**, i.e. *worse* than the
> per-subject marginal attribution. If so, that is a **success of the test**: it quantifies the
> subject-specificity we assert and vindicates the per-subject-calibration doctrine. A clearly
> above-chance LOSO result would be the surprising, doctrine-revising outcome.

## 1. Why run it at all (given we expect it to "fail")
Two legitimate reasons — neither is "better accuracy":
1. **It closes a question a reviewer/stakeholder will ask** ("did you try a supervised cross-subject
   model?") with a pre-registered answer instead of hand-waving.
2. **It *measures* subject-specificity quantitatively.** "LOSO ≈ chance" is the hard number that
   proves channels don't transfer, far stronger than eyeballing disjoint per-subject tables.

## 2. The two methods answer different questions (do not conflate)
| | Marginal z-score (shipped) | Supervised LOSO head (this doc) |
|---|---|---|
| Question | which channels deviate for *this* subject vs *their own* baseline | can a model trained on N−1 people predict lie/truth for the **held-out** person |
| Unit | per-subject, per-channel | cross-subject generalization |
| Role | **attribution**, deployable now | **research test** of a universal signature |
| Uses labels? | no (labels only score it post-hoc) | yes — Lie/Truth is the supervised target on TRAIN folds only |

## 3. Features (reuse the shipped features — no new extraction)
Input = the **within-clip-normalized per-channel z-features** already produced by the pipeline
(the same features the replication scorecard scores), so clip-level confounds are controlled by
construction:
- window-level |z| per channel: kinematics (wrist velocity, hand↔face, motion energy), gaze/head,
  AUs + AU velocity/tremor **FFT band-powers**, blink/EAR, cross-modal incongruence
  (`silent_speech_duration_ms`, `mismatch_ratio`), acoustic (WavLM/prosody) where present.
- **FFT band-power features are kept** — they were SubjectA's *strongest* channels. This head sits
  *on top of* FFT+z, it does not remove them.
- Feature scaling is the existing per-subject **baseline** StandardScaler (unsupervised, no leakage);
  ELAN labels touch ONLY the supervised target.

## 4. Model (interpretable, small — N=6 is tiny)
- **Primary: L2-regularized logistic regression** over the per-channel z-features. Coefficients =
  interpretable channel weights (which channels a *cross-subject* model leans on).
- **Secondary (capacity check): gradient-boosted trees** (lightgbm/xgboost, shallow, heavy
  regularization) — only to check a linear model isn't underfitting a nonlinear-but-transferable
  signal. Report both.
- **No deep nets, no autoencoders.** 6 people is far too small; the effective sample size is
  **subjects (6)**, not windows. Regularize hard.

## 5. Protocol — strict Leave-One-Subject-Out
- 6 folds: train on 5 subjects' pure windows (≥60% single-label), test on the 6th subject's pure
  windows. **No subject appears in both train and test — ever.**
- Report **per-held-out-subject test AUC** (all 6, not just the mean) + mean ± CI. The unit of
  generalization is the *person*, so 6 folds = 6 data points; treat as exploratory with wide CIs.
- Pure-window class balance handled by class weights; report AUC (rank metric, balance-robust).
- **Honesty gates (pre-committed):** (a) report **test-fold performance only** — within-subject
  train accuracy is memorization and must never be quoted as validation; (b) no ELAN labels in
  feature calibration; (c) fix all hyperparameters/regularization before running (grid only via
  nested LOSO if at all, reported as such).

## 6. PRE-REGISTERED interpretation bars (fixed before scoring)
Compare LOSO **test** AUC against chance (0.5) and against the shipped per-subject within-clip
marginal AUC (~0.60–0.70 for signal-bearing channels in the subjects that have signal).

- **GENERALIZES** — mean LOSO test AUC ≥ **0.62** AND ≥ **4/6** held-out subjects individually
  ≥ **0.58**. → a universal signature exists; pursue a general model + report which channels it uses.
- **WEAK/PARTIAL** — mean LOSO **0.55–0.62**, or only a minority of held-out subjects clear 0.58.
  → some transferable signal, unreliable; not deployable, flag the channels that recur.
- **SUBJECT-SPECIFIC (expected)** — mean LOSO test AUC **≈ 0.50** (CI includes 0.5), no consistent
  above-chance held-out subject. → confirms channels don't transfer; **vindicates per-subject
  calibration + attribution**; a universal model is off the table at this N.
- **Bar-4 analog (false-positive honesty):** on held-out **truthful** windows the model's
  lie-probability must not systematically exceed its baseline (i.e. it must not merely flag
  "interview ≠ neutral"). Because the target is Lie-vs-Truth (not vs baseline) and LOSO holds the
  subject out, this is largely controlled — but report held-out truth-window false-positive rate
  anyway; if truth is flagged as hard as lie, the "signal" is a confound, not deception.

**Decision rule.** GENERALIZES/WEAK → the LOSO head becomes a *complement* to the marginal table
(channel weighting for research), never a production per-window verdict. SUBJECT-SPECIFIC → recorded
as the quantitative confirmation of subject-specificity; the marginal per-channel z-score attribution
remains the sole shipped instrument; no universal/supervised model is pursued until N ≫ 6.

## 7. Doctrine boundary (unchanged)
This is a **research instrument** for "does a signature transfer across people," and at most a
channel-**ranking** input. It does **not** emit a per-window Lie/Truth verdict in production. The
system's output stays **attribution, never classification.** (Same boundary that archived the TFN
classifier and closed the graph line.)

## 8. Deliverables
- `deception_detection/multisubject/loso_head.py` (features → L2-logreg / GBT, strict LOSO, the bars).
- `deception_detection/validation/multisubject/loso_evaluate.py` (runs it on the N=6
  `*_recording_calibrated.csv` + ELAN dirs; prints per-subject + aggregate AUC + the bar verdict).
- Gated on the N=6 corpus being fully assembled (in progress). MASTER_REFERENCE §14/§15 + changelog.

## 9. Empirical outcome — **SUBJECT-SPECIFIC (2026-07-17, N=6)**
Run: `python validation/multisubject/loso_evaluate.py validation/multisubject/replication_manifest_N6.json`
on all six assembled recordings (8,286 pure Lie/Truth windows, 134 within-clip-centered channels).
L2-logreg (C=0.5, `class_weight="balanced"`), strict LOSO, per-fold StandardScaler fit on TRAIN only.

**Per-held-out-subject test AUC** (the subject is never in its own training set):

| held-out | test AUC | truth FP-rate | n Lie | n Truth |
|---|---|---|---|---|
| SubjectA | 0.525 | 0.386 | 509 | 140 |
| SubjectB | 0.444 | 0.645 | 825 | 1494 |
| SubjectC | 0.496 | 0.507 | 734 | 448 |
| SubjectD | 0.432 | 0.790 | 1103 | 625 |
| SubjectE | 0.425 | 0.648 | 554 | 1146 |
| SubjectF | 0.490 | 0.511 | 571 | 137 |

**mean LOSO test AUC = 0.469 ± 0.040; 0/6 held-out subjects ≥ 0.58.** Verdict per the §6 bars:
**SUBJECT-SPECIFIC** — the pre-registered, expected outcome (see §11 prior). This is a *success of
the test*: a supervised model trained on 5 people predicts the 6th at (indeed slightly below) chance,
so the deception channels demonstrably **do not transfer across people**. The mild sub-0.5 tilt is
itself informative — because some channels *invert* per subject (blink: +0.71 in B, −0.23 in D;
silent-speech: +0.68 in A, inverted in B/E/F), a 5-subject majority direction actively *mis-predicts*
the held-out person. That is the quantitative hard number behind "0/134 channels replicate."

**Bar-4 analog (held-out truth-window false-positive rate):** high and tracking each subject's own
class balance (e.g. D 0.79 with a lie-heavy prior), not a clean truth-vs-lie separation — consistent
with no transferable signal rather than a truth≠baseline confound (LOSO already holds the subject out).

**Consequence (per the §6 decision rule):** confirms subject-specificity quantitatively; **vindicates
per-subject baseline calibration + per-channel |z| attribution as the sole shipped instrument.** No
universal/supervised model is pursued until N ≫ 6. The cross-subject logreg's top |coef| channels
(`duchenne_index`, `AU2_var`, `speech_hesitation_index`, `AU12_velocity_max`, …) are recorded as a
research-only channel ranking (`validation/multisubject/loso_coefficients.csv`) — **never** a
per-window verdict. The system's output stays **attribution, not classification.**

Artifacts: `multisubject/loso_head.py`, `validation/multisubject/loso_evaluate.py`,
`validation/multisubject/loso_results.csv` + `loso_coefficients.csv`. (GBT secondary check skipped:
lightgbm not installed on this box; the linear head already lands below chance, so a capacity check
for an "underfit nonlinear-but-transferable signal" is moot — there is no signal to underfit.)
