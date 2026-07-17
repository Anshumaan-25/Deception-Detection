# Project retrospective — the full journey, honest assessment, and paths to accuracy
*Written 2026-07-17, after the N=6 corpus closed. Companion to `MASTER_REFERENCE.md` (the living
technical doc) — this is the narrative: everything we tried, what each attempt taught us, what
actually works today, and the concrete options for pushing accuracy higher.*

---

## 0. TL;DR — is the approach "trash"? No. Read this first.
Two different claims got conflated. Only one of them failed.

- **The INSTRUMENT works.** The pipeline extracts real behavioural signal, and per-subject baseline
  calibration + within-clip per-channel |z| attribution *recovers* that signal where it exists. The
  positive controls pass: SubjectA leaks through silent-speech/AU12 (AUC 0.68), SubjectB through
  blink rate (0.71). The scalar "deviation magnitude" is meaningless (we proved that 2026-07-08),
  but the **per-channel** attribution is not.
- **The UNIVERSAL MODEL failed.** A single classifier / fixed channel-weighting that transfers
  *across people* does not exist at N=6. Three independent instruments agree (marginal scorecard,
  supervised LOSO, cross-modal coupling). That is a **pre-registered negative result**, not a broken
  system — and it matches the deception-science literature (there is no reliable universal "tell").

The signal is **real but subject-specific, and uneven**: A/B strong, C/D moderate, E/F weak. The
honest one-line summary is *"we built a working per-subject behavioural-anomaly instrument and
proved, rigorously, that the anomaly channels don't generalise across people."* Everything below
follows from that.

---

## 1. The mission and the non-negotiable constraints
**Goal:** from interview video, surface *where* a subject's behaviour departs from their own truthful
baseline, as forensic **attribution** — never an automated "lie" verdict.

Constraints that shaped every decision (and rejected several tempting shortcuts):
1. **Attribution, not classification.** The system points at channels/moments; a human concludes.
   This is doctrine, not timidity — it's what keeps the tool defensible.
2. **Air-gapped inference.** Any outbound network handshake at inference is a critical failure
   (`TRANSFORMERS_OFFLINE`/`HF_HUB_OFFLINE` hardcoded).
3. **ELAN labels are VALIDATION-ONLY.** Ground-truth Truth/Lie labels may *score* a method after the
   fact; they may never train or calibrate it. This is why every result here is honest — the model
   never saw the answer key.
4. **Per-subject calibration.** Each subject is normalised against *their own* neutral baseline clip.
5. **Pre-registered falsification.** Before running an evaluation we wrote down the bars — including a
   go/no-go false-positive gate (Bar 4) — and let them kill methods. They did, five times.

---

## 2. The journey, in order (every major attempt and its fate)

### Phase A — Stage-1 diarization substrate (SPOVNOB), pre-July
Built the forensic audio/video diarization front-end (Modules 0–5): manifest-driven, deterministic,
air-gapped, click-anchored target identity, beard/audio-anchored enrollment fallback. This is the
solid foundation everything else stands on — it was never in question.

### Phase B — the merge + the first end-stage decision (2026-07-06)
Merged SPOVNOB with the deception cascade. WavLM swapped in for acoustic features; per-subject
baseline calibration + recording assembly implemented. **Decision: pursue a graph autoencoder
(ST-GAE) as the end-stage, and ARCHIVE the TFN classifier** (`predictive_engine.py`) — it was a
supervised per-window Lie/Truth classifier with *no honest validation path at N=1* (it would just
memorise one subject). First hard call in favour of rigour over a shippable-looking demo.

### Phase C — first real signal test (2026-07-08) — the pivotal finding
First real GPU cascade on SubjectA, scored against ELAN. **The single most important early result:**
- the scalar **`deviation_magnitude` has NO signal** (distance-from-baseline ≠ deception);
- **per-channel** deviations DO (AU12 lip-tremor, hand↔face, silent-speech incongruence, within-clip
  AUCs 0.68–0.70).
Also fixed a dead blink/EAR channel (merge seam had dropped it) and a multi-clip pool-restart bug.
This established the shipped instrument: **per-channel within-clip |z| attribution.**

### Phase D — graph end-stage, attempt 1: reconstruction ST-GAE (2026-07-09) — FALSIFIED
Built a spatio-temporal graph autoencoder; anomaly = reconstruction error. Fit was healthy, but
**Bar 4 falsified it**: reconstruction error tracked distance-from-baseline (clip identity), so the
*most truthful* clip looked *most anomalous*; truthful windows flagged at 81% vs 6% baseline. The
pre-registered gate did exactly its job.

### Phase E — graph end-stage, attempt 2: predictive coupling (2026-07-10) — FALSIFIED
Reframed: hide one modality node, predict it from the other ten (cross-modal coupling should break
under cognitive load). It even beat the marginal path on one channel (0.754) — but **Bar 4 failed
again** (truth flag-rate 93% vs 6%). Same failure mode: the aggregate coupling-z is a
distance-from-baseline meter, not a deception meter. **Graph line closed for N=1.**

### Phase F — the replication question (2026-07-10 → 07-17)
If no unsupervised objective works at N=1, the real question is empirical: **does SubjectA's
per-channel signature replicate in other people?** Built the pre-registered replication scorecard +
generic multi-subject toolchain and processed 5 more annotated subjects (B–F) through the identical
production path.
- **Scorecard fixes along the way** (bugs, not criteria changes): global-vs-local timestamp rebase;
  pooled→within-clip scoring (pooled fails SubjectA's own positive control — a Simpson's-paradox
  clip confound). Regression-tested.
- **N=2 (07-10):** SUBJECT-SPECIFIC — 0 channels replicate A↔B. A leaks via AU12/silent-speech, B
  via blink; disjoint.
- **N=6 (07-17):** SUBJECT-SPECIFIC confirmed — **0 of 134 channels replicate** (8 subject-specific,
  126 no-signal). SubjectB's blink (0.71) doesn't recur and SubjectD's blink is *inverted* (0.23);
  silent-speech leaks for A and D but inverts for B/E/F. Same channel, opposite sign across people.

### Phase G — the two pre-registered "is it really subject-specific?" cross-checks (2026-07-17)
1. **Supervised LOSO head** — train on 5 subjects, test on the held-out 6th. Result: **mean test AUC
   0.469** (0/6 subjects ≥ 0.58) — at/below chance. A model trained on 5 people cannot predict the
   6th; the sub-0.5 tilt comes from per-subject channel inversion actively mis-predicting.
2. **Coupling model, re-run per subject** — **Bar 4 fails on C, D, E** (F under-powered); with A(×2)
   and B, the go/no-go gate fails on all 5 evaluable subjects. SubjectE is the instructive case: it
   passed the three signal bars and Bar 4 still killed it.

**Three independent instruments, one conclusion: deception leakage is idiosyncratic per person.**

---

## 3. What we definitively KNOW now (earned, not assumed)
1. **Per-subject, per-channel behavioural attribution is real and recoverable** — for subjects who
   leak at all (A/B clearly, C/D moderately). The pipeline and calibration are sound.
2. **Scalar/global anomaly is a distance-from-baseline meter, never a deception meter.** Proven four
   ways (deviation_magnitude, ST-GAE recon, coupling-z global, LOSO false-positive rates).
3. **No universal channel transfers across people at N=6.** Not a tooling gap — measured three ways.
4. **Signal strength varies a lot by subject.** E and F have no channel ≥0.60. "Does everyone leak
   detectably?" — at this instrument's resolution, no.
5. **The pre-registration discipline works.** Bar 4 killed every over-claim before it shipped. This
   is the project's most valuable asset and must survive whatever comes next.

## 4. What we do NOT know (the open questions)
- Whether a **weak-but-broad** channel exists that's slightly-above-chance in *many* subjects even
  though none clears 0.60 alone (candidate: `head_pitch_somatic_dominant_freq` appears in the top-3
  of A, C, E, F). A per-subject-bar analysis can't see this; a meta-analysis across subjects could.
- Whether a **within-subject supervised** model (trained on that person's own labelled data) beats
  the marginal |z| — we've never been allowed to try (labels are validation-only, correctly).
- Whether **coarser decision units** (question-level, clip-level) turn modest window AUCs into usable
  accuracy.
- Whether the ceiling is the **construct** (there may be no strong universal deception signal to
  find) or the **N** (6 is tiny; effective sample size = 6 people, not 8,286 windows).

---

## 5. Honest capability statement (what you can claim today)
> "A deterministic, air-gapped forensic instrument that, given a subject's neutral baseline, flags
> per-channel behavioural departures during questioning and attributes them to specific modalities.
> On annotated subjects it recovers within-clip deception-correlated channels with AUC up to ~0.68–
> 0.71 **for that individual**. It makes no cross-subject or automated verdict claim; a rigorous,
> pre-registered evaluation shows the specific channels are subject-specific and do not generalise."

That is a **defensible, honest, and genuinely useful** claim for a forensic-analyst aid. It is *not*
a courtroom lie detector, and the evidence says no window-level universal lie detector is reachable
from this corpus.

---

## 6. Next steps to push accuracy — ranked, with what each REQUIRES
Ordered by expected accuracy-per-unit-effort. Each says honestly what data/doctrine change it needs.
None of these is started; they are options for you to choose among (see the decision at the end).

### 6.1 Coarser decision units (cheap, no new data) — **DONE 2026-07-17**
Aggregated windows into answer-level segments (`validation/multisubject/aggregate_evaluate.py`).
Full write-up: `validation/multisubject/ACCURACY_LEVERS_N6.md`. Result: segment aggregation trends
positive everywhere but is N-fragile (only 58 answer-segments total). The **robust** finding is at
the window level in §B below — the weak-universal *panel* LOSO beats the full-feature LOSO
(0.523 vs 0.469). Stays within the attribution doctrine (aggregating attribution; labels only score;
LOSO holds each subject out). **Cost: low. Needs: nothing new.**

### 6.2 Weak-but-broad channel meta-analysis (cheap, no new data) — **DONE 2026-07-17**
Directional random-effects (DerSimonian–Laird) pool of each channel's signed within-clip AUC across
the 6 subjects (`validation/multisubject/meta_analysis.py`). Found **5 universal-signal candidates**
— all real but tiny (pooled AUC 0.51–0.52, I²≈0, directionally consistent): emotion_confidence_mean
0.522, wavlm_latent_9 0.521, gaze_y_mean 0.516, AU6_max 0.485, AU2_velocity_max 0.514. Key insight:
**strength ⊥ generality** — these universal channels are entirely different from the strong
per-subject ones; the guessed `head_pitch_somatic_dominant_freq` did NOT survive the directional
test. Feeding just these 5 to the LOSO panel (§6.1-B) *improves* cross-subject transfer (0.523 vs
0.469 full) — the general-detector lead. **Cost: low. N=6 → wide CIs; hypothesis-generating.**

### 6.3 Per-subject supervised models (biggest per-person gain — but changes the data regime)
If deployment can enroll a subject with *their own* labelled baseline+interview data, train a
**within-subject** classifier (temporal train/test split, no cross-subject leakage). Given the signal
is subject-specific, a per-person model should beat the marginal |z| meaningfully. Best fit for
**longitudinal / repeat-subject** settings (same person over time). **Cost: medium. REQUIRES a
doctrine change** — ELAN-style labels become *training* data for that individual (still never for
other subjects), which you must explicitly authorise, plus a leak-proof within-subject protocol.

### 6.4 Hierarchical / mixed-effects model (the principled middle path — needs more subjects)
A population-level prior + per-subject random effects: shares statistical strength across people while
letting each have their own channel profile. This is the *correct* statistical answer to
"subject-specific but maybe a weak common core." **Needs moderate N (~20–50 annotated subjects)** to
estimate the population level with any confidence. **Cost: medium. Needs: more data.**

### 6.5 Scale the corpus massively (the only path to a true universal model)
Effective sample size is **subjects (6), not windows.** Subject-specificity at N=6 does not preclude
a weak universal signal emerging under heavy regularisation at N=50–200. This is the single biggest
lever *if* a general model is the goal — but it's the most expensive and may still confirm the null.
**Cost: high (annotation-bound). Needs: a lot of new annotated subjects. Honest prior: may reconfirm
subject-specificity — but at a sample size where that conclusion would be authoritative.**

### 6.6 Richer features / multimodal fusion (lower priority)
More/better features (e.g., learned temporal encoders, refined FFT bands, better acoustic fusion).
Deprioritised: our bottleneck is **N and the construct**, not feature richness — E/F's weakness isn't
obviously a feature-extraction problem. Revisit only after 6.1–6.4. **Cost: medium. Needs: nothing,
but low expected return right now.**

### 6.7 Ground-truth quality (enabling work for everything above)
Inter-annotator reliability, finer temporal labels, more balanced Lie/Truth per clip (SubjectF failed
the coupling FOCUS test purely for lack of balanced windows). Better labels raise the ceiling of
every method above. **Cost: medium (human-annotation). Needs: annotation effort.**

### What NOT to do
- Don't resurrect an unsupervised global-anomaly end-stage (ST-GAE recon / coupling-z) for a single
  subject — closed at both N=1 and N=6 by the same gate.
- Don't build a cross-subject supervised classifier and quote its *within-sample* accuracy — that's
  the memorisation trap the TFN archival and the LOSO honesty gates exist to prevent.
- Don't drop the pre-registration/Bar-4 discipline to make numbers look better. It's the whole
  reason these results are trustworthy.

---

## 7. Recommendation
Do **6.1 + 6.2 immediately** (cheap, no new data, honest accuracy from what we already have). In
parallel, decide the strategic fork that everything else hangs on:
- **Per-subject / personalised** deployment (repeat subjects, per-person enrollment) → **6.3**, and
  it changes the label doctrine — needs your sign-off.
- **General / cross-subject** detector as the end goal → there is no shortcut; it's **6.4 → 6.5**,
  i.e. *many more annotated subjects*, and the N=6 evidence says be prepared for the null.

Both are legitimate. Which one we pursue depends on what the deployed system is actually for and what
data you can get — which is the question I'll put to you rather than assume.
