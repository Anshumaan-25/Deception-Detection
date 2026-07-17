# Structural signatures — pre-registered design (the "universal is a structure, not a channel" test)

**Status:** DESIGN LOCKED 2026-07-17, pre-registered **before any implementation or scoring**.
**Prompted by:** the N=6 verdict (channels are subject-specific; 0/134 replicate; LOSO 0.469) plus the
one honest general-detector lead (weak-universal panel, LOSO window AUC **0.523**). Every prior
method asked *"which channel is universal?"* — answer: none. This programme asks the next question:
**is there a universal *structure* one level above channel identity?** Three structural hypotheses,
one primary endpoint, fixed bars, and validity kill-switches written down now — per the discipline
that let Bar 4 falsify the ST-GAE, the coupling model (5×), and the LOSO head honestly.

**Doctrine unchanged:** attribution, never a verdict. All scores below are per-window *attributions*
("this window shows freeze-and-leak structure / face↔voice contradiction"), never Lie/Truth output.
ELAN labels never touch feature construction or calibration; they are the LOSO evaluation target
only (M1/M3 are fully label-free constructions; M2 uses labels in training folds only, under the
`SUPERVISED_LOSO_DESIGN.md` research-instrument precedent). Production ships nothing from this doc
unless a bar is cleared *and* re-confirmed on subjects #7+.

---

## 0. Governance — scoped re-opening of two §14 doors (explicit, not silent)
`MASTER_REFERENCE.md` §14 closed three relevant doors after N=6. This pre-registration re-opens two,
narrowly, and *keeps the third shut*:

| §14 door | status here |
|---|---|
| "no more features / bigger models" | **Re-opened, scoped:** ONLY structure-level *re-representations of the existing 134 calibrated channels* (rank/shape/pair arithmetic). NO new extraction from video/audio; no pipeline changes. |
| "no per-subject supervised model" | **Stays closed.** Nothing here is per-subject supervised. |
| "no unsupervised global-anomaly end-stage" | **Stays closed.** M3 is designed to be *provably not that*: it fits **no model at all** (no baseline fit, nothing learned that could encode distance-from-baseline) — see §4 and its kill-switch. |

Multiple-comparison control (the audit's blocker #3): **M1 is the single PRIMARY hypothesis.** M3 is
SECONDARY, M2 is TERTIARY and **gated** (runs only if M1 or M3 reaches PROMISING). A claim of
progress may rest on the primary endpoint of the primary method; secondary results are reported as
exploratory regardless of how good they look.

## 1. The unifying hypothesis
Deception signal is real per person but channel-idiosyncratic (N=6, three instruments agreeing).
Hypothesis: the invariant lives one level up, in the **structure of the deviation vector**, not in
which channels deviate:
- **H1 (shape / freeze-and-leak):** lying produces simultaneous *suppression* of some controllable
  channels and *leakage* through some uncontrollable ones — the specific channels differ per person,
  but the **co-occurrence of "abnormally still" and "spiking" in the same window** is shared.
- **H3 (contradiction):** lying elevates *pairwise incongruence* between modality families that are
  normally co-active (face↔voice, mouth-AU↔speech-activity…). Supported by our own data: the one
  quasi-recurring channel across subjects is an incongruence feature (`silent_speech_duration_ms`,
  A 0.68 / D 0.63).
- **H2 (geometry):** a low-capacity learned embedding exists in which the lie-vs-truth separation
  *direction* is shared across subjects even though the driving channels differ.

## 2. Primary endpoint + pre-registered bars (fixed for all three methods)
**Endpoint:** strict leave-one-subject-out **window-level** AUC on the N=6 corpus (mean over the 6
held-out subjects + all per-subject values; same folds, same `_model()` L2-logreg evaluation layer
as `loso_head.py`). Segment-level numbers are reported but carry no decision weight (58 segments —
N-fragile, per `ACCURACY_LEVERS_N6.md`).

Reference points (fixed): chance 0.50; full-134 LOSO **0.469**; weak-universal panel **0.523** (the
number to beat).

| verdict | bar |
|---|---|
| **BREAKTHROUGH** | mean LOSO window AUC ≥ **0.62** AND ≥ 4/6 held-out subjects ≥ 0.58 (same bar as `SUPERVISED_LOSO_DESIGN.md` GENERALIZES — deliberately identical) |
| **PROMISING** | mean ≥ **0.56** AND beats the panel (≥ 0.543 = panel + 0.02) AND ≥ 4/6 subjects > 0.52 |
| **NULL** | anything else — the structure hypothesis (that variant) dies at N=6 |

**Validity kill-switches (any one → the method is INVALID regardless of its AUC):**
1. **Truth-FP honesty (Bar-4 analog, all methods):** held-out truth windows must not be scored as
   lie-like at a rate rivalling lie windows; report held-out truth FP-rate per subject; if truth ≈
   lie ≫ chance-rate, the score is a confound, not deception.
2. **Missingness confound (M1):** |Spearman| between any shape feature and the window's original
   missing-channel count must be < 0.3 across each subject. Fails → the shape features are a
   missingness detector (the audit's identified leak: acoustic NaNs are silence-correlated).
3. **Magnitude confound (M3):** |Spearman| between mean contradiction score and mean |deviation|
   percentile must be < 0.5 within each subject. Fails → it's a distance-from-baseline meter with a
   new name → discarded exactly as coupling v2 was.
4. **Subject-ID shortcut (M2):** the embedding input must contain no per-subject-constant columns
   (asserted in the fixture), and a linear subject-ID probe on the embedding is reported; strong
   subject decodability + weak transfer = shortcut, reported as such.

**Honest priors (written before any code):** M1 — genuinely uncertain; first instrument here that
*could* plausibly clear PROMISING, because it is built on (not against) the subject-specificity
finding. M3 — weakly positive prior (silent-speech recurs in A/D) but PROMISING at best;
BREAKTHROUGH would surprise. M2 — prior is NULL at N=6 (echoing the LOSO head); it exists to be
ready at N≈12, which is why it is gated. If all three land NULL, the structure hypothesis is dead at
this N and the general-detector question returns fully to "more subjects" — a legitimate outcome.

## 3. M1 — Deviation-shape features ("freeze-and-leak") — PRIMARY
**Construction (label-free, per window):** operate on the existing within-clip representation (the
scorecard's unit): per channel, the within-clip percentile of |centered deviation| (`pct`), and the
within-clip percentile of the *signed* centered value (`spct`). All rank-based → unit-free, robust
to the clip-confound by construction.

**Channel support S (fixes the audit's NaN + mixed-family traps) — frozen before scoring:**
- A new hand-curated map `multisubject/channel_families.py` assigns each of the 134 channels a
  family (hands, kinematics, head, gaze, blink/EAR, AU-upper, AU-mid, AU-mouth, AU-dynamics/FFT,
  acoustic, wavlm-latent, incongruence, emotion) and a semantic type: **activity-like**
  (velocities, variances, energies, band-powers, rates, durations — suppression is meaningful) vs
  **level-like** (means/positions — "low" ≠ "frozen").
- **S = activity-like channels whose NaN-fraction ≤ 10% in EVERY subject** (measured once on the
  N=6 corpus, then frozen; report |S|). This mechanically excludes `blink_rate` (100% NaN on F) and
  the acoustic-volatility family (18–38% NaN, silence-correlated — the leak-shaped confound).
  **wavlm-latent channels are excluded from S** (uninterpretable as freeze/leak; 20 of them would
  dominate the tails). Remaining NaNs inside S impute to the clip median (= neutral rank 0.5).
- The map is reviewed and committed **before** `structure_evaluate.py` ever runs; changing it after
  seeing results is prohibited.

**The 7 shape features (fixed list, no tuning):** over S per window —
1. `leak_top5` — mean of the 5 highest `pct` values ("something is screaming, whoever it is");
2. `high_frac` — fraction of channels with `pct` > 0.9;
3. `freeze_frac` — fraction with `spct` < 0.1 (abnormally *below* this clip's own activity norm);
4. `freeze_and_leak` — `high_frac × freeze_frac` (the H1 interaction term — the core claim);
5. `concentration` — 1 − normalized entropy of the `pct` distribution across S (peaky vs diffuse);
6. `spread` — IQR of `pct` across S;
7. `sign_asym` — mean(`spct`) − 0.5 (net suppression vs net excitation).

**Evaluation:** LOSO logreg on the 7 shape features alone (**primary**); shape + the 5 panel
channels (**secondary**, reported as exploratory). Kill-switches 1–2 apply.

**Required fixtures (added to `tests/verify_multisubject.py` BEFORE the real run):**
- *Positive control:* planted world where lie windows have freeze+leak structure on **different
  random channels per subject** — shape-LOSO must catch it (≥ 0.70) while channel-identity LOSO
  stays ≈ chance on the same world (the exact regime the old methods fail by construction).
- *Magnitude negative control:* lie = uniform inflation of all channels (same total |z|, no
  freeze/leak shape) → `freeze_and_leak`-driven LOSO must NOT beat chance (proves shape ≠ magnitude
  re-encoding — the Bar-4-shaped trap).
- *Missingness control:* inject label-correlated NaN patterns → kill-switch 2 must fire.

## 4. M3 — Pairwise cross-modal contradiction — SECONDARY
**What makes this NOT coupling v2 (pinned):** (i) **no fit of any kind** — no baseline-trained
model, no learned normality, nothing that can encode distance-from-baseline; pure within-clip rank
arithmetic on existing channels; (ii) contradiction is **definitionally orthogonal to global
arousal**: under a global deviation (the thing that killed coupling via Bar 4) all families move
*together*, so pairwise |a−b| stays LOW; a high score requires one-family-high-AND-one-family-low
simultaneously; (iii) within-clip percentile space throughout (clip confound controlled). Kill-switch
3 enforces (ii) empirically; if it fails anyway, M3 dies the same death as coupling and we say so.

**Construction (label-free, per window):** family activity `a_f` = mean within-clip `pct` of family
f's activity-like channels (same `channel_families.py`, same S-style NaN rule per family). Six
pre-named pairs (final memberships resolved when the family map freezes, pairs themselves fixed
now): face-AU-dynamics↔voice-activity, AU-mouth↔speech-activity, head↔gaze, hands↔face-AU-dynamics,
voice-activity↔hands, AU-upper↔AU-mouth. Features = the 6 pairwise |a_f − a_g| + their max + their
mean (8 total). Evaluation: LOSO logreg on the 8; kill-switches 1 + 3.

**Required fixture:** planted world where lie = one family up AND one family down (contradiction)
vs truth = both up (arousal, magnitude-matched) → M3 must separate (≥ 0.70) while any pure-magnitude
scorer reads ≈ chance; plus a kill-switch-3 firing fixture (global-arousal world → M3 must NOT fire).

## 5. M2 — Contrastive subject-invariant lie axis — TERTIARY, GATED
**Gate (pre-registered):** implemented/run ONLY if M1 or M3 ≥ PROMISING. Rationale: highest capacity
at tiny N; without a simpler instrument first establishing structural signal, a learned embedding on
6 subjects is overfit theater.

**Doctrine:** labels as training target on LOSO train folds only, research-instrument framing per
`SUPERVISED_LOSO_DESIGN.md`; never production. **Anti-shortcut (mandatory, the audit's trap):**
input = M1's 7 shape features + M3's 8 contradiction features + family aggregates — **never the raw
134** (kills imputation fingerprints like F's constant blink column); encoder = **linear projection**
(primary; capacity too low to carve 5 subject subspaces) with L2, triplet loss on within-subject
(anchor-truth, positive-truth, negative-lie) triplets pooled across train subjects; a 1-hidden-layer
(≤16 units, dropout ≥ 0.3) variant as secondary only. Determinism: reuse `coupling_fit._seed`.
Evaluation: frozen encoder → held-out subject → logreg on train folds' embeddings → LOSO endpoint;
kill-switches 1 + 4. Fixture: planted world where each subject separates on different input features
but a common 2-D structure exists → the linear axis must find it; leak fixture asserts the input
contains no per-subject-constant column.

## 6. Execution order (gates between steps)
1. `multisubject/channel_families.py` — curate, review, **freeze** (shared by M1+M3; the only new
   artifact with human judgment in it — this is deliberately done before any result exists).
2. M1 fixtures green → run M1 via `validation/multisubject/structure_evaluate.py` on
   `replication_manifest_N6.json`.
3. M3 fixtures green → run M3 (parallel with 2 is fine — different feature families).
4. M2 only if the §5 gate opens.
5. Outcome recorded in §8 below + `MASTER_REFERENCE` changelog same turn; any ≥ PROMISING method
   becomes a **candidate panel member** re-tested on subjects #7+ (slots into the §14 watch-metric
   machinery: does its LOSO window AUC hold/climb at N≈12?).

## 7. Deliverables
- `multisubject/channel_families.py` — the frozen 134-channel family/semantic map (M1+M3 shared).
- `multisubject/shape_features.py` (M1), `multisubject/contradiction_features.py` (M3),
  `multisubject/lie_axis.py` (M2, gated).
- `validation/multisubject/structure_evaluate.py` — one driver: builds features, runs the LOSO
  endpoint + all applicable kill-switches, prints per-subject + mean AUC + verdict per the §2 bars.
- Fixtures in `tests/verify_multisubject.py` (suite grows from 32; every planted world above).
- MASTER_REFERENCE §14/§15 + changelog entries, same turn as each landing.

## 8. Empirical outcome
*(intentionally empty at design time — to be filled by `structure_evaluate.py` results only, per the
pre-registration discipline. Expected fill order: M1, M3, then M2 iff gated open.)*
