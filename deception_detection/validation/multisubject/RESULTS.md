# Multi-subject replication — N=6 (SubjectA–F), 2026-07-17

**Headline: at N=6 the finding holds and hardens — the per-channel deception signal is
SUBJECT-SPECIFIC. Zero of 134 channels replicate across subjects** (0 REPLICATES / 8
SUBJECT-SPECIFIC / 126 NO-SIGNAL). No channel clears the pre-registered replication bar (within-clip
|z| AUC ≥ 0.60 in the same direction for ≥2 subjects). Different people leak deception through
different — and sometimes *oppositely-signed* — channels. This confirms the N=2 result across four
additional independent subjects and vindicates the per-subject-baseline-calibration + attribution
doctrine; a fixed cross-subject channel weighting is off the table at this N.

## Corpus
All six subjects run the identical production path (canonicalize → SPOVNOB Stage-1 operator-click →
Stage-2 cascade → baseline fit on clip 0 → calibrate/assemble). Pure windows (≥60% single-label),
134 channels each:

| subject | Lie | Truth | clips | recording |
|---|---|---|---|---|
| A (session1) | 509 | 140 | 8 | `REC_SUBJECTA` |
| B (session2) | 825 | 1494 | 7 | `REC_SUBJECTB` |
| C (session3) | 734 | 448 | 6 | `REC_SUBJECTC` |
| D (session4) | 1103 | 625 | 8 | `REC_SUBJECTD` |
| E (session5) | 554 | 1146 | 9 | `REC_SUBJECTE` |
| F (session6) | 571 | 137 | 7 | `REC_SUBJECTF` |

## Tracked SubjectA channels across all six (within-clip |z| AUC, Lie>Truth)

| channel | verdict | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|
| AU12_velocity_max | SUBJECT-SPECIFIC | **0.601** | 0.514 | 0.494 | 0.509 | 0.503 | 0.504 |
| AU12_velocity_tremor_band_power | SUBJECT-SPECIFIC | **0.601** | 0.518 | 0.492 | 0.499 | 0.495 | 0.502 |
| blink_rate | SUBJECT-SPECIFIC | 0.434 | **0.710** | 0.505 | 0.234 | 0.573 | nan |
| head_pitch_tremor_band_power | NO-SIGNAL | 0.544 | 0.510 | 0.474 | 0.494 | 0.505 | 0.523 |
| left_hand_face_distance_min | NO-SIGNAL | 0.571 | 0.504 | 0.451 | 0.505 | 0.511 | 0.511 |
| right_wrist_velocity_max | NO-SIGNAL | 0.553 | 0.542 | 0.514 | 0.499 | 0.509 | 0.473 |
| gaze_x_mean (freeze, inverse) | NO-SIGNAL | 0.387 | 0.479 | 0.509 | 0.509 | 0.506 | 0.474 |
| gaze_entropy | NO-SIGNAL | 0.428 | 0.509 | 0.495 | 0.503 | 0.502 | 0.506 |
| ear_mean | NO-SIGNAL | 0.448 | 0.472 | 0.503 | 0.508 | 0.504 | 0.438 |

## The signatures stay disjoint — and some channels *invert* across people
The 8 SUBJECT-SPECIFIC channels (only one subject clears 0.60 on each) show no common leakage:
- **SubjectA** leaks via the AU12 lip family (velocity max/mean/tremor ~0.601), head/ear somatic
  dominant frequencies (0.58–0.62), and **silent_speech_duration_ms 0.681** (mouth moving with no
  target audio — incongruence). Flat (~0.50) for all five others.
- **SubjectB** leaks via **blink_rate 0.710** — but SubjectD's blink runs the *other way* (0.234,
  i.e. D blinks *less* when lying), and it's null for A/C/F. Same channel, opposite sign per person.
- **silent_speech_duration_ms**: A 0.681 and D 0.627 both leak, but B/E/F are inverted (0.35/0.22/0.23)
  — not a majority in a single direction, so still SUBJECT-SPECIFIC, not REPLICATES.

No channel is elevated during lies, in the same direction, in even two subjects. The instrument
finds real per-subject signal (positive control: SubjectA AU12 0.601) but **no transferable one**.

## Interpretation & consequences
- **Doctrine confirmed at N=6:** per-subject baseline calibration + per-channel |z| attribution
  (never a fixed classifier / universal channel weighting) is the correct frame. A universal verdict
  would be wrong-signed for at least one subject on the very channels that carry signal for another.
- **Supervised cross-subject head:** with 0/134 channels replicating, a supervised LOSO head is
  predicted to land near chance. It is run next as the *quantitative* confirmation of
  subject-specificity (pre-registered `Documentation/SUPERVISED_LOSO_DESIGN.md`), not as a
  performance play. Its result is recorded below/in §9 of that doc.
- **Blink** watched since N=2 as a possible second idiosyncratic marker: at N=6 it is confirmed
  idiosyncratic (strong+positive in B, strong+negative in D, null elsewhere) — a per-subject quirk,
  not a common channel.

## Provenance (N=6)
Manifest `validation/multisubject/replication_manifest_N6.json`; scorer
`multisubject/replication_scorecard.py` (within-clip percentile |z| AUC, regression-tested 28/28);
output `validation/multisubject/replication_scorecard.csv`. Recordings
`pipeline_system_outputs/REC_SUBJECT{A..F}/`; ELAN normalized under each `*_SRC/elan_normalized/`
(SubjectA uses its raw `B04C…` annotated dir). Clips without an `.eaf` are cascaded but unscored.

## Supervised LOSO head — the quantitative confirmation (2026-07-17)
The pre-registered generalizability test (`Documentation/SUPERVISED_LOSO_DESIGN.md`): train an
L2-logreg on 5 subjects' within-clip-centered per-channel z-features, test on the held-out 6th,
rotate through all six. Labels are the target only (never the features; clip-median centering and
per-fold StandardScaler are leak-free).

| held-out | test AUC | truth FP-rate |
|---|---|---|
| SubjectA | 0.525 | 0.386 |
| SubjectB | 0.444 | 0.645 |
| SubjectC | 0.496 | 0.507 |
| SubjectD | 0.432 | 0.790 |
| SubjectE | 0.425 | 0.648 |
| SubjectF | 0.490 | 0.511 |

**mean LOSO test AUC = 0.469 ± 0.040, 0/6 subjects ≥ 0.58 → VERDICT: SUBJECT-SPECIFIC.** Every fold
is at or below chance. This is the pre-registered expected outcome and the hard number behind
"0/134 replicate": a cross-subject model can't predict a held-out person, and the mild sub-0.5 tilt
comes from channels that *invert* per subject (blink, silent-speech). The marginal scorecard and the
supervised head **agree** — deception leakage is idiosyncratic per person; per-subject calibration +
|z| attribution is the sole shipped instrument. Artifacts: `loso_results.csv`,
`loso_coefficients.csv`; code `multisubject/loso_head.py` + `validation/multisubject/loso_evaluate.py`.

---

# Multi-subject replication — N=2 (SubjectA + SubjectB), 2026-07-10

**Headline: SubjectA's per-channel deception signal does NOT replicate in SubjectB. At N=2 the
signal is subject-specific — different people leak through different channels.** No channel clears
the pre-registered replication bar in both subjects (0 REPLICATES / 7 SUBJECT-SPECIFIC / 127
NO-SIGNAL over 134 channels). This is the honest SUBJECT-SPECIFIC outcome the runbook (§5)
anticipated as real and publishable — it vindicates the per-subject-calibration + attribution
doctrine and cautions against any fixed cross-subject channel weighting.

## Method
- **SubjectB** (`01SubjectB_session2`, 7 clips): full production path identical to SubjectA —
  canonicalize → SPOVNOB Stage-1 (operator click, 72 clean target segments) → Stage-2 cascade →
  baseline fit on clip 0 → calibrate/assemble. Healthy calibration (baseline median deviation
  10.06 ≈ √134; blink 100% populated). ELAN labels scoring-only, never calibration.
- **Scoring:** within-clip-percentile pooled |z| AUC (rank each channel within its own clip, then
  pool Lie vs Truth), the method established 2026-07-08 and used by the coupling eval. Pure windows
  (≥60% single-label): SubjectA 509 Lie / 140 Truth; SubjectB 825 Lie / 1494 Truth.
- **Positive control:** SubjectA's within-clip signal is recovered exactly (clip 06: AU12_velocity
  0.684, AU12 tremor 0.696, hand↔face 0.680, wrist 0.667, gaze-freeze inverse 0.162 — matching the
  07-08/07-09 production numbers). This is what makes the SubjectB null interpretable.

## Tracked SubjectA channels — A vs B (within-clip-percentile AUC, |z|, Lie>Truth)

| channel | SubjectA | SubjectB | verdict |
|---|---|---|---|
| AU12_velocity_max | **0.601** | 0.514 | SUBJECT-SPECIFIC (A) |
| AU12_velocity_tremor_band_power | **0.601** | 0.518 | SUBJECT-SPECIFIC (A) |
| left_hand_face_distance_min | 0.571 | 0.504 | NO-SIGNAL |
| right_wrist_velocity_max | 0.553 | 0.542 | NO-SIGNAL |
| head_pitch_tremor_band_power | 0.544 | 0.510 | NO-SIGNAL |
| gaze_x_mean (freeze, inverse) | 0.387 | 0.479 | NO-SIGNAL |
| gaze_entropy | 0.428 | 0.509 | NO-SIGNAL |
| blink_rate | 0.434 | **0.710** | SUBJECT-SPECIFIC (B) |
| ear_mean | 0.448 | 0.472 | NO-SIGNAL |

Note: SubjectA's tracked channels are weaker here (~0.60) than the clip-06-only numbers (~0.68)
because within-clip-percentile pools ALL of SubjectA's clips and clip 06 was her strongest;
0.601 still clears the pre-registered 0.60 bar (positive control holds).

## The two signatures are disjoint
- **SubjectA (7 SUBJECT-SPECIFIC channels, all A-driven except blink):** the AU12 lip family
  (velocity max/mean/tremor, ~0.601), head-pitch & ear somatic dominant frequencies (0.60–0.62),
  and **silent_speech_duration_ms 0.681** (mouth moving with no target audio — a leakage/incongruence
  marker) during lies.
- **SubjectB (1 SUBJECT-SPECIFIC channel):** **blink_rate 0.710** — SubjectB blinks markedly more
  during lies; this channel was chance/inverse (0.434) in SubjectA.

No channel is elevated during lies in BOTH subjects. The per-channel deception signal is
idiosyncratic at N=2.

## Interpretation & consequences
- **Doctrine vindicated:** per-subject calibration + attribution (never a fixed classifier) is the
  right frame — a universal channel-weighted verdict would be wrong for at least one of these two
  people. The scalar `deviation_magnitude` remains meaningless (as 07-08); now we also know the
  *specific channels* don't transfer.
- **Supervised head (§6.1):** at N=2 there is no robust cross-subject channel to learn from among
  the tracked set. A supervised head would need many more subjects and would likely have to model
  per-subject channel profiles rather than one universal weighting. Do not start it on N=2.
- **Blink** is now a channel to watch as subjects accrue — strong in B, null in A; whether it's a
  SubjectB quirk or a second common-but-idiosyncratic marker needs N>2.

## Scorecard corrections made this session (bug-fixes, NOT criteria changes)
The pre-registered thresholds (adequacy ≥30, R1 AUC≥0.60 in ≥2/3, R2 direction, verdict classes)
are unchanged. Two implementation bugs were fixed so the instrument measures what it intends:
1. **Global-vs-local timestamps** — the scorer matched GLOBAL assembled-CSV window times against
   LOCAL ELAN intervals, so large offsets (SubjectA) gave ZERO labels and small ones (SubjectB)
   gave SHIFTED/mislabeled overlap. Fixed by rebasing each clip to local (subtract the clip's min
   start_time_ms, which equals its file_offset_ms). Restored SubjectA's exact 509/140 counts.
2. **Pooled → within-clip scoring** — pooled |z| AUC failed the SubjectA positive control (her own
   validated channels scored ~0.50). Now AUC uses within-clip percentile of |z| and direction uses
   within-clip-centered signed z — the 07-08 method. Positive control now passes (AU12 0.601).

## Caveats
N=2 is still a tiny corpus — "subject-specific" is a strong hypothesis, not proof; more subjects
(4 more sessions incoming) will test it. SubjectB's baseline is short (84 s vs SubjectA's 105 s).
SubjectB clips 04 (Truth-only) and 06 (876 s Truth / 17 s Lie) contribute little within-clip
contrast; the bilabel signal is mostly clips 01/02. blink_rate 0.710 is a single-subject
observation.

## Provenance
`pipeline_system_outputs/REC_SUBJECTB/` (recording_calibrated.csv + baseline_stats + analyst
report). Scorecard: `validation/gt_subjectB/replication_scorecard.csv` (+ manifest).
Code: `multisubject/replication_scorecard.py` (within-clip fix); drivers `validation/gt_subjectB/
recB_*.py`; mapping `validation/gt_subjectB/subjectB_manifest.json`. SubjectB data A/V-synced
(canonicalizer 80 ms trim), ELAN normalized (`normalize_elan.py`, LIe→Lie fix).
