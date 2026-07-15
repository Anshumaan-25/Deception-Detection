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
