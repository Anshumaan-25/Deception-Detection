# Ground-truth validation — SubjectA_session1 (2026-07-08)

First quantitative validation of the deviation pipeline against human deception annotations.
**Labels were used for scoring only — never for training or calibration** (calibration stayed
unsupervised on `00_baseline`; ELAN labels overlaid post-hoc).

## Data
`my_videos/00SubjectA_session1/`: `00_baseline.mp4` (Neutral, the calibration clip) +
`01..07_interview.mp4`, one subject / one session. ELAN `.eaf` labels Truth/Lie/Neutral
(`annotated Videos anushree/`). Video↔annotation mapping user-confirmed sequential
(`B04C001→00 … B04C008→07`; `01_interview`=B04C002 is unannotated). Richest clip:
`06_interview` = Truth×3 (3.7–64.9s) → Lie×8 (88.2–309.4s).

## Method
Per-clip cascade (`calibrate=False`, no SPOVNOB — ELAN times are clip-local) →
`BaselineCalibrator.fit(00_baseline)` (106 windows, 134 features) → `apply()` per clip →
window labeled by majority ELAN overlap (window counted as "pure" at overlap ≥ 0.6) →
rank (Mann-Whitney) AUC. Pure windows pooled: Lie=509, Truth=140.

## Results

**The single scalar `deviation_magnitude` (L2 over 134 z-scores) carries no usable signal:**

| aggregate | Lie-vs-Truth AUC |
|---|---|
| L2 (current) | 0.384 (inverted) |
| median \|z\| | 0.505 |
| L2 minus top-5/window | 0.435 |
| count of \|z\|>3 | 0.400 |

Diagnosis (now measured, previously suspected from rec_ca clip-2): top-5 features carry a
median **42%** (p90 73%) of each window's Σz²; the dominant features are the body
`motion_energy` / `macro_motion_energy` family — which for this subject runs **higher during
Truth** (med|z| ≈ 2.7–3.5 vs 1.6–1.7 in lies; a stillness-during-lies / freeze pattern).
Opposing per-channel directions cancel or invert in any scalar aggregate.

**Individual channels DO separate, in coherent clusters** (within `06_interview` only —
immune to clip-identity confounds; n_lie=189, n_truth=51):

| channel (within-clip 06) | AUC |
|---|---|
| AU12 velocity tremor band power | 0.681 |
| left hand↔face distance (min) | 0.679 |
| AU12 velocity max / var / somatic / mean | 0.664–0.677 |
| right/left wrist velocity max | 0.631–0.660 |
| AU1 velocity max | 0.635 |
| head-pitch tremor band power | 0.625 |
| AU4 var | 0.624 |
| gaze x/z mean & var, gaze entropy (inverse: Truth>Lie) | 0.162–0.345 |

Pooled (all clips) top cluster was AU9/AU4/disgust_leak/head_pitch (0.61–0.68); within-06 the
AU9/disgust part shrinks toward chance (0.53–0.58) — partially clip-confounded — while
AU4_var and head_pitch hold (~0.62).

Behavioral story for this subject during lies: **more facial micro-dynamics (AU12/AU1
velocity+tremor), more hand-near-face, more wrist motion, suppressed gaze variability, and
reduced gross body-motion energy.**

## Caveats
N=1 subject; windows overlap (2s/1s hop → adjacent windows share data, effective n is lower);
134 comparisons (best within-06 AUCs remain ≳3 SE above 0.5 after overlap deflation, but treat
magnitudes as provisional); annotator = single coder. Effect sizes (0.62–0.68 window-level)
are consistent with the deception literature's ceiling — encouraging, not conclusive.

## Implications
1. **Empirical mandate for attribution over any scalar score** — the ST-GAE end-stage design
   (per-channel deviation attribution) is now justified by measurement, not doctrine alone.
2. `deviation_magnitude` should be treated as bookkeeping, not signal; a per-channel deviation
   report (z per feature-family per window) is the forensic deliverable.
3. Cross-modal channels behave in opposite directions per subject — any future aggregate must
   be direction-aware and per-subject.

## Reproduce
Outputs + canonical media: `pipeline_system_outputs/GT_SUBJECTA_20260708/` (gitignored).
Scripts (this dir): `gt_cascade.py` (per-clip cascade), `gt_score.py` (calibrate + ELAN
overlay + AUC), `gt_attrib.py` (per-feature attribution, L2-domination, within-clip control).
Canonicalize with **system** ffmpeg (`/usr/bin/ffmpeg`, NVENC); run cascades with
`~/anaconda3/envs/spovnob_env/bin/python` (absolute path — the project `.venv` shadows conda).
