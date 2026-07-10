# Multi-subject replication — plan, tooling, and desktop runbook

**Status:** tooling BUILT + tested on the laptop 2026-07-10 (`tests/verify_multisubject.py`
19/19). **Data pending:** additional annotated subjects from the *same experiment*
(identical question set + interview structure across subjects; no further SubjectA sessions
exist) — to be shared by the user ~2026-07-11. This document is the runbook for the desktop
Claude Code session that processes them.

> **Read this if you are Claude Code on the work desktop.** Everything below was written on
> the laptop, before the new videos existed, so that you can pick up without re-deriving
> context. The tools are done; your job is to *run* them on real data and record what happens.

---

## 1. Why this exists — the N=1 ceiling

Every validated claim in this project is, so far, a claim about **one human being** (SubjectA,
one session). The headline finding — per-channel |z| attribution separates Lie from Truth at
within-clip AUC 0.68–0.70 on channels like AU12 velocity dynamics and hand↔face distance, while
the scalar deviation magnitude does *not* (RESULTS.md, RESULTS_PRODUCTION.md) — could be a real
behavioral signal or a SubjectA quirk. There is no way to tell from one subject. More annotated
subjects is the single highest-value input the project can receive, and it is the hard
prerequisite for any supervised model (§ future work).

The purpose of the new corpus is to answer exactly one question: **does SubjectA's per-channel
signal replicate across subjects?** The tooling here is built to answer it *honestly* — with
criteria fixed in advance, so we cannot rationalize whatever we happen to see.

## 2. What was built (`deception_detection/multisubject/`)

### 2.1 `intake_validator.py` — gate a subject package BEFORE spending GPU time

A "subject package" is one directory holding a subject's session videos and (for annotated
subjects) their ELAN `.eaf` files — the shape of `my_videos/00SubjectA_session1/`.

Run: `python -m multisubject.intake_validator <subject_dir> [--no-media-probe]`
(from `deception_detection/`). Prints a PASS/WARN/FAIL checklist, writes
`intake_validation.json` next to the package, exits non-zero on any FAIL.

**Clip-index convention (established by SubjectA — keep it):**
- `C001` = the dedicated neutral **baseline** clip → needs **no** `.eaf`.
- `C002..Cnnn` = interview clips → each needs a matching `.eaf`.
- pipeline **`file_index f` ↔ clip `C{f+1:03d}`** (baseline is `file_index 0`).
- videos/eafs are matched on the `C###` token anywhere in the filename; files without it fall
  back to sorted order (WARN — rename them).

**Checks:** V1 ≥2 videos with known extensions · V2 baseline identifiable + unambiguous ·
V3 no duplicate clip indices · E1 every interview has an `.eaf` (FAIL) / orphan `.eaf`s (WARN) /
no annotations at all (WARN — subject is processable but yields no validation) · E2 every `.eaf`
parses with `end>start` intervals · E3 label vocabulary ⊆ {Truth, Lie, Neutral} (else WARN) ·
E4 ≥10 s annotated per `.eaf` · M1 (needs `ffprobe`; `--no-media-probe` to skip) every video
readable with **both** audio and video streams · M2 baseline duration ≥60 s (WARN) / ≥30 s
(FAIL below).

**Why a WARN vs a FAIL:** FAIL = the cascade would crash or produce nonsense (no baseline, a
missing/corrupt `.eaf`, a duplicate index, a stream-less video). WARN = processable but
imperfect (unknown labels, thin annotation, an unannotated subject). An unannotated subject is
deliberately only a WARN — it can be *processed* (calibration + report) and stress-tests the
pipeline across people; it simply contributes no AUC.

### 2.2 `replication_scorecard.py` — does the signal replicate, on pre-registered terms

After each subject has been through the cascade (§4), this consumes each subject's
`*_recording_calibrated.csv` + ELAN dir, scores every calibrated channel per subject
(|z| AUC, Lie vs Truth, pure windows only, ELAN for SCORING ONLY), and emits one
`replication_scorecard.csv`: channel × subject AUCs + a per-channel verdict.

Run: `python -m multisubject.replication_scorecard <manifest.json> [--out DIR]`
where the manifest is
```json
{"subjects": [
  {"name": "SubjectA", "recording_dir": "pipeline_system_outputs/REC_SUBJECTA",
   "elan_dir": "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"},
  {"name": "SubjectB", "recording_dir": "pipeline_system_outputs/REC_SUBJECTB", "elan_dir": "..."}
]}
```

#### PRE-REGISTERED CRITERIA — locked 2026-07-10, before any non-SubjectA data was seen
Changing these *after* looking at new results invalidates the exercise (this is the same
discipline that let Bar 4 falsify the ST-GAE honestly).

- **ADEQUACY** — a subject counts for a channel only with **≥30 pure Lie AND ≥30 pure Truth**
  windows with non-NaN values on that channel (a "pure" window is ≥60% covered by one label,
  as in the SubjectA scorer).
- **R1 (magnitude)** — `|z| AUC ≥ 0.60` in **≥2/3 of adequate subjects**, with ≥2 adequate.
- **R2 (direction)** — a subject *expresses* a direction only when `|median lie z − median
  truth z| ≥ 0.25` (a noise-sized median difference has no informative sign; note that with 3
  subjects "2 of 3 share a sign" is pigeonhole-guaranteed for pure noise, which is exactly why
  the 0.25 floor exists). R2 holds when ≥2/3 of adequate subjects express a direction **and**
  all expressed directions share one sign.
- **Per-channel verdict:** `REPLICATES` (R1∧R2) · `DIRECTION-ONLY` (R2, weak magnitude) ·
  `SUBJECT-SPECIFIC` (≥2 adequate, some single subject clears 0.60 but it doesn't generalize) ·
  `NO-SIGNAL` (≥2 adequate, nothing clears 0.60 anywhere) · `INSUFFICIENT-DATA` (<2 adequate).

The scorecard reports SubjectA's validated `TRACKED_CHANNELS` first — "does *this* set
replicate?" is the whole point. It also scores every other channel, so a signal that is
*stronger* across subjects than it was in SubjectA can surface.

### 2.3 Tests — `tests/verify_multisubject.py` (19 checks, CPU, synthetic)

Intake: well-formed → PASS; each failure class fires its check (missing baseline, missing/corrupt
`.eaf`, duplicate index, too-few videos); label/annotation issues WARN; unannotated subject
processable; verdict JSON written. Scorecard: a planted 3-subject world where one channel
separates in all subjects (→REPLICATES), one is a single-subject quirk (→SUBJECT-SPECIFIC — the
noise-sign pigeonhole must NOT rescue it), one is consistent-but-weak (→DIRECTION-ONLY), one is
pure noise (→NO-SIGNAL), one all-NaN (→INSUFFICIENT-DATA); adequacy floor + baseline/low-overlap
exclusion verified.

## 3. Doctrine reminders (do NOT violate when processing the new corpus)

- **ELAN labels are for scoring only** — never calibration, never training. Calibration fits on
  `C001` (the neutral baseline clip) alone; labels never touch the fit.
- **Per-subject calibration.** Each subject is their own control: fit `BaselineStats` on *their*
  `C001`, apply to *their* clips. Never pool baselines across subjects.
- **Attribution, not classification.** The system reports which channels deviated and how; it
  emits no verdict. The scorecard measures *how well channels separate* for research — it is not
  a classifier and must never be wired into production output.

## 4. Desktop runbook — do this when the videos arrive

Prerequisite: `git pull` on the desktop so these commits are present. Work from
`deception_detection/`, conda env `spovnob_env`.

1. **Validate every package first** (cheap, no GPU):
   `python -m multisubject.intake_validator <path/to/SubjectB_session>`
   Fix every FAIL before proceeding. Re-run until PASS/WARN.
2. **Run the cascade per subject** exactly as SubjectA was run (canonicalize → SPOVNOB Stage-1
   → Stage-2 per-clip cascade → recording-level fit/apply/assemble). The recording-level step
   now also emits `<rid>_analyst_report.html` (Pass 5). Reuse `validation/gt_subjectA/recA_*.py`
   as the template — they encode the exact invocation.
3. **Sanity-check each subject's report** — open `<rid>_analyst_report.html`. If the
   data-quality panel shows a DEGENERATE baseline or many dead channels, stop and fix that
   subject before scoring; a broken calibration poisons its AUCs.
4. **Build the scorecard manifest** (§2.2) listing SubjectA + each new subject, then:
   `python -m multisubject.replication_scorecard manifest.json`
5. **Record the outcome** — write a short `validation/multisubject/RESULTS.md` (headline: which
   TRACKED channels REPLICATE, which are SUBJECT-SPECIFIC, any new cross-subject signal), add a
   MASTER_REFERENCE changelog line, and update §14. Attach `replication_scorecard.csv`.
6. **Re-run the coupling evaluation per subject** (`coupling_evaluate.py`, adjusting TAG/paths):
   the coupling model now gets N independent shots at its four pre-registered bars instead of
   one. Record per-subject verdicts.

## 5. Interpreting the verdicts

- **Several TRACKED channels REPLICATE** → the strongest result the project has ever had: the
  per-channel attribution generalizes. Promote those channels in the analyst report's node
  table; consider the supervised head (§6).
- **TRACKED channels go SUBJECT-SPECIFIC** → the SubjectA signal was (partly) a quirk. This is a
  real, publishable finding and redirects effort — do not bury it. The scalar-deviation null
  (AUC 0.384) was exactly this kind of honest negative and it shaped the whole ST-GAE mandate.
- **A non-TRACKED channel REPLICATES** → a new cross-subject signal SubjectA under-showed; add
  it to `TRACKED_CHANNELS` and the report.

## 6. Future work unlocked by N>1 (deferred until the corpus exists)

These are **not** startable on the laptop today — they need the multi-subject data. Recorded
here so the desktop session knows the roadmap.

### 6.1 Supervised attribution head (roadmap §14 item; becomes legitimate at N>1)
With ≥3 annotated subjects, a small supervised model over the **per-channel z-features** becomes
defensible for the first time. Design constraints, pre-committed:
- **Model:** gradient-boosted trees (e.g. `xgboost`/`lightgbm`) or L2 logistic regression over
  the calibrated window features — interpretable feature importances, not a black box.
- **Validation:** strict **leave-one-subject-out** cross-validation. Reporting within-subject
  accuracy would just re-memorize SubjectA — LOSO is the only honest metric at this N.
- **Doctrine boundary:** the head informs *channel weighting / ranking* in the attribution
  report; it does **not** emit a per-window Lie/Truth verdict in production. It is a research
  instrument for "which channels carry signal across people," not a deployed classifier.
- **Prereq:** ≥3 subjects PASS intake and clear calibration health; the archived
  `analytics/predictive_engine.py` (a superseded TFN classifier) is the cautionary precedent —
  it was archived precisely because it had no honest validation path at N=1.

### 6.2 Coupling model, N>1 (COUPLING_MODEL_DESIGN.md)
The predictive coupling model (`stgae/coupling_*.py`) currently has one shot at its four
pre-registered bars on SubjectA (`coupling_evaluate.py`, pending on the desktop). With N>1 it
gets an independent evaluation per subject; a coupling that breaks during deception in ≥2/3
subjects would be the relational analogue of a REPLICATES channel. Same bars, same decision rule.

### 6.3 Cross-subject baseline-duration study (opportunistic)
Open design question (COUPLING_MODEL_DESIGN, ST_GAE_DESIGN): minimum baseline duration for the
learned fits. If subjects arrive with differing baseline lengths, the scorecard's per-subject
fit health vs baseline duration is free data toward answering it.
