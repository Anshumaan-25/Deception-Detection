# RESUME STATE — N=6 multi-subject replication (checkpoint 2026-07-15)

> **If the user says "continue yesterday's task", start here.** This is the live handoff for the
> N>1 replication run. Read this top-to-bottom, then run the "WHAT TO DO ON RESUME" checklist.
> Authority for everything else: `Documentation/MASTER_REFERENCE.md` (+ its changelog).

> **✅ N=6 PROGRAMME COMPLETE (2026-07-17).** All six subjects (A–F) cascaded + assembled; the
> N=6 replication scorecard AND the pre-registered supervised LOSO head both ran. **VERDICT:
> SUBJECT-SPECIFIC.** Scorecard: 0 REPLICATES / 8 SUBJECT-SPECIFIC / 126 NO-SIGNAL over 134 channels.
> LOSO: mean held-out test AUC 0.469 (0/6 subjects ≥ 0.58) — a model trained on 5 people predicts the
> 6th at/below chance. Deception leakage is idiosyncratic per person; per-subject calibration +
> |z| attribution is vindicated as the sole shipped instrument. Full write-ups:
> `validation/multisubject/RESULTS.md` (N=6 section) + `Documentation/SUPERVISED_LOSO_DESIGN.md` §9.
> Nothing is running; all outputs on disk. **Next only if the user wants it:** optional per-subject
> coupling 4-bar re-eval (low value now the marginal signal demonstrably doesn't transfer), or
> extend the corpus (N ≫ 6) before revisiting any universal model. Local commit made; user pushes.
>
> <details><summary>(historical) 2026-07-15 wrap-up note — superseded by the line above</summary>
>
> ALL PROCESSES STOPPED at user request. SubjectC's Stage-2 cascade killed mid-run (2/6). On resume,
> C was re-cascaded from scratch + assembled; D/E/F clicked → Stage-1 → cascade → assemble; then the
> N=6 scorecard + LOSO. All done — see the ✅ note above.
> </details>

## 1. The goal
Test whether SubjectA's per-channel deception signal **replicates across subjects**. Corpus:
SubjectA (session1), SubjectB (session2), and 4 new sessions SubjectC–F. Production path per subject:
canonicalize → SPOVNOB Stage-1 (operator click) → Stage-2 cascade → baseline-fit/calibrate/assemble
→ within-clip replication scorecard.

## 2. Result so far (N=2): SUBJECT-SPECIFIC
SubjectA vs SubjectB: **0 channels replicate**. SubjectA leaks via AU12 lip-tremor + silent-speech
incongruence; SubjectB via **blink rate (0.71, null in A)**. Per-channel leakage looks idiosyncratic
per person. Full write-up: `deception_detection/validation/multisubject/RESULTS.md`. The graph line
(ST-GAE recon + coupling) is CLOSED — falsified 3× by Bar 4 (SubjectA×2, SubjectB×1).
**Adding C–F turns N=2 → N=6 to confirm/deny the subject-specificity.**

## 3. Per-subject status (as of this checkpoint)
| Subject | prep+canon | clicked | Stage-1 | cascade | assembled | scored |
|---|---|---|---|---|---|---|
| A (session1) | ✓ | ✓ | ✓ | ✓ | ✓ (`REC_SUBJECTA`) | ✓ |
| B (session2) | ✓ | ✓ | ✓ | ✓ | ✓ (`REC_SUBJECTB`) | ✓ |
| C (session3) | ✓ | ✓ | ✓ | ✓ | ✓ (`REC_SUBJECTC`) | ✓ |
| D (session4) | ✓ | ✓ | ✓ | ✓ | ✓ (`REC_SUBJECTD`) | ✓ |
| E (session5) | ✓ | ✓ | ✓ | ✓ | ✓ (`REC_SUBJECTE`) | ✓ |
| F (session6) | ✓ | ✓ | ✓ | ✓ | ✓ (`REC_SUBJECTF`) | ✓ |

**All six complete + scored (N=6 verdict SUBJECT-SPECIFIC).** The steps below are retained only as
the reusable recipe for any *future* subject (session7+); nothing in the current corpus needs them.

## 4. WHAT TO DO ON RESUME (checklist)
Work from `deception_detection/`, conda env `spovnob_env`. `PY=~/anaconda3/envs/spovnob_env/bin/python`.

**First, take stock** — see what finished overnight:
```
ls pipeline_system_outputs/REC_SUBJECT{C,D,E,F}/*_recording_calibrated.csv 2>/dev/null   # assembled?
ls pipeline_system_outputs/REC_SUBJECT{C,D,E,F}_00*/*_windowed_features.csv 2>/dev/null | wc -l
ls audio_diarization/session/rec_subject{c,d,e,f}/pipeline_output.json 2>/dev/null        # Stage-1 done?
```

**Then, per subject that still needs work** (`<tag>` = subjectC/D/E/F, `<TAG>` = REC_SUBJECTC/…):

**(a) If not yet clicked — operator click (USER step).** Launch the click UI on the baseline and
give the user `http://localhost:5050`. MUST use setsid+env.sh (survives tool timeouts; loads CUDA):
```
cd audio_diarization; mkdir -p session/rec_<tag>
IN=/home/user1/Documents/Deception_Detection/deception_detection/pipeline_system_outputs/<TAG>_SRC/spovnob_input
setsid bash -c "cd $(pwd) && source env.sh >/dev/null 2>&1 && exec python click_ui.py '$IN/00_baseline.mp4' --model-store /home/user1/model_store --work-dir session/rec_<tag> --port 5050" </dev/null > session/rec_<tag>/click_ui.log 2>&1 & disown
```
Wait for it to bind 5050 (a long baseline = a few-min prescan) in a SEPARATE background waiter — do
NOT put the wait loop in the same call (a tool timeout would kill the UI). User clicks target's
face + Export → writes `session/rec_<tag>/clicks.json`.

**(b) SPOVNOB Stage-1 (Layers 0-3)** after the click → `pipeline_output.json`:
```
cd audio_diarization
setsid bash -c "cd $(pwd) && source env.sh >/dev/null 2>&1 && \
  mapfile -t V < <(find -L '<abs .../<TAG>_SRC/spovnob_input>' -maxdepth 1 -name '*.mp4' | sort) && \
  exec python pipeline_runner.py --run --videos \"\${V[@]}\" --clicks session/rec_<tag>/clicks.json \
    --work-dir session/rec_<tag> --model-store /home/user1/model_store \
    --manifest session/rec_<tag>.manifest.jsonl --operator \${USER}" \
  </dev/null > session/rec_<tag>/pipeline_runner.log 2>&1 & disown
```

**(c) Stage-2 cascade (GPU) — MUST set LD_LIBRARY_PATH or InsightFace silently drops to CPU:**
```
cd deception_detection
SP=~/anaconda3/envs/spovnob_env/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib|tr '\n' ':')$SP/torch/lib"
M=validation/gt_<tag>/<tag>_manifest.json
# shard the clip bases across N processes (per-clip = best GPU use); each shard = a subset:
setsid env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" $PY validation/multisubject/cascade_generic.py $M <base1> <base2> > pipeline_system_outputs/<TAG>_SRC/cascade_s1.log 2>&1 & disown
# …repeat for more shards. Verify GPU: grep 'Applied providers .*CUDAExecutionProvider' the log; GPU ~50-70%, not ~6%.
```
Clip bases are in the manifest (`00_baseline`, `01_interview`, …). onnxruntime MUST be
`onnxruntime-gpu==1.19.2` (check: `$PY -c "import onnxruntime as o;print(o.__version__)"`; if it's
1.17.1 again, `pip install onnxruntime-gpu==1.19.2`).

**(d) Assemble** (baseline fit + calibrate + global timeline + analyst report):
```
$PY validation/multisubject/assemble_generic.py validation/gt_<tag>/<tag>_manifest.json
```
Sanity: baseline median deviation ≈ √n_features (NOT 0 = degenerate); blink populated.

**(e) When all of C/D/E/F are assembled — score the whole N=6 corpus:**
Build `validation/multisubject/replication_manifest_N6.json` with a block per subject:
```
{"subjects":[
 {"name":"SubjectA","recording_dir":"pipeline_system_outputs/REC_SUBJECTA","elan_dir":"/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"},
 {"name":"SubjectB","recording_dir":"pipeline_system_outputs/REC_SUBJECTB","elan_dir":"pipeline_system_outputs/SUBJECTB_SRC/elan_normalized"},
 {"name":"SubjectC","recording_dir":"pipeline_system_outputs/REC_SUBJECTC","elan_dir":"pipeline_system_outputs/REC_SUBJECTC_SRC/elan_normalized"},
 … D/E/F likewise (elan_dir = pipeline_system_outputs/<TAG>_SRC/elan_normalized) …
]}
```
Then: `$PY -m multisubject.replication_scorecard validation/multisubject/replication_manifest_N6.json --out validation/multisubject`
→ per-channel REPLICATES/SUBJECT-SPECIFIC/NO-SIGNAL across 6 subjects. Update
`validation/multisubject/RESULTS.md` + MASTER_REFERENCE changelog. Optionally rerun coupling per
subject (`validation/gt_subjectB/coupling_evaluate_B.py` as the template).

## 5. Key facts / gotchas (don't relearn these)
- **Scorecard scores WITHIN-CLIP** (pooled AUC fails the SubjectA positive control). Fix already in
  `multisubject/replication_scorecard.py`; regression in `tests/verify_multisubject.py` (28/28).
- **ELAN mapping:** `replication_scorecard` globs `*C{f+1:03d}*.eaf`. Each subject's
  `<TAG>_SRC/elan_normalized/` already has unique `C{f+1:03d}_<tag>.eaf` with title-cased labels
  (built by `validation/multisubject/prep_subject.py`). SubjectA uses its raw `B04C…` dir directly.
- **Clips with no `.eaf`** (SubjectD C003/C004, SubjectE C002) are cascaded but not scored — fine.
- **onnxruntime GPU + LD_LIBRARY_PATH recipe** — see step (c); this is the #1 time-sink if missed.
- **Click UI:** audio browser-cache bug fixed (content-versioned `/audio` URL); hard-refresh the
  browser between subjects. Launch with setsid so a tool timeout can't kill it.
- **Cascade is MediaPipe-CPU-bound** — GPU sits ~half-idle at 3-way; shard per-clip (6-9 way) to
  use the box. All generic drivers are manifest-driven in `validation/multisubject/`.

## 6. Uncommitted work to protect (git, top repo)
Modified: `Documentation/{MASTER_REFERENCE,COUPLING_MODEL_DESIGN,MULTISUBJECT_REPLICATION_PLAN}.md`,
`audio_diarization/click_ui.py`, `deception_detection/multisubject/replication_scorecard.py`,
`deception_detection/tests/verify_multisubject.py`.
New: `deception_detection/validation/{gt_subjectA/COUPLING_RESULTS.md, gt_subjectB/, gt_subjectC/,
gt_subjectD/, gt_subjectE/, gt_subjectF/, multisubject/}`, this file. Plus 2 unpushed
`ffmpeg_ingestion` commits from earlier. (`pipeline_system_outputs/` is gitignored — the cascade
outputs + canonical media live there and are NOT in git; they persist on disk.)
