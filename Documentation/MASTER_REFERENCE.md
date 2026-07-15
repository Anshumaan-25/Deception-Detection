# MASTER REFERENCE — Forensic Deception Detection Mono-repo

> **This is the living master document for the entire project.** It is maintained continuously:
> every meaningful change to the pipeline (code, design decisions, model swaps, validation results)
> gets reflected here, with a dated entry in the [Changelog](#changelog) at the bottom.
> If this document and the code disagree, **the code wins** — and the discrepancy is a bug in this
> document that must be fixed.
>
> **Last synced with code:** 2026-07-10 — graph-line v2 (**predictive cross-modal coupling
> model**, `stgae/coupling_*.py`) was **evaluated on the desktop and FALSIFIED by Bar 4**
> (`coupling_evaluate.py` on `REC_SUBJECTA_SYNCED_*`; write-up
> `validation/gt_subjectA/COUPLING_RESULTS.md`, design outcome `COUPLING_MODEL_DESIGN.md §9`).
> Result: Bar 0 (fit 0.801) and Bar 1 (0.754, a coupling feature that *beats* the marginal path)
> PASS, but Bars 2/3 and the **go/no-go Bar 4 FAIL** (truth flag-rate 93% vs baseline 6% — the
> aggregate coupling-z is still a distance-from-baseline meter). **This is the SECOND
> pre-registered graph formulation killed by the same small-baseline brittleness** (v1
> reconstruction `STGAE_RESULTS.md`, 2026-07-09; v2 coupling here) → the blocker is **N=1, not
> the architecture**; the **graph line is closed for the n=1 era** (revisit only with N>1 + a
> supervised head). **The marginal per-channel z-score attribution remains the shippable method**
> (0.68–0.70). N=1 is the ceiling on every claim — **new annotated subjects (same experiment)
> arriving ~2026-07-11** (§14.5) are now the critical unblock. Also new:
> **analyst report generator** (`report/`, §9) — per-recording self-contained HTML is now the
> analyst-facing end deliverable (tests 22/22; e2e re-verified 375/375); and the
> **multi-subject replication toolchain** (`multisubject/`, tests 19/19) — intake validator +
> pre-registered replication scorecard, ready for tomorrow's corpus. **Desktop runbook for the
> new subjects: `Documentation/MULTISUBJECT_REPLICATION_PLAN.md`.**

---

## 1. Mission & use case

Given recorded interview footage of a single subject, produce a **mathematically grounded,
per-window behavioral deviation analysis** of that subject — localized in time, attributable to
specific feature channels, and calibrated against the subject's *own* neutral behavior.

Core doctrine:

- **Attribution, not classification.** The system never outputs a truth/lie verdict. It outputs
  *where* and *how strongly* behavior deviates from the subject's own baseline, per feature, per
  time window. Interpretation belongs to a human analyst.
- **No ground truth in production, ever.** ELAN-annotated videos serve two offline roles only:
  scoring/validation of the unsupervised pipeline (labels overlaid *after* calibration — the
  2026-07-08 proof-of-signal) and a small corpus for possible future supervised work. Nothing
  in the production path consumes annotations.
- **Per-subject calibration.** Every recording ships with a dedicated baseline clip (the subject
  giving generic information). That clip defines "normal" for that subject — both for today's
  z-score calibration and for the future ST-GAE's reconstruction-error training.
- **Forensic rigor on the audio side.** Speaker identity is established by operator witness (a
  click on the target's face), not by clustering inference. No audio synthesis/separation ever.
  Bit-deterministic, hash-chained audit trail (see §4).

## 2. Production data model

A **recording** = one subject, one session, dropped into the intake as a bucket of clips:

| Clip | Role |
|---|---|
| `file_index 0` | **Dedicated baseline video** — subject gives generic info. Defines "normal". Overridable via `baseline_file_index` in the batch profile. |
| `file_index 1..N` | **Interview videos** — the material under analysis. |

All clips share one global clock: SPOVNOB's `file_offset_ms` (cumulative duration of preceding
files) anchors every clip's local timestamps onto the recording timeline.

Failure policy: an unusable **baseline** clip fails the whole recording
(`BaselineCalibrationError`); a failed **interview** clip is excluded from assembly with a warning
while the recording stays alive.

## 3. Repository map

```
Deception_Detection/
├── audio_diarization/            ← SPOVNOB: forensic speaker diarization (own pinned .venv)
│   ├── layer1_enrollment/        ← visual-anchored enrollment (click UI, face lock, MAR-FSM)
│   └── SPOVNOB_MASTER_REFERENCE.md   ← deep authority for everything audio-side (1886 lines)
├── deception_detection/          ← multimodal behavioral analysis (conda spovnob_env)
│   ├── main_pipeline.py          ← MultimodalProductionOrchestrator: per-clip cascade +
│   │                                recording-level orchestration (THE core file)
│   ├── app/
│   │   ├── batch_daemon.py       ← intake watcher, BatchLedger, BatchOrchestrator, GPU workers
│   │   └── recording_intake.py   ← pairs mp4/wav files to SPOVNOB file_index by stem
│   ├── analytics/
│   │   ├── dynamic_window_engine.py    ← sliding-window aggregation (2 s / 1 s stride)
│   │   ├── confidence_math.py          ← confidence-weighted mean/var/max/min, FFT block
│   │   ├── baseline_calibrator.py      ← fit/apply + BaselineStats (+ legacy calibrate())
│   │   ├── recording_assembler.py      ← global-timeline assembly + deviation_percentile
│   │   ├── context_mapper.py           ← session manifest → question_id / phase context
│   │   ├── temporal_window_generator.py ← training-corpus windowing (ELAN labels; offline only)
│   │   ├── predictive_engine.py        ← DEAD CODE (superseded TFN classifier — see §11)
│   │   └── diagnostic_visualizer.py
│   ├── audio_isolation/core/
│   │   ├── diarization_bridge.py ← SPOVNOB pipeline_output.json → seam contract (§5)
│   │   ├── diarizer_engine.py    ← cross-modal identity anchoring + isolation
│   │   └── acoustic_extractor.py ← WavLMAcousticExtractor (canonical 20-col acoustic schema)
│   ├── ffmpeg_ingestion/         ← canonicalizer (canonical MP4/WAV + _hubert-suffix WAV)
│   ├── opencv_streaming/         ← CanonicalStreamReader (RAM-buffered master-clock frames)
│   ├── mediapipe_pose/           ← ParallelMediaPipePool (12 workers)
│   ├── OpenFace-3.0/ + openface_pipeline/  ← AU + gaze extraction
│   ├── Yolo_v8/                  ← PersonDetector + FaceLock (TensorRT)
│   ├── stgae/                    ← graph line: v1 recon ST-GAE (falsified) + v2 coupling_*.py
│   │                                (predictive cross-modal coupling — COUPLING_MODEL_DESIGN.md)
│   ├── report/                   ← analyst report generator (self-contained HTML end deliverable;
│   │                                analyst_report.py assembly + render_html.py; §9, Pass 5)
│   ├── multisubject/             ← intake_validator.py + replication_scorecard.py (N>1 corpus;
│   │                                MULTISUBJECT_REPLICATION_PLAN.md is the desktop runbook)
│   ├── dashboard/ + frontend/    ← diagnostic dashboard / browser UI
│   ├── tests/                    ← verify_*.py self-test suite (§10)
│   ├── validation/gt_subjectA/   ← ground-truth validation: scorer/attribution/robustness
│   │                                scripts + RESULTS.md + clip06_timeline.html (2026-07-08)
│   ├── SPOVNOB_intake/           ← intake watch folder (incl. SESSION_TEST_MOCK fixture)
│   ├── pipeline_system_outputs/  ← per-session + per-recording outputs (§9; gitignored —
│   │                                incl. GT_SUBJECTA_20260708 outputs + canonical media)
│   └── weights/trt_engines/      ← compiled TensorRT engine cache
├── session/                      ← SPOVNOB batch outputs (batch01, rec_ca, …)
├── my_videos/                    ← real test footage (gitignored): CA Beard / NT / UB clips +
│                                    00SubjectA_session1/ (first ELAN-annotated recording w/
│                                    dedicated baseline — see 2026-07-08 changelog)
├── Documentation/
│   ├── MASTER_REFERENCE.md       ← this document
│   ├── PIPELINE_ARCHITECTURE.md  ← the block diagram (Mermaid) — visual companion to this doc
│   └── (historical docs, diagram exports — see §15)
└── README.md                     ← short public-facing overview
```

## 4. Stage 1 — SPOVNOB speaker diarization (`audio_diarization/`)

**Deep authority:** `audio_diarization/SPOVNOB_MASTER_REFERENCE.md`. Condensed contract here.

Mission: given a batch of clips (one subject, one session) plus **one operator click** on the
target's face in clip 0, emit PTS-timestamped WAV segments containing ONLY the visually-verified
target's speech, overlap-excluded, each SHA-256 hashed, with a hash-chained audit log of every
decision.

Doctrine: identity by witness not inference (ArcFace lock from the operator click); **zero
synthesis** (no source separation, ever — contaminated audio is excluded, not repaired);
bit-identical determinism (CUDA deterministic mode, float32-only, self-tested at every gate run);
append-only hash-chained manifest written before any destructive op.

Layers:

| Layer | Function |
|---|---|
| 0 | PTS-true audio extraction + Silero VAD speech map |
| 1 | Visual-anchored enrollment: click → ArcFace face lock → MAR-FSM speech-window capture → ECAPA-TDNN `E_seed`/`E_composite`/`E_anti` → Triple Gate A/B/C → FREEZE. Alternate **audio-anchored** path for bearded / unreliable-MAR subjects. |
| 2 | Per-session calibrated sliding-window ECAPA cosine scoring → median pooling → HIGH/MEDIUM/SUB/REJECT tiering → edge trim |
| 3 | PyAnnote overlap detection → NaN-void overlapped blocks → bridge gaps < 400 ms → slice + SHA-256 hash final WAVs |

Model stack (frozen, loaded once): Silero VAD (CPU), YOLOv8m, InsightFace buffalo_l
(SCRFD + ArcFace + 2d106det), ECAPA-TDNN, PyAnnote segmentation-3.0.

**Runs as a sealed subprocess** in its own pinned `.venv` (torch 2.1.2+cu121; environment gate
fail-closes on any of 23 exact package-pin mismatches). Never imported in-process by the
deception side.

## 5. Stage 2 — the merge seam (SPOVNOB → deception detection)

**Contract artifact:** `pipeline_output.json` (schema `spovnob-pipeline-output-v1`), containing
`clean_segments[]` — each with `file_index`, local + global start/end ms, `wav_path`,
`wav_sha256` — and per-file `file_offset_ms`.

- `audio_isolation/core/diarization_bridge.py` — `DiarizationBridge` parses that JSON and adapts
  it per clip to the seam contract consumed by `main_pipeline.py`:
  `[("TARGET", start_ms, end_ms), ...]`. Pure stdlib. Also exposes `file_offset_ms(file_index)`
  for global-timeline assembly.
- `app/recording_intake.py` — pairs a recording bucket's mp4/wav files to SPOVNOB `file_index`
  by normalized stem (tolerant of `_canonical` / `_hubert` suffixes — note: `_hubert` is a
  historical **filename convention** for the canonical acoustic-extraction WAV, unrelated to the
  HuBERT model, and deliberately not renamed).

Fallback semantics in `process_video_session`: `pyannote_segments=None` → legacy whole-clip mock
(backward-compatible single-clip runs); `[]` → diarization ran but target is silent →
`TARGET_SILENT`.

## 6. Stage 3 — intake & orchestration (`app/batch_daemon.py`)

`watchdog` filesystem watcher on the intake folder → `BatchLedger` (crash-safe batch state) →
`BatchOrchestrator` → spawns an isolated GPU worker **process** per batch.

Batch profile keys (JSON dropped with the bucket): `session_id`, `offset_ms` (default 0),
`baseline_file_index` (default 0), `session_manifest_path` (optional — feeds ContextMapper),
`elan_annotation_file` (optional — **training-corpus path only**, via
`temporal_window_generator.py`; never used in production inference).

## 7. Stage 4 — per-clip cascade (`main_pipeline.py :: process_video_session`)

Runs identically on the baseline clip and every interview clip. In recording mode it is called
with `calibrate=False` (Phase 4 skipped, manifest stage marked `"skipped"`).

**Phase 1 — Visual & audio extraction.**
`CanonicalStreamReader` streams canonical MP4 frames on the master clock →
double-buffered chunked **YOLOv8** detection (TensorRT `.engine` auto-detected next to the `.pt`,
PyTorch fallback) → **FaceLock** (InsightFace, TRT engine cache) maintains the persistent target
person stream → in parallel: **MediaPipe pose pool** (12 workers: geometry, motion,
lip-motion logs) and **OpenFace 3.0** (AUs, gaze). Then cross-modal identity anchoring:
`MediaPipeAudioDiarizer.anchor_target_identity(pyannote_segments, visual_lip_logs)` correlates
diarized speech with observed lip motion, producing per-frame `is_audio_active`,
`mismatch_incongruence` (audio active, lips still), `silent_incongruence` (lips moving, no audio),
`diarizer_conf`. Isolation then writes the **target-only WAV**, which boots the
**WavLMAcousticExtractor** (`microsoft/wavlm-large`, layer 14 — see §12 for the re-tune debt).
Since 2026-07-07 the extractor is **single-pass**: one chunked+batched full-clip forward (30 s
chunks, 1 s context halo trimmed by absolute center time, fp16 autocast, encoder truncated above
layer 14 — profile tuned for the RTX 6000 Ada production box) caches the layer-14 latent sequence
in RAM; the window-level and frame-level consumers below are then pure numpy over that cache
(previously every 2 s window ran its own WavLM forward — 2× redundant at 1 s stride).

**Phase 2 — Raw feature compilation** (`compile_raw_features`).
Vectorized kinematics: 3D hand-to-face distance (L/R), 3D wrist velocity (L/R), gaze velocity,
AU onset velocity (AU1/2/4/6/9/12/25/26 — genuine expression onsets run ~250–500 ms; posed ones
don't), 7-landmark `macro_motion_energy`, plus `ear` / `is_blinking` merged per-frame from the
pool's lip logs (seam fixed 2026-07-08 — they were previously computed but dropped, leaving every
windowed blink feature NaN). Exact frame-by-frame inner join of pose × OpenFace on
the shared master-clock timestamp, then **frame-level acoustic injection**: the 18-column
`frame_*` block (`audio_isolation/core/frame_alignment.py`) pools WavLM latents into each video
frame's 33.3 ms interval by **absolute-timestamp bucketing** (searchsorted over latent-frame
centers — drift-free by construction, no index-ratio math), masked to NaN wherever
`is_audio_active != 1` or the waveform sits below the attenuation floor. Output: **30 fps fused
CSV**. *(This frame-level CSV — now cross-modal — is the planned input of the future ST-GAE, with
`is_audio_active` as the acoustic-node loss mask — §14.)*

**Phase 3 — Sliding-window aggregation** (`DynamicWindowEngine`).
Parameters: 2000 ms window, 1000 ms stride, `min_fill_rate=0.25` (windows with < 15 of 60
possible frames are nullified), `min_confidence_threshold=0.35`. All aggregation is
**confidence-weighted** (shared math in `analytics/confidence_math.py`). Injects the 20-column
WavLM acoustic block per window, FFT behavioral-periodicity block, and context columns via
`ContextMapper` (question_id / phase from the session manifest). Output: per-clip
**windowed CSV** (uncalibrated in recording mode).

**Phase 4 — legacy self-calibration** (single-clip mode only, `calibrate=True` default):
z-scores the clip against its own first 30 s. Kept strictly for backward compatibility; the
recording path replaces it entirely with §8.

## 8. Stage 5 — recording-level calibration & assembly (`process_recording_session`)

Four passes:

1. **Per-clip cascade** on every clip with `calibrate=False` (§7).
2. **Fit**: `BaselineCalibrator.fit(baseline_windowed_csv)` on the baseline clip — pools **all**
   its windows (no duration cap) → per-feature mean/std (`ddof=1`; zero-variance → NaN std) →
   persisted as `BaselineStats` JSON. Raises `BaselineCalibrationError` (fails the recording) on
   missing file, < 2 windows, or all-NaN features. Metadata columns (`NON_FEATURE_COLUMNS`) are
   never treated as features.
3. **Apply**: `BaselineCalibrator.apply(clip_csv, stats, out_csv)` z-scores **every** clip
   (baseline included) against the fitted baseline stats — never against the clip itself. Adds
   `deviation_magnitude` per window; deliberately does **not** add `deviation_percentile`.
4. **Assemble**: `assemble_recording(clips, out_csv)` rebases each clip's window times by its
   `file_offset_ms`, adds `file_index` + `clip_window_id` provenance, concatenates in file order,
   renumbers `window_id` sequentially, and computes `deviation_percentile` **once over the whole
   recording** (rank-pct over combined `deviation_magnitude` — percentile is only meaningful over
   one shared population). Windows never straddle clip boundaries (windowing is per-clip, so hard
   breaks are automatic).

Returns a dict: `recording_id`, `baseline_file_index`, `baseline_stats_json`,
`recording_calibrated_csv`, `clips[]`.

**Sanity check — the correct expectation (corrected 2026-07-09).** Earlier notes said "the
baseline clip's own deviations land near 0." That is **mathematically wrong** for
`deviation_magnitude = √(Σ z²)` over ~134 features: a baseline window z-scored against its own
clip's stats sits ≈ `√(n_features)` from the centroid (≈ 11.6 for 134 features), **not** 0. The
right invariants are: (a) per-feature z across baseline windows has mean ≈ 0, std ≈ 1
(construction); (b) baseline `deviation_magnitude` median ≈ `√(n_features)`; (c) interview clips
that genuinely deviate sit **above** the baseline median. A baseline median of **0** is the
signature of a **degenerate** run (all features NaN/constant → `√0`), not a healthy one — which
is exactly what the `rec_ca` dry-run turned out to be (see §12 / 2026-07-09 changelog). First
healthy calibration: `REC_SUBJECTA` (baseline |mean z| 0.000, std 1.000, median dev 9.85;
interviews 11–20).

## 9. Output artifacts (`pipeline_system_outputs/`)

Per clip (`<session_id>/`):

| Artifact | Content |
|---|---|
| `metadata.json` | master manifest: per-stage status, model/layer metadata, output paths |
| `<sid>_raw_features_30fps.csv` | Phase 2 fused frame-level features |
| `audio_isolation/isolated_target_audio.wav` | target-only speech |
| `<sid>_windowed_features.csv` | Phase 3 windowed features (uncalibrated in recording mode) |
| `<sid>_calibrated_features.csv` | z-scored vs baseline stats (+ `deviation_magnitude`) |

Per recording (`<recording_id>/`):

| Artifact | Content |
|---|---|
| `<rid>_baseline_stats.json` | fitted `BaselineStats` (means, stds, window count, source CSV) |
| `<rid>_recording_calibrated.csv` | **CURRENT END DELIVERABLE (data)** — all clips on the global timeline, z-scored, with `deviation_magnitude` + recording-wide `deviation_percentile` |
| `<rid>_analyst_report.html` | **CURRENT END DELIVERABLE (analyst-facing)** — self-contained HTML (air-gap safe, zero external requests): data-quality/calibration-health panel, per-clip deviation-percentile strips, direction-aware node table, validated-channel timelines, flagged-window drill-down (≥ p95), conditional coupling lane. Generated by `report/analyst_report.py` (Pass 5 of `process_recording_session`, non-fatal); ELAN overlay only via `--elan-dir` (validation mode, never default) |

## 10. Windowed feature schema (column families)

Metadata: `window_id`, `start_time_ms`, `end_time_ms`, `frame_count`, `cumulative_confidence`,
`emotion_label_mode`; context: `context_phase`, `question_id`, `phase_elapsed_ms`; recording-level
provenance (post-assembly): `file_index`, `clip_window_id`.

Feature families (all confidence-weighted):

- **Kinematics**: `left/right_wrist_velocity_{mean,max}`, `motion_energy_mean`,
  `macro_motion_energy_mean`, `postural_stillness_mean`, hand-face distances
- **Gaze/head**: `gaze_{x,y,z}_mean`, `gaze_velocity_{mean,var}`, gaze entropy,
  `head_{yaw,pitch,roll}_mean`
- **AUs** (AU1/2/4/6/9/12/25/26): `{AU}_mean/max/var` + `{AU}_velocity_{max,mean}` (onset speed),
  composite markers (e.g., Duchenne AU6×AU12), `ear_mean` / blink rate
- **Cross-modal incongruence**: `mismatch_ratio`, `silent_speech_duration_ms`, `diarizer_conf`
- **Acoustic (20 cols, canonical source `acoustic_extractor.ACOUSTIC_COLUMN_NAMES`)**:
  `acoustic_volatility`, `prosodic_velocity`, `wavlm_latent_0..15`, `vocal_entropy`,
  `acoustic_energy_rms`
- **FFT periodicity block** (`confidence_math.FFT_COLUMN_NAMES`)
- **Calibration outputs**: z-scored versions of all of the above + `deviation_magnitude`,
  `deviation_percentile` (recording-level only)

**Frame-level acoustic family — raw 30 fps CSV only, never windowed** (canonical source
`frame_alignment.FRAME_ACOUSTIC_COLUMN_NAMES`, 18 cols): `frame_wavlm_latent_0..15`,
`frame_prosodic_velocity`, `frame_acoustic_energy_rms`. Aligned per video frame by absolute
timestamp bucketing, NaN wherever the target is not verifiably speaking (`is_audio_active` is the
mask column). The `frame_` prefix keeps them disjoint from `DynamicWindowEngine`'s explicit
aggregation list, so the windowed schema is untouched.

## 11. Environments, hardware & model stack

| | audio_diarization | deception_detection |
|---|---|---|
| Env | pinned `.venv` (`env.sh`) | conda `spovnob_env` |
| Torch | 2.1.2+cu121 | 2.5.1+cu121 |
| Guards | environment gate: 23 exact pins, fail-closed; GPU determinism self-test | `transformers==4.40.1` (WavLM verified importable) |
| Isolation | sealed subprocess only | never imports SPOVNOB in-process |

Hardware: NVIDIA RTX 6000 Ada (this dev box). Production target machine is still an open question
(nv05's env is hard-incompatible with SPOVNOB's pins).

Deception-side models: YOLOv8 (TensorRT-first), InsightFace FaceLock (TRT engine cache),
MediaPipe pose (12-worker pool), OpenFace 3.0, WavLM (`microsoft/wavlm-large`, layer 14,
hidden_size-driven latent reshape — 1024/16=64 per latent group).

## 12. Known issues, dead code & validation debt

**Validation debt (remaining):**

1. **`offset_ms` — MEASURED 2026-07-09: an ~80 ms A/V desync exists in the canonical files
   (video LEADS audio by 80 ms), and it is NOT yet compensated.** Root cause: the SubjectA
   MPEG-2 source is open-GOP — its video stream `start_time` is 0.080 s (two leading
   AV_PKT_FLAG_DISCARD B-frames) while audio `start_time` is 0.000 s, so there is 80 ms of audio
   before the first video frame. The canonicalizer's ffmpeg drops the video's 80 ms lead
   (rebases video to 0) but keeps audio at 0 → canonical-video-time T corresponds to
   canonical-audio-time T−80 ms. Confirmed two ways: exact from PTS (`v.start_time − a.start_time
   = 80 ms`, consistent across all 8 clips) and behaviorally (mouth-open ⟶ acoustic-energy
   cross-correlation peaks at ~−100 ms = 80 ms artifact + ~20 ms physiology). Impact: **negligible
   for the 2 s / 1 s window-level work** (ELAN AUCs, is_audio_active) — 80 ms is 2.4 frames; but
   **material for the frame-level ST-GAE congruence node**, which must not learn an artificial
   cross-modal lag. Note `process_recording_session`'s `offset_ms` only shifts diarization
   *segments* (is_audio_active), NOT the frame-level WavLM↔video pooling in `frame_alignment.py`
   — so the fix belongs in the **canonicalizer** (align both streams to a common zero, e.g. trim
   the pre-video audio: `atrim=start=(v.start−a.start)`) so all downstream consumers get synced
   data. **FIXED 2026-07-09** — `canonicalizer.py` now probes per-stream start_time and trims the
   audio front by the skew; verified end-to-end (mouth↔energy cross-correlation lag −100 ms →
   **0 ms** on re-cascaded `REC_SUBJECTA_SYNCED`). The recording-run also passes `offset_ms=80`
   to `segments_for` so the diarization is_audio_active mask lands on the rebased frames. (The
   original REC_SUBJECTA window-level deliverable still carries the 80 ms — negligible there.)
2. `WAVLM_LAYER_INDEX = 14` is a proportional-depth placeholder (HuBERT-base layer 7/12 scaled to
   24 layers), not re-tuned on real audio. Same unblock as (1).
3. ~~No real end-to-end run through the GPU stack.~~ **Per-clip path resolved 2026-07-07**: the
   per-clip cascade (`process_video_session`) runs **end-to-end on real footage on the GPU**
   (CA-Beard clip, all four phases, valid output CSVs — see changelog). **The recording-level path
   is now also validated on real footage**: a full 4-clip `process_recording_session` (`rec_ca`,
   488.7 s) completed — 351 windows assembled on a strictly-monotonic global timeline (each clip
   rebased by its `file_offset_ms`), baseline fitted on 97 windows, baseline clip's own median
   deviation ≈ 0 (the §8 sanity check), `deviation_percentile` recording-wide. **Remaining:**
   `offset_ms` is still 0/unmeasured and `WAVLM_LAYER_INDEX` unre-tuned — both need a recording that
   has a *real dedicated baseline* clip (rec_ca's clip 0 is just an interview clip treated as
   baseline; fine for wiring validation, not for a forensic result). **That recording now exists**
   (`00SubjectA_session1`, received 2026-07-08, already proof-of-signal-validated per-clip); the
   SPOVNOB production pass on it is the next-session task.

**Bugs found by the first real GPU runs 2026-07-07 (now fixed):**

- **[FIXED 2026-07-07] Multi-clip recording hung on the 2nd clip.** `process_recording_session` runs clip 0
  (BASELINE, 2888 frames) to completion, then **hangs at the start of clip 1's ingestion** (0 %
  GPU, one core busy then idle, frozen indefinitely). faulthandler stack of the hang: the **main
  thread is blocked in `ParallelMediaPipePool.submit_task → task_queue.put()`** (the task pipe is
  full) while the **result-drainer thread is blocked in `result_queue.get()`** (empty). Diagnosis:
  the 12 MediaPipe worker processes **stop consuming `task_queue` after clip 0 finishes**, so on
  clip 1 the task pipe fills and the producer blocks forever. Root cause is worker lifecycle/state
  across the clip boundary in the *reused* orchestrator + pool (the pool persists across all clips;
  workers are `daemon` procs). **The per-clip path is unaffected** — clip 0 always succeeds, and a
  single `process_video_session` on any clip succeeds. **Reproduces deterministically** by running
  two ≥30 s clips through one orchestrator (clip 0 OK, clip 1 hangs ~90 s in). Distinct from — and
  downstream of — the pipe-buffer *deadlock* fixed the same day: that was the *result* queue
  overflowing; this is the *task* queue starving because the workers go quiet. **Fix:** the
  MediaPipe pool is now **rebuilt per clip** — `ParallelMediaPipePool.restart_workers()` (fresh
  queues + 12 fresh worker processes + a fresh drainer thread) is called at the top of every
  `process_video_session`, beside the existing per-session FaceLock reset. Verified on the 2-clip
  repro: clip 0 (69.9 s) **and** clip 1 (60.3 s) both complete, no hang (clip 1 previously hung
  forever). Cost ≈ 12 `fork()`s (~1 s) per clip. Why the reused workers went silent after clip 0
  wasn't pinned down (the worker loop is exception-safe, so likely queue/pipe-state corruption at
  the boundary); rebuilding sidesteps it.
4. The 2026-07-07 single-pass WavLM rewrite (chunked full-clip forward, fp16 autocast, encoder
   truncation, whole-clip normalization, ~30 s transformer context instead of per-2 s-window
   forwards) **has now executed on GPU across every 2026-07-07/08 run** (rec_ca ×4 clips +
   SubjectA ×8 clips, frame block correctly NaN-masked). Still pending: the one-time A/B of
   `use_amp` / `truncate_encoder` against fp32 full-stack (each independently toggleable via
   `WavLMAcousticExtractor` kwargs / module constants).
   An adversarial multi-agent review (2026-07-07, same day) found and fixed 8 real bugs in this
   rewrite before it ever reached hardware — see the dedicated changelog entry below. All 8 now
   have regression tests; none were hypothetical.

**Dead code / warts:**

- `analytics/predictive_engine.py` — superseded TFN classifier (decision 2026-07-06:
  attribution-not-classification; ST-GAE replaces it). **Archived in place 2026-07-06**: entire
  file commented out under an ARCHIVED/SUPERSEDED header, kept (not deleted) as a record of the
  approach. Was never imported by anything (only referenced in `generate_manual_docx.py` prose).
- ~~`tests/verify_confidence_fusion.py` NameError~~ — **fixed 2026-07-07** (half-applied rename:
  `motion_energy` declared, `macro_motion_energy` used).
- `deception_detection/.venv/bin/pip` — broken symlink to the pre-restructure path
  (`~/Documents/Audio_Diarization/...`), stale since the mono-repo rename.
- `openface_pipeline/detectors/{face_detector,landmark_detector}.py` — **clean-room
  reconstructions (2026-07-07)**: the desktop originals were lost in the July transfer incident
  (they exist in no git history, no local copy). Rebuilt against `api/extractor.py`'s call
  contract, decode/alignment flows ported from OpenFace-3.0's own `demo2.py`/`demo.py`/
  `STAR/demo.py`. Compile-verified only — never run on GPU. Recover the desktop originals if
  possible and diff. Also: `unified_detector.py`'s top-level copy carries two relocation path
  patches; `weights/trt_engines/MLT.engine` must be compiled on the production GPU box
  (`tools/compile_pipeline_trt.py`) before OpenFace inference can run.
- Desktop-only artifacts still unrecovered: `session/batch01/` real diarization fixture
  (verify_diarization_bridge check 10 skips without it) and the audio-side model store
  (`/home/user1/model_store` on the Ubuntu box) + `wheelhouse/`.
- Legacy `calibrate()` z-scores `question_id`/`phase_elapsed_ms` (metadata) — pre-existing wart,
  kept for parity; the new fit/apply path excludes them correctly.
- `tools/generate_manual_docx.py` prose still says HuBERT (cosmetic).

## 13. Verification suite (`deception_detection/tests/`)

All pure pandas/numpy on synthetic data — no GPU, no real footage. Run from
`deception_detection/` with `python tests/verify_<name>.py`.

| Script | Covers | Status (2026-07-07 run, laptop, pandas 3.0.2) |
|---|---|---|
| `verify_diarization_bridge.py` | SPOVNOB JSON → seam contract (incl. real `session/batch01` fixture) | ✅ (check 10 skips — batch01 fixture is desktop-only) |
| `verify_merge_seam.py` | seam semantics in `process_video_session` | ✅ |
| `verify_recording_intake.py` | mp4/wav ↔ file_index pairing | ✅ |
| `verify_end_to_end_pipeline.py` | full mocked cascade | ✅ 375/375 |
| `verify_behavioral_periodicity.py` | FFT block | ✅ 80/80 |
| `verify_recording_calibration.py` | fit/apply/BaselineStats/assembly (11 checks) | ✅ 11/11 |
| `verify_confidence_fusion.py` | confidence-weighted math | ✅ 25/25 (NameError fixed 2026-07-07) |
| `verify_frame_acoustics.py` | frame-level WavLM↔30fps alignment + window-formula parity + 8 bug regressions | ✅ 30/30 |
| `verify_wavlm_truncation.py` | StableLayerNorm encoder-truncation correctness (tiny synthetic model, no GPU) | ✅ 9/9 |
| `verify_acoustic_gating.py` | window-level acoustic block gated by `is_audio_active` | ✅ 4/4 |
| `verify_stgae.py` | ST-GAE graph_spec / masking / feature-count-norm loss / zero-grad masking / determinism / noise-failure (CPU torch) | ✅ 16/16 |
| `verify_coupling.py` | coupling model (v2): mask isolation (bitwise), vectorized≡sequential 11-target pass, ÷F_n target loss, target-validity zero-grad, planted-coupling recovery + break-spike specificity + domain-shift robustness (simulated v1 failure), noise→degenerate gate (CPU torch) | ✅ 21/21 (2026-07-10) |
| `verify_report.py` | analyst report: assembly integrity (p95 flag rule, 11-group node table), dead/uncalibratable channel surfacing, degenerate-baseline alert, **baseline index recovered from stats (not assumed 0)**, coupling-lane conditionality, ELAN strictly validation-mode, self-contained HTML (no external URLs, no NaN in JSON) | ✅ 24/24 (2026-07-10) |
| `verify_multisubject.py` | intake validator (failure classes, WARN vs FAIL, verdict JSON) + replication scorecard (5 verdict classes on a planted 3-subject world; noise-sign pigeonhole guarded; adequacy floor; **non-zero baseline recovery**) + **run_replication driver** (intake→score chaining, FAIL gates scoring) + **GLOBAL-timestamp rebase & WITHIN-CLIP scoring regression** (Simpson's-paradox fixture: pooled inverts to 0.17, within-clip recovers 0.66) | ✅ 28/28 (2026-07-10) |

## 14. Roadmap (future, in intended order — nothing scheduled)

1. ~~SPOVNOB production pass~~ + ~~ST-GAE end-stage~~ — **both done 2026-07-09** (see changelog).
   Production run validated (`REC_SUBJECTA`), and the reconstruction-ST-GAE was implemented,
   evaluated, and **falsified by its own pre-registered Bar 4** (`validation/gt_subjectA/
   STGAE_RESULTS.md`; `ST_GAE_DESIGN.md` §10). The **marginal per-channel z-score attribution is
   the shippable method** (within-06 AUCs 0.68–0.70, direction-aware).
2. **Remaining validation debt:** re-tune `WAVLM_LAYER_INDEX` empirically; the one-time WavLM
   `use_amp`/`truncate_encoder` fp32 A/B (§12.4).
3. ~~**Graph attribution v2 — predictive coupling model.**~~ **BUILT + EVALUATED + FALSIFIED
   2026-07-10** (`validation/gt_subjectA/COUPLING_RESULTS.md`; `COUPLING_MODEL_DESIGN.md §9`).
   Masked-node prediction with per-feature residuals + within-clip scoring. Outcome: Bar 0
   (0.801) and Bar 1 (0.754 — a coupling feature above the marginal path) PASS, but the go/no-go
   **Bar 4 FAILS** (truth flag-rate 93% vs baseline 6%) along with Bars 2/3. Per the
   pre-registered decision rule (fail Bar 1 **or** 4 → discard), **v2 is discarded and the graph
   line is CLOSED for the n=1 era.** Two pre-registered graph objectives (v1 reconstruction, v2
   coupling) both fell to the same small-baseline brittleness → the blocker is **N=1**. The
   coupling substrate (`coupling_{model,fit,attribute}.py`, tests 21/21) is retained for a future
   **supervised** graph head once N>1 subjects exist.
4. **VideoMAE v2** (deferred, no re-entry): its criterion presumed a *working* ST-GAE, which the
   2026-07-09 evaluation did not yield.
5. **Supervised training path**: needs N>1 annotated subjects; the ELAN corpus +
   `temporal_window_generator.py` exist for it.
   **⏳→ N=2 REACHED (2026-07-10): SubjectB processed; SubjectA's signal does NOT replicate**
   (`validation/multisubject/RESULTS.md`). Production cascade on SubjectB (7 clips) + within-clip
   replication scorecard: **0 REPLICATES / 7 SUBJECT-SPECIFIC / 127 NO-SIGNAL**. SubjectA's AU12
   lip-tremor family (0.60) and silent-speech incongruence (0.68) are A-specific; **SubjectB leaks
   through blink_rate (0.71), null in A**. The per-channel deception signal is **idiosyncratic
   (per-subject)** at N=2 — which *vindicates* per-subject calibration + attribution and *cautions
   against* any fixed cross-subject channel weighting. **Consequence for the supervised head: do
   NOT start it — at N=2 there is no robust cross-subject channel to learn; it would need many
   more subjects and likely per-subject channel profiles.** 4 more sessions incoming will test the
   subject-specificity hypothesis. (Coupling 4-bar re-eval per subject still pending — lower value
   now the marginal signal itself doesn't transfer.)
   **TOOLING READY (2026-07-10):** `multisubject/intake_validator.py` (gate each package before
   GPU time) + `multisubject/replication_scorecard.py` (pre-registered REPLICATES/SUBJECT-SPECIFIC
   verdicts). **Full desktop runbook + pre-registered criteria + N>1 future-work specs
   (supervised head §6.1, per-subject coupling §6.2, baseline-duration study §6.3):
   `Documentation/MULTISUBJECT_REPLICATION_PLAN.md`.**
6. **Production deployment target**: resolve dev-box vs nv05 (SPOVNOB pin incompatibility).

## 15. Document governance

| Document | Role |
|---|---|
| **`Documentation/MASTER_REFERENCE.md`** (this) | Living master — always current; updated with every change |
| `Documentation/PIPELINE_ARCHITECTURE.md` | The block diagram (Mermaid) — visual companion, kept in sync |
| `Documentation/ST_GAE_DESIGN.md` | **Frozen** — v1 graph design + §10 falsification record |
| `Documentation/COUPLING_MODEL_DESIGN.md` | **Frozen** — graph-line v2 design + pre-registered bars + §9 falsification record (2026-07-10; full evidence `validation/gt_subjectA/COUPLING_RESULTS.md`) |
| `Documentation/MULTISUBJECT_REPLICATION_PLAN.md` | Multi-subject runbook + pre-registered replication criteria + N>1 future-work specs — **the desktop handoff for the new corpus** |
| `audio_diarization/SPOVNOB_MASTER_REFERENCE.md` | Deep authority for the audio side |
| `deception_detection/RECORDING_TIMELINE_AND_ACOUSTIC_UPGRADE_PLAN.md` | **Historical** — completed plan (Phases A+B, done 2026-07-02) |
| `deception_detection/MERGE_INTEGRATION_PLAN.md` | **Historical** — merge plan; known-stale vs code even before completion |
| `Documentation/Audio_Diarization.md`, `Documentation/SPOVNOB_Master_Multimodal_Pipeline_premerge.md`, `Documentation/issues_and_changes_diarization_analysis.md` | **Historical** — pre-merge reference docs restored 2026-07-07 from the SPOVNOB source tree |
| `README.md` | Short public-facing overview |

Update protocol: when code changes, update the affected section(s) **and** add a dated Changelog
line. Completed plan docs are frozen as history, never edited retroactively.

## Changelog

- **2026-07-06** — Document created (commit baseline `4433b4c`). State captured: SPOVNOB complete;
  merge seam complete; WavLM swap (Phase B) and baseline-clip calibration + recording assembly
  (Phase A) implemented and unit-verified, committed in `92adbd3`; prototype block diagram replaced
  by `PIPELINE_ARCHITECTURE.md`; ST-GAE decided as future end-stage (attribution over
  classification), TFN superseded; VideoMAE v2 deferred; real-footage validation on hold.
- **2026-07-06** — `analytics/predictive_engine.py` archived in place (fully commented out with a
  superseded-approach header, per user instruction — not deleted).
- **2026-07-07** — **Laptop restore of all gitignored sub-repos and runtime assets** from the
  intact pre-merge originals in `~/Documents/SPOVNOB/` (copy, originals untouched): `mediapipe_pose/`,
  `opencv_streaming/`, `ffmpeg_ingestion/` (each with its standalone `.git`), `OpenFace-3.0/`
  (incl. `.git` — sole copy of the `extraction-api` branch — and all weights; 177 MB of test
  scratch excluded), `weights/{yolov8n,yolov8m}.pt`, `Yolo_v8/PersonTracking4/weights/` (buffalo_l
  ONNX pack, 341 MB), `SPOVNOB_intake/SESSION_TEST_MOCK` fixture (sha256-verified against its
  generation log), three historical docs into `Documentation/`. All copies MD5/magic-verified;
  every `main_pipeline.py`/`batch_daemon.py` project import now resolves on disk.
- **2026-07-07** — **`openface_pipeline/` materialized as a top-level package** (the layout §3
  documents): recovered `api/extractor.py` + `detectors/unified_detector.py` (newest surviving
  working-tree versions; unified_detector got two relocation path patches), **reconstructed**
  `detectors/face_detector.py` + `detectors/landmark_detector.py` (desktop originals lost — see
  §12), new `__init__.py` sys.path bootstrap, RetinaFace + STAR weights copied to
  `openface_pipeline/weights/`. `.gitignore`: added `deception_detection/openface_pipeline/weights/`.
  Committed in `1734661`.
- **2026-07-07** — Test suite fully green (7/7, incl. pandas 3.0.2 compatibility): fixed
  `verify_confidence_fusion.py` NameError (half-applied rename) and `verify_diarization_bridge.py`
  macOS `/var`→`/private/var` tempdir-resolution assert. Committed in `5d86218`.
- **2026-07-07** — `MASTER_REFERENCE.md` + `PIPELINE_ARCHITECTURE.md` moved into `Documentation/`
  alongside the Br-Clip-Flow diagram exports (user reorganization); §3 map and §15 table updated
  to match. Committed in `5d86218`.
- **2026-07-07** — **Frame-level WavLM alignment (ST-GAE prerequisite) + single-pass rewrite.**
  New `audio_isolation/core/frame_alignment.py` (pure-numpy, unit-testable): absolute-timestamp
  bucketing of WavLM latents onto the 30 fps master clock (drift-free vs index-ratio math — a
  29.97-treated-as-30 assumption would be seconds off at 1 h) + the window-formula math.
  `acoustic_extractor.py` rewritten single-pass: one chunked/batched full-clip forward (30 s
  chunks, 1 s halo, fp16 autocast, encoder truncated above layer 14, whole-clip normalization,
  process-wide model singleton) caches layer-14 latents; window path (unchanged 20-col contract)
  and new frame path both read the cache. `compile_raw_features` now injects the 18-column
  `frame_*` block masked by `is_audio_active`. Hardware profile tuned for the production desktop
  (RTX 6000 Ada 48 GB, 44 cores, 512 GB RAM, Ubuntu, R580/CUDA 12.x). New
  `tests/verify_frame_acoustics.py` (20 checks); full suite 8/8. GPU execution still pending
  (§12.4). Committed together with the review fixes below in `8baea6e`.
- **2026-07-07** — **Adversarial multi-agent review of the single-pass WavLM rewrite — 8 bugs
  found, verified, and fixed same-day, before any GPU run.** Four independent lenses (alignment
  math, GPU-inference correctness, pipeline wiring, test adequacy), each finding independently
  cross-examined by a second agent trying to refute it:
  1. **CRITICAL** — encoder truncation silently corrupted every production feature: wavlm-large's
     StableLayerNorm encoder applies one unconditional final LayerNorm to whatever hidden_states
     entry ends up last, and truncating to exactly `WAVLM_LAYER_INDEX` layers made that the exact
     entry read (`out.hidden_states[WAVLM_LAYER_INDEX]`) — an extra norm on 100% of forward passes,
     not fp16 noise. **Fix:** keep `WAVLM_LAYER_INDEX + 1` layers instead (`_load_wavlm`); the norm
     now lands one entry past the one read. Proven bitwise-exact on a tiny synthetic StableLayerNorm
     model (`tests/verify_wavlm_truncation.py`, no GPU/download needed).
  2. **CRITICAL** — a single non-finite (inf/NaN) value — a realistic fp16-autocast overflow —
     poisoned every subsequent window forever via `cs[hi]-cs[lo]` cumsum cancellation, silently
     indistinguishable from legitimate silence-masking. **Fix:** `pool_latents_to_intervals` /
     `interval_rms` now exclude non-finite rows/samples from the running sum and null only the
     interval(s) that actually trap one.
  3. **MAJOR** — the pre-existing 20-column window-level acoustic block was never gated by
     `is_audio_active` (only by the isolated WAV's own RMS floor, which the diarizer's
     attenuation-not-zeroing leaves blind to who is speaking) — a loud interviewer segment could
     leak into the target's acoustic features. **Fix:** gated in both `dynamic_window_engine.py`
     and `temporal_window_generator.py` on confidence-weighted `is_audio_active` ≥ 0.5 (also skips
     the WavLM cache lookup entirely when not speaking).
  4. **MAJOR** — chunk/overlap seam stitching silently assumed `chunk_seconds`/
     `chunk_overlap_seconds` divide evenly into the 20 ms WavLM hop; a non-conforming retune would
     desync every seam with no error. **Fix:** `validate_chunk_alignment()` (standalone, unit-tested)
     fails fast in `__init__` instead.
  5–8 (**minor**): `frame_acoustic_energy_rms` not nulled when a frame trapped zero latents (only
     the RMS-floor path was masked); zero-norm-row cosine divergence from the original
     `F.normalize`-eps semantics (both cosine helpers now eps-clamp to match); empty-latents
     `pool_latents_to_intervals` returned width-1 instead of the documented `[K, H]`.
  All 8 fixed with dedicated regression coverage: `verify_frame_acoustics.py` grew from 20 to 30
  checks, plus two new files (`verify_wavlm_truncation.py`, `verify_acoustic_gating.py`). Full
  suite: **10/10 files, all green.** Committed in `8baea6e`.
- **2026-07-07** — **FIRST successful end-to-end GPU run of the Stage-2 deception cascade on real
  footage** (`process_video_session`, CA-Beard clip, all four phases → valid `raw_features_30fps`
  / `windowed` / `calibrated` CSVs; gaze unit-vectors, frame-level WavLM NaN-masked exactly where
  `is_audio_active`=0, `deviation_magnitude` populated). Getting there required unblocking OpenFace
  and clearing a chain of integration/env debt — none of it hypothetical, all found by actually
  running the stack (working tree; not yet committed):
  - **OpenFace → PyTorch-native (Strategy B, user-chosen over compiling TensorRT).** The
    reconstructed `openface_pipeline` was TensorRT-*only* and not wired to loadable weights. Fixes,
    all validated on a real frame (face conf 0.999, unit gaze, finite AUs; MLT checkpoint an exact
    394/394 match, loaded strict): `unified_detector.py` — lazy `import tensorrt` + a real PyTorch
    MLT inference path (mirrors `OpenFace-3.0/demo2.py`; AUs raw, no sigmoid); `extractor.py` —
    resolve the MLT weight by glob (`stage2_epoch_*.pth`) and fix two float32→float64 upcasts
    (`img - (104,117,123)` and the ImageNet normalize) that produced `DoubleTensor`s;
    `main_pipeline.py` — point `legacy_weights_dir` at `OpenFace-3.0/weights`; `face_detector.py` —
    disable RetinaFace's redundant ImageNet-backbone pretrain (a relative `./weights/…tar` that only
    resolved with CWD=OpenFace-3.0); `landmark_detector.py` — STAR `device_id` `None`→`0`.
    The reconstructed detectors are now **runtime-validated**, not just compile-checked (§12).
  - **Canonicalizer** (`ffmpeg_ingestion/core/*`): `-fps_mode cfr` → `-vsync cfr` (this box ships
    ffmpeg 4.4; `-fps_mode` needs ≥5.0).
  - **YOLO** (`Yolo_v8/.../detector.py`): passed a *list of torch Tensors* to
    `ultralytics.track()`, which ≥8.1 rejects — pass the numpy frame list instead.
  - **InsightFace/FaceLock on GPU**: env had onnxruntime-gpu **1.15.1** (needs cuDNN 8) but cuDNN
    **9.1** → CUDA EP wouldn't load, everything ran on CPU (~15× realtime). Upgraded to
    **onnxruntime-gpu 1.19.2** (CUDA 12 + cuDNN 9; CUDA EP binds once torch is imported first, which
    loads the CUDA-12 libs `RTLD_GLOBAL`). Also removed the `TensorRTExecutionProvider` from
    `face_lock.py`'s provider list — InsightFace's loader falls back to **CPU-only** the instant any
    requested provider errors, so listing an unavailable TRT EP silently forced CPU.
  - **CUDA stream serialization** (`main_pipeline.py`, `detector.py`, `extractor.py`): running YOLO
    (torch, in a `ThreadPoolExecutor`) concurrently with InsightFace (onnxruntime CUDA EP, main
    thread) tripped "operation not permitted when stream is capturing" (ultralytics' profiler calls
    `torch.cuda.synchronize()`). Removed the three custom `torch.cuda.Stream` contexts and made the
    per-chunk YOLO→FaceLock→MLT path **serial** (also keeps ByteTrack IDs strictly in frame order).
  - **MediaPipe pool deadlock** (`mediapipe_pose/parallel_pool.py`): the 12-worker pool wrote
    results to a `multiprocessing.SimpleQueue` that was only drained *after* the whole ingestion
    loop. Its backing OS pipe (~64 KB) blocks the producer once full, so after a few hundred frames
    the workers block on `result_queue.put()`, stop consuming tasks, and the main thread then blocks
    on `task_queue.put()` — a hard deadlock (0 % CPU/GPU, memory held). Invisible on the 90-frame
    smoke clip (fits the pipe), fatal on a real 2888-frame clip. **Fix:** a daemon drainer thread
    empties `result_queue` continuously into a lock-guarded dict; `collect_results()` waits for the
    count. Verified on the 600-frame clip that previously hung → completes in 55.7 s (~10 fps,
    ~3× realtime).
  - **Env, `spovnob_env`**: installed `timm`, `ultralytics`, `lapx` (ByteTrack), `py-spy`; cached
    `microsoft/wavlm-large` (was absent — only `wavlm-base-plus`); **numpy re-pinned 1.26.4** after
    ultralytics silently pulled numpy 2.2.6 (broke onnxruntime's C-ABI) and opencv 5.0 (runs on
    1.26 despite a cosmetic `numpy>=2` metadata nag). Throughput on the RTX 6000 Ada ≈ **5× realtime**
    on a target-only clip (InsightFace runs all 5 buffalo_l models per frame; batching is a future
    perf item, not a correctness one). Also corrected stale §12 notes: `session/batch01`,
    `/home/user1/model_store`, and the SPOVNOB stack are all **present** on this box.
- **2026-07-07** — **Recording-level (`process_recording_session`) run attempted on real footage
  (`session/rec_ca`, the 4 CA-Beard clips), all 4 canonicalized + paired via `collect_recording_clips`
  ✓.** Outcome: **clip 0 (BASELINE) processed to completion** (per-clip cascade with `calibrate=False`,
  4 diarized target segments applied, Phase 4 correctly deferred, 97 windows) — but the run then
  **hangs on clip 1** (new open bug, §12: MediaPipe workers stop consuming `task_queue` after the
  first clip; main blocks on `submit_task`). So the recording-level fit/apply/assemble is **not yet
  validated on real footage**, and `<rid>_recording_calibrated.csv` was not produced. During the
  MediaPipe *deadlock* investigation the pool's result-drainer was added and verified on a single
  600-frame clip (55.7 s); the *multi-clip* task-queue hang is separate and remains open. **State at
  end of session:** all fixes are working-tree only (uncommitted); temporary run scripts used for
  bring-up (`_smoke_run.py` etc.) were removed. **Resume next session at the §12 "Open bugs" entry.**
- **2026-07-07** — **Multi-clip hang FIXED** (§12): `ParallelMediaPipePool.restart_workers()`
  rebuilds the pool (fresh queues + workers + drainer) at the top of every `process_video_session`,
  so each clip gets a healthy pool instead of reusing one that goes silent after clip 0. Verified on
  the 2-clip repro (clip 0 69.9 s + clip 1 60.3 s, both complete). **Full 4-clip `rec_ca`
  `process_recording_session` then completed on real footage (488.7 s, 4/4 clips): 351 windows on a
  monotonic global timeline (rebased by `file_offset_ms`), baseline fitted on 97 windows, baseline
  clip median deviation ≈ 0 (§8 sanity check), recording-wide `deviation_percentile` — recording-
  level path validated.** Observation for a later pass: clip 2 (file_index 2, only 1 target segment)
  shows much larger deviations (median 15.3, max 64.8) than the others (median ≈ 0) — plausibly a
  real attribution signal, but worth confirming a few features aren't dominating the aggregate. First commit `e709b72` (top repo) landed the per-clip
  fixes + docs; the deadlock/hang pool fixes (`parallel_pool.py`) and canonicalizer live in nested
  repos (`mediapipe_pose/`, `ffmpeg_ingestion/`) and still need pushing.
- **2026-07-08** — **First ground-truth-annotated recording received + validation run designed**
  (`my_videos/00SubjectA_session1/`): `00_baseline.mp4` + `01..07_interview.mp4`, one subject/one
  session, with ELAN `.eaf` annotations (`annotated Videos anushree/`, labels **Truth/Lie/Neutral**,
  ms-precision, single tier). **Video↔annotation mapping verified** (user-confirmed sequential
  rename `B04C001→00 … B04C008→07`; the naive same-index mapping FAILS the annotation-range ⊆
  video-duration check — `01_interview`=B04C002 has no `.eaf`). Baseline is itself annotated as one
  `Neutral` block ✓. Richest clip: `06_interview` = Truth×3 (3.7–64.9 s) → Lie×8 (88.2–309.4 s),
  within-clip Truth/Lie contrast. **Methodology locked: labels are scoring/validation ONLY — never
  training or calibration** (N=1 subject; calibration stays unsupervised on `00_baseline`, ELAN
  labels overlaid post-hoc into the reserved `target_ground_truth` column to measure whether
  `deviation_magnitude` separates Lie from Truth/Neutral: pooled + within-clip rank AUC).
  Run design: ELAN scoring is per-clip-local, so the proof-of-signal path needs **no SPOVNOB pass**
  (per-clip cascade `calibrate=False` → `BaselineCalibrator.fit(00)/apply` → scorer); the full
  SPOVNOB recording run stays as a separate production-path validation. Env note: canonicalization
  must use the **system** ffmpeg 4.4.2 (`/usr/bin/ffmpeg`, has NVENC) — `spovnob_env`'s conda ffmpeg
  4.2.2 is built **without NVENC** and fails `h264_nvenc`; also the project `.venv` python
  autoactivates over conda, so Stage-2 runs pin the absolute
  `~/anaconda3/envs/spovnob_env/bin/python`. All 8 clips canonicalized (30 fps CFR verified);
  cascade parallelized 3-way across processes (disjoint session dirs) to load the RTX 6000.
- **2026-07-08** — **PROOF-OF-SIGNAL: first quantitative validation against human deception
  labels — full write-up in `deception_detection/validation/gt_subjectA/RESULTS.md`.** All 8
  clips cascaded (3-way parallel, ~50 min GPU); calibration fit on `00_baseline` (106 windows /
  134 features); 509 Lie + 140 Truth pure windows scored. Headline: **the scalar
  `deviation_magnitude` (L2) carries no usable signal (Lie-vs-Truth AUC 0.384, inverted)** —
  measured cause: top-5 features carry median 42% (p90 73%) of Σz², dominated by the body
  `motion_energy` family which runs *higher during Truth* for this subject (freeze-during-lies),
  so opposing channel directions cancel/invert any scalar. **Individual channels DO separate
  within-clip** (clip 06, confound-controlled): AU12 velocity dynamics 0.66–0.68, hand↔face
  distance 0.68, wrist velocity 0.63–0.66, head-pitch tremor 0.63, AU4_var 0.62; gaze
  variability *inverts* (0.16–0.35, gaze freezes during lies). Consequences: (1) empirical
  mandate for the ST-GAE per-channel **attribution** end-stage — measured, no longer just
  doctrine; (2) `deviation_magnitude` is bookkeeping, not signal; (3) aggregates must be
  direction-aware + per-subject. Caveats: N=1 subject, overlapping windows, 134 comparisons —
  magnitudes provisional. Artifacts: `pipeline_system_outputs/GT_SUBJECTA_20260708/`
  (gitignored: outputs + canonical media); committable scripts in `validation/gt_subjectA/`.
  `.gitignore` += `my_videos/` (4 GB source footage must never stage). Remaining next session:
  full SPOVNOB production-path run on this recording; ST-GAE design.
- **2026-07-08** — **Overlap-robustness check passed + visual proof-of-signal.** Within-06
  AUCs rerun on strictly non-overlapping windows (95 Lie / 26 Truth): signal holds or sharpens
  (AU12 tremor 0.681→0.692, hand↔face 0.679→0.687, gaze-x inverse 0.162→0.138; motion-energy
  null control stays at chance). New in `validation/gt_subjectA/`: `gt_robustness.py` (the
  check) and `clip06_timeline.html` (six channel traces vs the annotated Truth/Lie bands —
  first mock of the per-clip ST-GAE attribution report). RESULTS.md updated with both.
- **2026-07-08** — **Blink/EAR channel was dead pipeline-wide — found by output audit, fixed,
  smoke-verified.** Audit of all 8 GT-run windowed CSVs: `blink_count`/`blink_rate`/`ear_mean`/
  `ear_var` 100 % NaN (and in rec_ca: only nullification-branch zeros). Root cause: the
  MediaPipe pool computes EAR/is_blinking into its `lip_logs` stream, but the merge seam
  (`main_pipeline.py` zip loop) only copied `is_moving` — `ear`/`is_blinking` never reached
  `pose_data`, so the frame CSV had no such columns and the window engines' `if "is_blinking"
  in window_df` guard silently skipped. Fixes: seam now copies both into `pose_rec`;
  `parallel_pool.py` keeps `ear=NaN` when FaceMesh is missing (the old 0.0 sentinel would read
  as a permanent blink; verified nothing consumes lip-log `ear` — the anchor uses `is_moving`).
  12 s real-footage smoke: frame CSV has `ear` (0.188–0.362, 0 % NaN), windowed blink features
  fully populated, an actual blink lands in one window (count 1, EAR dip + var spike).
  NOTE: `GT_SUBJECTA_20260708` outputs predate the fix (blink features NaN there); the next
  full run (SPOVNOB production pass) regenerates them with blink alive.
- **2026-07-08** — **Full-document sync pass** (sections, not just changelog): header sync line →
  `04f660b` baseline + clean trees; §1 doctrine now names the validation/scoring role of ELAN
  labels; §3 map adds `validation/gt_subjectA/` + `00SubjectA_session1` footage; §7 Phase 2
  documents the blink/EAR merge; §12 validation-debt items 1/2 marked unblocked by the SubjectA
  baseline recording, item 3 notes the recording exists, item 4 corrected (single-pass WavLM has
  now run on GPU; only the fp32 A/B remains); §14 roadmap item 1 → SPOVNOB production pass,
  item 2 carries the empirical attribution mandate + per-channel directions, item 3 records the
  VideoMAE hold decision.
- **2026-07-09** — **First full PRODUCTION-path run on SubjectA + click-UI hardening + air-gap
  fixes.** SPOVNOB Stage-1 (`rec_subjectA`, operator-clicked, hash chain intact, 26 clean
  segments) → recording-level Stage-2 (`REC_SUBJECTA`, 8 clips, real target-only audio, blink
  alive) → ELAN re-score. Full write-up: `validation/gt_subjectA/RESULTS_PRODUCTION.md`.
  - **Click UI (`click_ui.py`, `vision.py`) — three operator-reported issues fixed + self-tested:**
    (1) **Frame/PTS mismatch root-caused** — the SubjectA MPEG-2 is open-GOP with 2 leading
    `AV_PKT_FLAG_DISCARD` packets, so `video_frame_pts_ms` listed 2640 PTS for 2638 decoded
    frames and **mislabeled every frame (and click) 80 ms early on all 8 clips**; now excludes
    discard-flagged packets (pairing exact, first frame 80 ms). Cache schema → v2 (stale v1
    pre-scans carry the shifted pairing). (2) **Audio playback** added (`/audio` endpoint +
    play/pause + frame-sync + `p` hotkey) — an off-screen interviewer is now identifiable by ear.
    (3) **Per-seed removal** (`remove_seed` + ✕ buttons) — a misclicked beard-mode seed no longer
    forces a full restart. self-test extended (12c) + green.
  - **Air-gap mandate hardcoded** (defense-grade): `acoustic_extractor.py` forces
    `TRANSFORMERS_OFFLINE`/`HF_HUB_OFFLINE`/`HF_DATASETS_OFFLINE` before importing transformers
    (clip 06 had failed mid-run on a huggingface 407 proxy phone-home); `environment_gate.py`
    (SPOVNOB side, already offline) gains `HF_DATASETS_OFFLINE` for parity.
  - **First genuinely-calibrated recording.** Baseline z |mean| 0.000 / std 1.000, all 134
    features non-NaN (blink included), median dev 9.85 ≈ √134. Interviews deviate above baseline
    (11–20). Corrected the §8 "baseline ≈ 0" myth (mathematically impossible for an L2 aggregate;
    0.0 = degenerate, which is what rec_ca was).
  - **offset_ms MEASURED** (§12.1): ~80 ms A/V desync (video leads audio), open-GOP artifact,
    exact from PTS + corroborated by mouth↔energy cross-correlation. Negligible at 2 s windows;
    must be fixed in the canonicalizer before the frame-level ST-GAE. Deferred to ST-GAE build.
  - **Signal REPLICATED on the independent production pipeline** (isolated audio, real
    diarization). Within-06 visual AUCs reproduce to ±0.01 (AU12 tremor 0.696, hand↔face 0.680,
    wrist 0.667; gaze inversion **bit-identical** 0.162). **Full 11-node attribution table**
    (RESULTS_PRODUCTION.md) shows a bipolar freeze+leakage signature: au_mouth/hands/au_upper
    activate (0.63–0.70), gaze/head/blink/voice/body freeze (0.16–0.38). Blink's first-ever
    measurement: weak/inverse (0.365, freeze cluster). Isolated `wavlm_latent_4` 0.574 (was
    0.608 whole-clip — interviewer contamination removed). Empirically re-confirms the
    direction-aware ST-GAE mandate.
  - **ST-GAE design (`Documentation/ST_GAE_DESIGN.md`, new)** finalized + user-reviewed: 11
    nodes LOCKED with **mandatory feature-count-normalized loss** (÷F_n so the 2-dim blink node
    isn't drowned by the 18-dim voice node); §6 gains **Bar 4 — Holdout Truth Stability**, the
    go/no-go false-positive gate against small-baseline overfitting.
- **2026-07-09** — **Canonicalizer A/V-sync fix + full ST-GAE implementation + v1 falsification.**
  - **Canonicalizer** (`ffmpeg_ingestion/core/canonicalizer.py`): probes per-stream `start_time`
    and trims the audio front by the video-vs-audio skew, correcting the ~80 ms open-GOP A/V
    desync (§12.1) at the source so all downstream consumers get synced streams. Verified:
    mouth↔energy cross-correlation lag −100 ms → **0 ms** on the re-cascaded `REC_SUBJECTA_SYNCED`.
  - **ST-GAE package** (`deception_detection/stgae/`: graph_spec, dataset, model, fit, attribute)
    implemented per `ST_GAE_DESIGN.md`: 11-node graph, learned adjacency, tiny masked autoencoder
    (23 k params, feature-count-normalized loss), per-subject baseline fit with anti-overfit
    stack + loud degenerate-fit gate. Tests `tests/verify_stgae.py` (16 checks, green): graph
    integrity, masking, ÷F_n loss equality (blink==voice), zero-grad on masked nodes,
    determinism, and an overfit-on-noise failure that fires. A key model fix surfaced from the
    tests: the temporal-only bottleneck let the AE *copy* the input (reconstructs noise); added a
    channel bottleneck (embed 16→latent 4) so reconstruction is genuinely compressive.
  - **v1 FALSIFIED** (`validation/gt_subjectA/STGAE_RESULTS.md`; `ST_GAE_DESIGN.md` §10): fit is
    healthy (recon ratio 0.274) but attribution fails all four bars — reconstruction error tracks
    *distance from the neutral baseline* (clip identity), not deception (the most **truthful**
    clip 04 is the **most** "anomalous"; all-lie clip 07 nearly the least). Bar 4 truth-flag-rate
    81% vs baseline 6% — exactly the small-baseline brittleness it guards against. Root cause is
    structural (domain gap dominates + node-aggregation dilutes the discriminative signal), not a
    hyperparameter. **The marginal per-channel z-score attribution remains the shippable method**
    (0.68–0.70). VideoMAE stays deferred (no working ST-GAE to feed it). The pre-registered bars
    did their job — prevented shipping a plausible-but-broken model. `stgae/` retained as the
    substrate for a future predictive/contrastive/supervised v2.
- **2026-07-10** — **Graph-line v2: predictive cross-modal coupling model built (laptop);
  real-data evaluation pending on the desktop.**
  - **Design pre-registered first** (`Documentation/COUPLING_MODEL_DESIGN.md`): the objective
    changes from *"can I reconstruct this window?"* (v1, falsified) to per-node masked
    prediction — *"hide this node entirely; can the other 10 predict it?"* Residual = "this
    channel stopped moving the way the subject's other channels say it should" (a decoupling
    detector, aimed at the validated freeze+leakage signature). Both v1 failure modes addressed
    by construction: conditional couplings transfer across the baseline↔interview domain gap
    far better than marginals + pooled scoring is within-clip; residuals stay per-feature.
    Bars 1–4 mirror ST_GAE_DESIGN §6; new Bar 0 = synthetic capability gate + fit ratio < 0.90.
  - **Code** (`stgae/coupling_model.py`, `coupling_fit.py`, `coupling_attribute.py`; graph_spec/
    dataset reused unchanged, v1 modules untouched as the record): learned per-node **mask token
    replaces the target's encoder stream** — the forward output is invariant to the target's
    input bit-for-bit at any depth (the only leak-safe construction under message passing);
    no bottleneck (nothing to copy), ~12 k params; loss = ÷F_n target-node error with
    zero-valid-target frames carrying zero weight. Degenerate gate's reference (predict-zero)
    now reads directly as "do this subject's channels carry information about each other?".
  - **RTX 6000 Ada profile** (design §7): whole baseline tensor set resident in VRAM (zero
    per-step transfer), the 11-target pass folded into the batch dim (one forward, not 11 —
    equivalence unit-tested), TF32 on; fp16 AMP deliberately omitted (launch-overhead-bound).
    Same code path runs CPU for the test suite.
  - **Tests `tests/verify_coupling.py` — 21/21 green (laptop, CPU)**: mask isolation (bitwise),
    vectorized≡sequential, ÷F_n equality, target-validity zero-grad; capability: a planted
    cross-node coupling is learned (ratio 0.344, driven node 0.16 vs independent node 1.00),
    **breaking it spikes exactly the broken node** (coupling-z median 88 vs quiet −0.1), a
    **marginal scale-shift with couplings intact stays 12× below the break** (7.5 vs 88 — the
    simulated v1 domain-gap failure does not recur), pure noise fires the degenerate gate
    (ratio 1.002).
  - **Evaluator ready** (`validation/gt_subjectA/coupling_evaluate.py`): fit on synced baseline,
    ELAN scoring-only overlay, per-node + per-feature AUCs (pooled on within-clip percentiles),
    bar-by-bar verdict printout. **Not yet run** — needs the desktop's
    `pipeline_system_outputs/REC_SUBJECTA_SYNCED_*`. Decision rule pre-registered: pass all →
    complement channel set to the marginal table; fail Bar 1 or 4 → graph line closed for n=1.
- **2026-07-10** — **Analyst report generator (`deception_detection/report/`) — the last mile of
  the attribution doctrine.** New per-recording end deliverable
  `<rid>_analyst_report.html` replacing "hand the analyst a 4,000-row CSV":
  - `report/analyst_report.py` (assembly + CLI) + `report/render_html.py` (renderer). ONE
    self-contained HTML — inline CSS + vanilla-JS canvas, **zero external requests** (air-gapped
    box), light/dark theming, design language inherited from the clip06_timeline mock. Pure
    pandas/numpy/stdlib — no torch, runs anywhere.
  - Sections: provenance header + doctrine disclaimer (attribution, never a verdict); data-quality
    & calibration-health panel (dead channels, uncalibratable constants, **degenerate-baseline
    alert** — the near-zero signature that rec_ca hit; would have caught the dead blink channel
    months earlier); per-clip deviation-percentile strips; direction-aware node table
    (suppressed/freeze ▼ vs elevated/leakage ▲, 11 groups); validated-channel timelines with
    hover tooltips (gaps for missing data, never fake zeros); flagged-window drill-down (≥ p95
    with top contributing channels); **conditional coupling lane** (renders only if
    coupling_attribution.csv exists AND the fit passed the 0.90 gate — states why when withheld);
    ELAN overlay **only** behind `--elan-dir` (validation mode, loudly labeled, never default).
  - Wired as non-fatal **Pass 5** of `process_recording_session` (a rendering failure never
    fails the recording); also standalone: `python -m report.analyst_report <recording_dir>`.
  - Tests `tests/verify_report.py` **22/22 green** (synthetic fixture incl. broken cases);
    `verify_end_to_end_pipeline.py` re-run after the Pass-5 hook: **375/375**.
  - Roadmap note added (§14.5): **more annotated subjects from the same experiment exist**
    (same questions/structure; no further SubjectA sessions) — videos to be shared ~2026-07-11.
- **2026-07-10** — **Multi-subject replication toolchain (`deception_detection/multisubject/`) +
  desktop handoff doc — built ahead of the N>1 corpus.** The N=1 ceiling is about to lift; this
  is the machinery to answer "does SubjectA's per-channel signal replicate?" *honestly*.
  - `intake_validator.py`: gate a subject package (videos + ELAN) BEFORE any GPU time. PASS/WARN/
    FAIL checklist (missing/ambiguous baseline, missing/corrupt/orphan .eaf, duplicate clip index,
    label vocabulary, thin annotation, and — with ffprobe — A+V stream presence and baseline
    duration), writes `intake_validation.json`, non-zero exit on FAIL. FAIL = cascade would
    break; WARN = processable-but-imperfect (an unannotated subject is a WARN — processable,
    yields no AUC). Clip convention documented: C001=baseline (file_index 0), C00n↔file_index n-1.
  - `replication_scorecard.py`: consumes each subject's `*_recording_calibrated.csv` + ELAN dir,
    scores every channel per subject (|z| AUC, Lie vs Truth, pure windows, ELAN scoring-only),
    emits channel×subject `replication_scorecard.csv` with per-channel verdicts. **Criteria
    PRE-REGISTERED 2026-07-10 before any non-SubjectA data was seen:** adequacy ≥30 pure Lie &
    ≥30 pure Truth per subject; R1 magnitude (AUC≥0.60 in ≥2/3 adequate); R2 direction (median
    shift ≥0.25 to express a sign — guards the 3-subject noise-sign pigeonhole — all expressed
    signs agree); verdicts REPLICATES / DIRECTION-ONLY / SUBJECT-SPECIFIC / NO-SIGNAL /
    INSUFFICIENT-DATA. SubjectA's validated channels reported first.
  - Tests `tests/verify_multisubject.py` **19/19** (planted 3-subject world exercising every
    verdict incl. the noise-sign trap; every intake failure class). Note: with 94 windows/class a
    noise AUC crosses 0.60 for ~1 seed in 6 — the scorecard correctly called that seed
    SUBJECT-SPECIFIC (honest), so the fixture pins a clean seed and asserts the no-signal property.
  - **Desktop handoff: `Documentation/MULTISUBJECT_REPLICATION_PLAN.md`** — the runbook for the
    Claude Code session that processes the new subjects (validate → cascade → report sanity-check
    → scorecard → record), the pre-registered criteria, verdict-interpretation guide, and the
    N>1 future-work specs: **§6.1 supervised attribution head** (GBT/logreg over per-channel z,
    strict leave-one-subject-out, informs channel weighting NOT a production verdict — legitimate
    only at N>1), **§6.2 per-subject coupling evaluation**, **§6.3 baseline-duration study**.
- **2026-07-10** — **Pre-flight hardening for the N>1 desktop session (de-risk real data).**
  Cross-checked this session's new code against the REAL SubjectA schema/file conventions and
  fixed the one genuine gap plus added a turnkey driver:
  - **Baseline is no longer assumed to be `file_index 0`.** Both the analyst report and the
    replication scorecard now recover the baseline clip index from `<rid>_baseline_stats.json`
    via new `analytics.baseline_calibrator.parse_baseline_file_index` (parses the
    `_NNN_windowed_features.csv` suffix). `process_recording_session` allows `baseline_file_index`
    != 0 for mis-named batches; the old hardcode would have mislabeled the baseline, run the
    health check on the wrong clip, and scored the true baseline as an interview. Report gains a
    baseline-index field; scorecard skips the recovered index.
  - Verified the report's `TRACE_CHANNELS` + `NODE_GROUPS` prefixes against the canonical column
    sources (`dynamic_window_engine`, `confidence_math.FFT_COLUMN_NAMES`,
    `acoustic_extractor.ACOUSTIC_COLUMN_NAMES`) — all names resolve; AU12/AU1_ and
    motion_energy/macro_motion prefix collisions checked clear.
  - **`multisubject/run_replication.py`** — one-command driver chaining intake → (manual GPU
    cascade) → scorecard around a single manifest; run before the cascade to validate, again
    after to score; FAIL gates scoring (non-zero exit). **`replication_manifest.template.json`**
    pre-filled with SubjectA's paths + a copy-me subject block.
  - Tests: `verify_report.py` 22→**24** (non-zero baseline recovery), `verify_multisubject.py`
    19→**24** (non-zero baseline in scorecard + driver end-to-end + FAIL-gates-scoring).
    Regression-checked: `verify_recording_calibration` green, `verify_end_to_end_pipeline`
    375/375 (Pass-5 report hook), coupling 21 / stgae 16 unaffected. Runbook (§4) + plan doc
    (§2.3–2.5) updated with the driver and the baseline-index rule.
- **2026-07-10 (desktop)** — **Home commits pulled + verified on the RTX 6000 box, then graph-line
  v2 EVALUATED and FALSIFIED.** Fast-forwarded the four laptop commits (coupling model `aece590`,
  analyst report `6552a61`, multisubject toolchain `da86748`, pre-flight hardening `90ea680`);
  re-ran all four new/graph suites on this box — **coupling 21/21, report 24/24, multisubject
  24/24, stgae 16/16** (torch 2.5.1+cu121, matches the laptop numbers). Then ran the one task that
  needs the desktop's data: `coupling_evaluate.py` on `REC_SUBJECTA_SYNCED_*` (all 8 synced raw
  CSVs + ELAN present).
  - **Result — v2 FALSIFIED by Bar 4** (full write-up `validation/gt_subjectA/COUPLING_RESULTS.md`;
    `COUPLING_MODEL_DESIGN.md §9`): fit healthy (12 255 params, ratio **0.801 < 0.90 gate**, Bar 0
    PASS) and — unlike v1 — **Bar 1 PASSES at 0.754** (feature `postural_stillness` coupling, above
    the marginal path's ~0.70). But the go/no-go **Bar 4 FAILS**: global coupling-z median baseline
    −0.63 / truth 39.69 / lie 22.23, flag-rate baseline 6% / **truth 93%** / lie 91% — truth spikes
    harder than lie, the same small-baseline brittleness as v1. Bars 2 (top nodes
    `[voice, au_upper, head_pose]`, no au_mouth/hand) and 3 (gaze decoupling 0.295) also fail.
  - **Root cause:** the design's bet (conditional couplings transfer across the domain gap where
    marginals don't) **holds per-feature within-clip** (Bar 1) but **breaks in aggregate** — the
    node-summed global coupling-z is still a distance-from-baseline meter (GLOBAL(sum) AUC 0.146,
    *inverted*, same sign flip as scalar `deviation_magnitude` 0.384). Bar 4 reads the aggregate on
    purpose (production false-positive gate) and fails.
  - **Decision (pre-registered, not moved):** v2 discarded like v1; **graph line CLOSED for the
    n=1 era.** Two independent pre-registered graph objectives, one failure mode → the obstacle is
    **N=1**, not the model. The coupling substrate is retained for a future supervised head once
    N>1 subjects exist. **The marginal per-channel z-score attribution remains the shipped
    instrument** (unchanged). Artifacts: `pipeline_system_outputs/REC_SUBJECTA/coupling_fit/` +
    `coupling_attribution.csv` (761 windows, file_index 2–7; clip 1 has no `.eaf`; baseline clip 0
    is the Bar-4 reference). Next: the new annotated subjects (§14.5) — now the critical unblock,
    not a nice-to-have.
- **2026-07-10 (desktop, SubjectB bring-up)** — Started the N>1 replication on the first new
  subject (`my_videos/01SubjectB_session2`, 7 clips; `validation/gt_subjectB/`). Bring-up + 2 fixes:
  - **Naming:** SubjectB carries two colliding original-ID series (`B06Cxxx`, `B41Cxxx`) that break
    the toolchain's `C###` clip token; the real sequence is the `-NN_` suffix. Mapping empirically
    verified (every ELAN max-end fits its matched video) and pinned in
    `validation/gt_subjectB/subjectB_manifest.json`. Canonicalized to bare `00_baseline..
    06_interview` names so the whole downstream pipeline is convention-identical to SubjectA (same
    80 ms open-GOP A/V skew auto-trimmed). A normalized ELAN dir (`normalize_elan.py`) gives each
    interview a unique `C{f+1:03d}` token (+ fixes a `LIe→Lie` label typo) so the pre-registered
    `replication_scorecard.py` runs unmodified. SPOVNOB Stage-1 ingests bare symlinks to the **raw**
    (audio-bearing) videos — the canonical MP4s are video-only (audio split to `_hubert.wav`), so
    Layer-0 rejects them. Note: SubjectB clips 04 (Truth-only) and 06 (876 s Truth / 17 s Lie) are
    effectively single-label; within-clip contrast concentrates in clips 01/02 (the long B06
    interviews); pooled budget 829 s Lie / 1498 s Truth clears the ≥30-window adequacy floor.
  - **click_ui.py audio-cache bug FIXED (recurring):** `/audio` was served on a bare static URL
    with `max_age=30d`, so the browser replayed a *previous* session's audio at the same
    `localhost:5050/audio`. The `video_sha8` field (present, labeled "browser cache-buster") was
    never wired in. Now the served HTML uses `audio?v=<video_sha8>` (content-unique) and is served
    `no-store`; stdlib self-test still green.
- **2026-07-10 (desktop) — N=2 REPLICATION RESULT: SubjectA's signal is SUBJECT-SPECIFIC.**
  SubjectB fully processed (SPOVNOB Stage-1 72 segments → Stage-2 cascade 7 clips → calibrate/
  assemble; healthy baseline median dev 10.06 ≈ √134, blink 100% populated). Scored against
  SubjectA with the pre-registered `replication_scorecard`. Full write-up
  `validation/multisubject/RESULTS.md`.
  - **Result: 0 REPLICATES / 7 SUBJECT-SPECIFIC / 127 NO-SIGNAL.** No tracked channel clears the
    0.60 bar in both subjects. SubjectA's AU12 lip-tremor family (0.601) + silent-speech
    incongruence (0.681) are A-specific; **SubjectB leaks through blink_rate (0.710), null in A
    (0.434)**. Positive control intact (SubjectA within-clip-06 AU12 0.68). Conclusion: per-channel
    deception leakage is **idiosyncratic per person** at N=2 — vindicates per-subject calibration +
    attribution; cautions against fixed cross-subject weighting; supervised head stays deferred (no
    robust cross-subject channel to learn at N=2).
  - **Two scorecard BUG-FIXES (not criteria changes; pre-registered thresholds unchanged):**
    (1) **global-vs-local timestamps** — scorer matched GLOBAL assembled-CSV window times against
    LOCAL ELAN intervals (SubjectA → 0 labels, SubjectB → shifted/mislabeled); fixed by rebasing
    each clip to local (subtract clip min start == file_offset_ms); restored SubjectA's exact
    509/140 counts. (2) **pooled → within-clip scoring** — pooled |z| AUC failed the SubjectA
    positive control (her own channels ~0.50); now AUC uses within-clip |z| percentile + direction
    on within-clip-centered z (the 07-08 / coupling-eval method). Regression added to
    `verify_multisubject.py` (Simpson's-paradox fixture, global times: pooled inverts to 0.17,
    within-clip recovers 0.66) → suite **24→28 green**.
  - **SubjectB bring-up recap:** dual-series naming (`B06Cxxx`/`B41Cxxx`) resolved via bare-name
    canonicalization + empirically-verified manifest; ELAN normalized (unique C-tokens + `LIe→Lie`);
    SPOVNOB on raw audio-bearing symlinks; click_ui audio browser-cache bug fixed. Drivers:
    `validation/gt_subjectB/`. Cascade ran 3-way (GPU only ~half-used — shard per-clip next time).
- **2026-07-15 (desktop) — N=6 corpus bring-up (SubjectC–F) + generic tooling + onnxruntime GPU
  fix.** All 4 remaining sessions arrived (SubjectC/D/E/F; single-series `B05/B31/B36/B03` naming).
  Built manifest-driven generic drivers in `validation/multisubject/`: `prep_subject.py` (manifest
  + bare symlinks + normalized ELAN with title-cased labels), `canonicalize_generic.py`,
  `cascade_generic.py`, `assemble_generic.py`. Intake caught real issues (lowercase/mixed-case
  labels normalized; D/E have interviews with no `.eaf` → cascaded, not scored). All 30 clips
  canonicalized. **onnxruntime GPU regression fixed:** env had `onnxruntime-gpu 1.17.1` (CUDA 11,
  wrong for this CUDA-12 box) → InsightFace fell back to CPU; restored `onnxruntime-gpu==1.19.2`
  and set the CUDA-lib launch recipe (`import torch` first + `LD_LIBRARY_PATH`=nvidia libs) — cascade
  now binds `CUDAExecutionProvider` (details: §11 stack + the cascade-throughput memory). SubjectC
  clicked + Stage-1 (49 segs) + Stage-2 cascade (GPU). D/E/F await operator clicks. Next: finish
  D/E/F → N=6 replication scorecard.
