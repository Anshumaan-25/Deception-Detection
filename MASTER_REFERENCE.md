# MASTER REFERENCE — Forensic Deception Detection Mono-repo

> **This is the living master document for the entire project.** It is maintained continuously:
> every meaningful change to the pipeline (code, design decisions, model swaps, validation results)
> gets reflected here, with a dated entry in the [Changelog](#changelog) at the bottom.
> If this document and the code disagree, **the code wins** — and the discrepancy is a bug in this
> document that must be fixed.
>
> **Last synced with code:** 2026-07-06 (commit `4433b4c`).

---

## 1. Mission & use case

Given recorded interview footage of a single subject, produce a **mathematically grounded,
per-window behavioral deviation analysis** of that subject — localized in time, attributable to
specific feature channels, and calibrated against the subject's *own* neutral behavior.

Core doctrine:

- **Attribution, not classification.** The system never outputs a truth/lie verdict. It outputs
  *where* and *how strongly* behavior deviates from the subject's own baseline, per feature, per
  time window. Interpretation belongs to a human analyst.
- **No ground truth in production, ever.** ELAN-annotated videos exist only as a small offline
  training corpus for possible future supervised work. Nothing in the production path consumes
  annotations.
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
│   ├── dashboard/ + frontend/    ← diagnostic dashboard / browser UI
│   ├── tests/                    ← verify_*.py self-test suite (§10)
│   ├── SPOVNOB_intake/           ← intake watch folder (incl. SESSION_TEST_MOCK fixture)
│   ├── pipeline_system_outputs/  ← per-session + per-recording outputs (§9)
│   └── weights/trt_engines/      ← compiled TensorRT engine cache
├── session/                      ← SPOVNOB batch outputs (batch01, rec_ca, …)
├── my_videos/                    ← real test footage (CA Beard / NT clips / UB Beard-Tougher)
├── PIPELINE_ARCHITECTURE.md      ← the block diagram (Mermaid) — visual companion to this doc
├── MASTER_REFERENCE.md           ← this document
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

**Phase 2 — Raw feature compilation** (`compile_raw_features`).
Vectorized kinematics: 3D hand-to-face distance (L/R), 3D wrist velocity (L/R), gaze velocity,
AU onset velocity (AU1/2/4/6/9/12/25/26 — genuine expression onsets run ~250–500 ms; posed ones
don't), 7-landmark `macro_motion_energy`. Exact frame-by-frame inner join of pose × OpenFace on
the shared master-clock timestamp → **30 fps fused CSV**. *(This frame-level CSV is the planned
input of the future ST-GAE — §13.)*

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
| `<rid>_recording_calibrated.csv` | **CURRENT END DELIVERABLE** — all clips on the global timeline, z-scored, with `deviation_magnitude` + recording-wide `deviation_percentile` |

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

**Validation debt (as-built but unproven on real footage — blocked on the "no dry runs" hold):**

1. `offset_ms` audio↔video alignment defaults to 0; never empirically measured.
2. `WAVLM_LAYER_INDEX = 14` is a proportional-depth placeholder (HuBERT-base layer 7/12 scaled to
   24 layers), not re-tuned on real audio.
3. No real end-to-end `process_recording_session` run through the GPU stack. Everything is
   verified against synthetic CSVs (§13); the orchestration wiring itself is compile-checked only.

**Dead code / warts:**

- `analytics/predictive_engine.py` — superseded TFN classifier (decision 2026-07-06:
  attribution-not-classification; ST-GAE replaces it). **Archived in place 2026-07-06**: entire
  file commented out under an ARCHIVED/SUPERSEDED header, kept (not deleted) as a record of the
  approach. Was never imported by anything (only referenced in `generate_manual_docx.py` prose).
- `tests/verify_confidence_fusion.py` — pre-existing `NameError: macro_motion_energy` at line
  ~120 (present before the WavLM swap; confirmed via git-stash test). Not yet fixed.
- `deception_detection/.venv/bin/pip` — broken symlink to the pre-restructure path
  (`~/Documents/Audio_Diarization/...`), stale since the mono-repo rename.
- Legacy `calibrate()` z-scores `question_id`/`phase_elapsed_ms` (metadata) — pre-existing wart,
  kept for parity; the new fit/apply path excludes them correctly.
- `tools/generate_manual_docx.py` prose still says HuBERT (cosmetic).

## 13. Verification suite (`deception_detection/tests/`)

All pure pandas/numpy on synthetic data — no GPU, no real footage. Run from
`deception_detection/` with `python tests/verify_<name>.py`.

| Script | Covers | Status (2026-07-02 run) |
|---|---|---|
| `verify_diarization_bridge.py` | SPOVNOB JSON → seam contract (incl. real `session/batch01` fixture) | ✅ 10/10 |
| `verify_merge_seam.py` | seam semantics in `process_video_session` | ✅ |
| `verify_recording_intake.py` | mp4/wav ↔ file_index pairing | ✅ |
| `verify_end_to_end_pipeline.py` | full mocked cascade | ✅ 375/375 |
| `verify_behavioral_periodicity.py` | FFT block | ✅ |
| `verify_recording_calibration.py` | fit/apply/BaselineStats/assembly (11 checks) | ✅ 11/11 |
| `verify_confidence_fusion.py` | confidence-weighted math | ❌ pre-existing NameError (§12) |

## 14. Roadmap (future, in intended order — nothing scheduled)

1. **Real-footage validation** (first thing once the dry-run hold lifts): one real recording
   (baseline + interviews) through SPOVNOB → `process_recording_session` end-to-end; measure
   `offset_ms`; re-tune `WAVLM_LAYER_INDEX` empirically.
2. **ST-GAE end-stage** (concept/design only): spatio-temporal graph autoencoder over the raw
   synchronized 30 fps frame-level features (Phase 2 CSV). Fit per subject on the baseline clip
   (= "normal"); reconstruction error on interview clips = anomaly. Deliverable: **Temporal
   Anomaly Attribution Report** — timestamped cognitive-friction segments + node-wise
   reconstruction-error map. Open design questions: node/edge definition; per-subject fit cost;
   minimum baseline duration; replace-vs-complement the z-score path; VideoMAE v2's role.
3. **VideoMAE v2** (deferred): deep spatiotemporal latents as a 4th parallel extractor branch,
   presumed to feed the ST-GAE graph.
4. **Supervised training path** (distant): the ELAN-annotated corpus +
   `temporal_window_generator.py` exist for this; explicitly out of scope now.
5. **Production deployment target**: resolve dev-box vs nv05 (SPOVNOB pin incompatibility).

## 15. Document governance

| Document | Role |
|---|---|
| **`MASTER_REFERENCE.md`** (this) | Living master — always current; updated with every change |
| `PIPELINE_ARCHITECTURE.md` | The block diagram (Mermaid) — visual companion, kept in sync |
| `audio_diarization/SPOVNOB_MASTER_REFERENCE.md` | Deep authority for the audio side |
| `deception_detection/RECORDING_TIMELINE_AND_ACOUSTIC_UPGRADE_PLAN.md` | **Historical** — completed plan (Phases A+B, done 2026-07-02) |
| `deception_detection/MERGE_INTEGRATION_PLAN.md` | **Historical** — merge plan; known-stale vs code even before completion |
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
