# MASTER REFERENCE — Forensic Deception Detection Mono-repo

> **This is the living master document for the entire project.** It is maintained continuously:
> every meaningful change to the pipeline (code, design decisions, model swaps, validation results)
> gets reflected here, with a dated entry in the [Changelog](#changelog) at the bottom.
> If this document and the code disagree, **the code wins** — and the discrepancy is a bug in this
> document that must be fixed.
>
> **Last synced with code:** 2026-07-07 — committed baseline `8baea6e` **plus uncommitted
> working-tree changes** from the first real GPU cascade run (OpenFace PyTorch-native path,
> canonicalizer/YOLO/FaceLock/stream fixes, env changes — see the latest Changelog entry).

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
don't), 7-landmark `macro_motion_energy`. Exact frame-by-frame inner join of pose × OpenFace on
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

**Validation debt (as-built but unproven on real footage — blocked on the "no dry runs" hold):**

1. `offset_ms` audio↔video alignment defaults to 0; never empirically measured.
2. `WAVLM_LAYER_INDEX = 14` is a proportional-depth placeholder (HuBERT-base layer 7/12 scaled to
   24 layers), not re-tuned on real audio.
3. ~~No real end-to-end run through the GPU stack.~~ **Per-clip path resolved 2026-07-07**: the
   per-clip cascade (`process_video_session`) runs **end-to-end on real footage on the GPU**
   (CA-Beard clip, all four phases, valid output CSVs — see changelog). **Recording-level path is
   still BLOCKED by an open bug** (see "Open bugs" below): `process_recording_session` completes
   clip 0 (BASELINE) but hangs on clip 1. Until that's fixed, `offset_ms` and `WAVLM_LAYER_INDEX`
   remain unmeasured on real footage.

**Open bugs (found by the first real GPU runs 2026-07-07, NOT yet fixed — start here next session):**

- **Multi-clip recording hangs on the 2nd clip.** `process_recording_session` runs clip 0
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
  overflowing; this is the *task* queue starving because the workers go quiet. Candidate fixes next
  session: recreate/reset the MediaPipe pool per clip (mirroring how FaceLock is already recreated
  per session at `main_pipeline.py:326`); and/or add a worker-liveness check + a `submit_task`
  timeout so a dead pool fails loudly instead of hanging.
4. The 2026-07-07 single-pass WavLM rewrite (chunked full-clip forward, fp16 autocast, encoder
   truncation, whole-clip normalization, ~30 s transformer context instead of per-2 s-window
   forwards) is alignment-math-verified (§13) but has never executed on a GPU. Each optimization
   is independently toggleable via `WavLMAcousticExtractor` kwargs / module constants — first
   real-footage run should A/B `use_amp` and `truncate_encoder` against fp32 full-stack once.
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
| **`Documentation/MASTER_REFERENCE.md`** (this) | Living master — always current; updated with every change |
| `Documentation/PIPELINE_ARCHITECTURE.md` | The block diagram (Mermaid) — visual companion, kept in sync |
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
