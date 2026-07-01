# Forensic Deception Detection — Mono-repo

Two pipelines, one repository. The **audio diarization** pipeline isolates and verifies the target speaker's clean speech from interview recordings. The **deception detection** pipeline consumes that output and performs multimodal behavioral analysis (YOLO+FaceLock visual tracking, MediaPipe pose, OpenFace, acoustic feature extraction).

```
.
├── audio_diarization/       ← forensic speaker diarization (self-contained, own .venv)
├── deception_detection/     ← multimodal deception analysis (conda spovnob_env)
└── .gitignore
```

The two pipelines are **environment-isolated**: audio diarization runs in its own pinned `.venv` (activated by `audio_diarization/env.sh`); deception detection runs in the `spovnob_env` conda environment. The interface between them is `pipeline_output.json` — produced by audio diarization, consumed by deception detection as a black box.

---

## Audio Diarization (`audio_diarization/`)

Forensic speaker diarization pipeline. Processes batches of interview-style video recordings and isolates the clean, uncontaminated speech of a single visually-verified target speaker. Fully deterministic and reproducible, runs offline/air-gapped with SHA-256-pinned vendored models, records every decision in an append-only hash-chained session manifest.

Full reference: [`audio_diarization/SPOVNOB_MASTER_REFERENCE.md`](audio_diarization/SPOVNOB_MASTER_REFERENCE.md)

### Implementation status

| Module | File | Status |
|---|---|---|
| 0a — Session manifest (hash-chained audit log) | `session_manifest.py` | ✅ Complete |
| 0b — Environment gate (determinism + model vendoring checks) | `environment_gate.py` | ✅ Complete |
| 1 — Layer 0 preprocessor (PTS-true extraction + VAD segment map) | `layer0_preprocessor.py` | ✅ Complete |
| 2 — Layer 1 enrollment (visual-anchored speaker profile) | `layer1_enrollment/` | ✅ Complete |
| 3 — Layer 2 tracker (calibrated sliding-window target tracking) | `layer2_tracker.py` | ✅ Complete |
| 4 — Layer 3 contamination flagging (overlap exclusion) | `layer3_contamination.py` | ✅ Complete |
| 5 — Pipeline runner (production entrypoint) | `pipeline_runner.py` | ✅ Complete |

### Quick start

```bash
cd audio_diarization
source env.sh                  # activate pinned .venv
python pipeline_runner.py --selftest
```

Self-tests (stdlib-only, no GPU needed):
```bash
python3 session_manifest.py
python3 environment_gate.py --selftest
python3 layer0_preprocessor.py --selftest
python3 -m layer1_enrollment --selftest
python3 layer2_tracker.py --selftest
python3 layer3_contamination.py --selftest
python3 pipeline_runner.py --selftest
```

Full batch run via the operator script:
```bash
bash audio_diarization/run.sh
```

### Environment

Ubuntu 22.04 LTS · NVIDIA RTX 6000 Ada (48 GB) · Python 3.10 · CUDA 12.1 · PyTorch only.
Pinned deps: [`audio_diarization/requirements.txt`](audio_diarization/requirements.txt).
Setup: [`audio_diarization/UBUNTU_SETUP_GUIDE.md`](audio_diarization/UBUNTU_SETUP_GUIDE.md).
Model weights are never committed — staged into a local SHA-256-pinned model store.

---

## Deception Detection (`deception_detection/`)

Multimodal behavioral analysis pipeline. Consumes the audio diarization output (`pipeline_output.json`) and the original video clips, then runs YOLO+FaceLock visual tracking, MediaPipe 3D pose, OpenFace AU extraction, and WavLM acoustic feature extraction through a `DynamicWindowEngine` to produce per-window deception scores.

Integration plan: [`deception_detection/MERGE_INTEGRATION_PLAN.md`](deception_detection/MERGE_INTEGRATION_PLAN.md)

### Structure

```
deception_detection/
├── main_pipeline.py            ← production entrypoint
├── app/
│   ├── batch_daemon.py         ← intake watcher; routes N-clip batches or single clips
│   ├── recording_intake.py     ← pairs diarization file_index ↔ canonical clip/wav
│   └── server.py               ← FastAPI sidecar for the dashboard
├── audio_isolation/core/
│   ├── diarization_bridge.py   ← merge seam adapter (pure stdlib)
│   ├── diarizer_engine.py      ← MediaPipeAudioDiarizer (voice isolation)
│   └── acoustic_extractor.py   ← acoustic feature extraction
├── analytics/                  ← DynamicWindowEngine, BaselineCalibrator, etc.
├── frontend/                   ← React dashboard (Vite + Tailwind)
├── tests/                      ← verify_diarization_bridge, verify_merge_seam, etc.
└── requirements.txt
```

### External components (clone separately)

These sub-repos must be cloned into place — they are gitignored because they contain large model weights and have their own git histories:

```bash
git clone https://github.com/Anshumaan-25/Phase_1-openface_base.git  deception_detection/OpenFace-3.0
git clone https://github.com/Anshumaan-25/ffmpeg_ingestion.git        deception_detection/ffmpeg_ingestion
git clone https://github.com/Anshumaan-25/MediaPipe.git               deception_detection/mediapipe_pose
git clone https://github.com/Anshumaan-25/OPENCV_STREAMING.git        deception_detection/opencv_streaming
```

### Environment

Conda environment `spovnob_env` (torch 2.5.1+cu121, mediapipe, transformers, pyannote, fastapi).
Deps: [`deception_detection/requirements.txt`](deception_detection/requirements.txt).

---

## Running a full session

1. **Operator click** — launch the click UI from the audio diarization side to mark the target on clip 0:
   ```bash
   cd audio_diarization && source env.sh && python click_ui.py
   ```

2. **Audio diarization** — run the forensic diarization on the batch of clips:
   ```bash
   bash audio_diarization/run.sh
   # produces: audio_diarization/session/<batch>/pipeline_output.json + clean WAVs
   ```

3. **Stage recording bucket** — drop the pipeline_output.json + canonical clips into the deception intake:
   ```
   deception_detection/SPOVNOB_intake/<recording_id>/
     ├── pipeline_output.json
     ├── session_profile.json
     ├── clip0_canonical.mp4 + clip0.wav
     └── ...
   ```

4. **Deception detection** — the batch daemon picks it up automatically:
   ```bash
   conda activate spovnob_env
   cd deception_detection && python app/batch_daemon.py
   ```
