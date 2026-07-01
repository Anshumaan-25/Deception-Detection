# SPOVNOB ↔ Deception Pipeline — Merge Integration Plan

> **Status:** Planning / design (no code changed yet). Authored 2026-06-29.
> **Scope:** Wire the completed SPOVNOB forensic audio-diarization pipeline into the main
> multimodal deception-detection pipeline, replacing the placeholder diarization path.
> **Authority:** The code in both repos is ground truth; line numbers below were verified at
> authoring time and should be re-checked if the files move.

---

## 1. The two systems

| | SPOVNOB (audio diarization) | Deception pipeline (this repo) |
|---|---|---|
| Location | `/home/user1/Documents/Audio_Diarization/` (parent dir) | `/home/user1/Documents/Audio_Diarization/SPOVNOB/` |
| Entry | `pipeline_runner.py --run` / `run.sh` | `main_pipeline.py` → `MultimodalProductionOrchestrator.process_video_session(...)` |
| Job | Emit PTS-stamped, overlap-excluded, **clean target-only speech** segments + WAVs for a same-session batch of clips | Multimodal (visual + pose + audio) feature extraction → windowing → calibration for deception scoring |
| Env (this box) | `.venv` (torch 2.1.2+cu121, numpy 1.26.4, onnxruntime-gpu 1.17.1, speechbrain 1.0.0, pyannote 3.1.1) | conda `spovnob_env` (torch 2.5.1+cu121, numpy 1.26.4, onnxruntime-gpu 1.15.1, speechbrain 1.1.0, pyannote 3.1.1) |
| Posture | Air-gapped, bit-deterministic, hash-chained audit, fail-closed env gate | Best-effort, online (HuggingFace HuBERT), no determinism mandate |

> **Note on `Yolo_v8/deception.yml`:** that spec (numpy 2.2.6, torch 2.10.0, CUDA 12.8,
> onnxruntime 1.23.2) has `prefix: /home/nv05/...` — it is the **production box (`nv05`)**
> environment, *not* this dev box. It is hard-incompatible with SPOVNOB. See §4.

---

## 2. Locked decisions (from the 2026-06-29 planning session)

1. **A "session" = a BATCH of clips from one recording.** SPOVNOB enrolls across the batch
   (its native, most-robust mode). The deception pipeline grows an outer loop.
2. **Keep the human click.** The operator clicks the target once (clip 0) via SPOVNOB's
   `click_ui.py`. No auto-click bridge. We can reuse SPOVNOB's `run.sh` click→batch flow.
3. **Black box (no forensics propagated).** The deception layer consumes only SPOVNOB's
   `pipeline_output.json` + clean WAV timeline. SPOVNOB still writes its hash-chained manifest
   internally; we ignore it downstream.

---

## 3. The integration seam (verified locations)

The deception pipeline already has a fully-formed audio integration seam wired to a **mock**.
Real diarization replaces the mock; everything downstream is reused unchanged.

**Producer side (SPOVNOB) — the deliverable we consume:**
- `pipeline_runner.run_pipeline(videos, clicks_path, work_dir, model_store, manifest_path, operator)`
  → writes `<work_dir>/pipeline_output.json` (schema `spovnob-pipeline-output-v1`) whose
  `clean_segments[]` each carry: `file_index`, `start_local_ms`, `end_local_ms`,
  `start_global_ms`, `end_global_ms`, `duration_ms`, `wav_path`, `wav_sha256`.

**Consumer side (deception) — the seam:**
- `main_pipeline.py:384` — the mock to replace:
  ```python
  mock_pyannote_segments = [("SPEAKER_00", 0.0, timestamp_ms)]  # To be wired to actual PyAnnote
  ```
  Contract: `list[(speaker_id: str, start_ms: float, end_ms: float)]`.
- `main_pipeline.py:387-394` — `anchor_target_identity(segments, visual_lip_logs)` picks the
  target speaker; `target_segments = [s for s in segments if s[0]==target_id]`.
- `main_pipeline.py:393-417` — **incongruence features** per frame: `is_audio_active`,
  `mismatch_incongruence` (audio on, lips still), `silent_incongruence` (lips moving, audio
  off), `diarizer_conf`. **These need a per-frame "is the target speaking at t?" signal** —
  exactly SPOVNOB's segment timeline.
- `main_pipeline.py:421-424` — `execute_isolation_pipeline(input_wav_path=canonical_wav_path,
  visual_speech_logs=..., pyannote_segments=..., output_dir=...)` →
  `diarizer_engine.py:104` → `isolate_voice_channel` (`diarizer_engine.py:64`) attenuates all
  non-target audio by `attenuation_factor=0.05` and writes `isolated_target_audio.wav`.
- `acoustic_extractor.py:37` — `HuBERTAcousticExtractor(isolated_wav_path)` consumes that WAV
  (16 kHz mono) → 20-column acoustic feature schema (`ACOUSTIC_COLUMN_NAMES`,
  `acoustic_extractor.py:18`).

**Key insight:** SPOVNOB is a *superset* of this seam. Its clean-segment timeline maps to
`[("TARGET", start_ms, end_ms), …]` and feeds BOTH consumers (incongruence frames AND the
isolation WAV) with no change to the downstream code — we change only the **source** of
`pyannote_segments`. Because `execute_isolation_pipeline` already does the WAV masking, we do
**not** need a separate masking adapter; feeding it SPOVNOB's authoritative (overlap-excluded)
segments instead of the whole-clip mock is the entire improvement.

---

## 4. Why subprocess isolation (not in-process import)

SPOVNOB's `environment_gate.run_gate()` is **fail-closed**: it halts on any drift from its 23
exact package pins, on `torch.version.cuda != "12.1"`, on a forbidden import, on a model-store
hash change, and after a ~10 s GPU determinism self-test. The deception env does **not** satisfy
those exact pins:

| Package | SPOVNOB `.venv` | Deception `spovnob_env` (this box) | Deception `deception.yml` (nv05 prod) |
|---|---|---|---|
| torch | 2.1.2+cu121 | 2.5.1+cu121 | 2.10.0 (CUDA 12.8) |
| numpy | 1.26.4 | 1.26.4 | **2.2.6** (ABI break) |
| onnxruntime-gpu | 1.17.1 | 1.15.1 | 1.23.2 |
| speechbrain | 1.0.0 | 1.1.0 | absent |
| scipy | 1.12.0 | 1.13.1 | 1.15.3 |
| pyannote.audio | 3.1.1 | 3.1.1 | absent |

On this dev box the envs are *close* (so GPU coexistence is low-risk), but still not pin-exact,
so the gate would halt in-process. On the nv05 production box they are fundamentally
incompatible. **Decision:** SPOVNOB runs in its own pinned `.venv` as a sealed subprocess; the
deception pipeline (whatever env/box) consumes its output files. This is robust to the env
discrepancy and honors decision #3 (black box).

**Open deployment question:** confirm whether production is `nv05` (`deception.yml`). If so,
SPOVNOB's `.venv` (or an equivalent pinned venv + vendored model store) must be replicated there
too, since `spovnob_env` does not exist on that box.

---

## 5. Target architecture

```
RECORDING SESSION (N clips, one subject, one room/mic)
        │
        ▼  (human, once)
  SPOVNOB click UI (clip 0)  →  clicks.json
        │
        ▼  [ SPOVNOB .venv — sealed subprocess: source env.sh ]
  pipeline_runner.py --run --videos clip_0..N --clicks clicks.json
        │
        ▼  pipeline_output.json  +  layer3/clean/*.wav   (per-file, overlap-excluded)
        │
        ▼  [ deception env: spovnob_bridge.py — NEW ]
  parse summary → group clean_segments by file_index → map each deception clip → file_index
  per clip: emit [("TARGET", start_ms, end_ms), …]   (reconciled to the frame timebase)
        │
        ▼  [ deception env: main_pipeline.py — outer batch loop + seam swap ]
  for clip i in batch:
      process_video_session(canonical_mp4_i, canonical_wav_i, ...,
                            pyannote_segments = bridge.segments_for(i))
          → incongruence features (unchanged)
          → execute_isolation_pipeline (unchanged; now fed authoritative segments)
          → HuBERTAcousticExtractor ← isolated_target_audio.wav (unchanged)
          → windowing / calibration (unchanged)
```

Two GPU stacks never collide: SPOVNOB runs to completion **before** the deception pass —
sequential processes, separate envs, never co-resident.

---

## 6. New component — `spovnob_bridge.py` (deception repo)

Pure Python in the deception env (numpy + stdlib + `scipy.io.wavfile`, already present). **Imports
nothing from SPOVNOB.** Two responsibilities:

**(a) Orchestration.** Invoke SPOVNOB once per batch as a subprocess:
```
bash -lc 'cd /home/user1/Documents/Audio_Diarization && source env.sh && \
  python pipeline_runner.py --run --videos <clips...> --clicks <clicks.json> \
  --work-dir <work> --model-store <store> --manifest <manifest.jsonl> --operator <id>'
```
(Or accept a pre-computed `pipeline_output.json` path, so the deception run can be decoupled
from the SPOVNOB run.) Surfaces non-zero exit + stderr as a hard error.

**(b) Parse + adapt.** Load `pipeline_output.json` →
- `segments_for(file_index) -> list[("TARGET", start_ms, end_ms)]` using `clean_segments`
  filtered by `file_index`, in the timebase the deception pipeline expects (§7).
- `isolated_wav_for(file_index)` (optional) — direct path to SPOVNOB clean WAVs, if we ever
  bypass `execute_isolation_pipeline`.
- Robust **clip → file_index mapping** by basename (not positional), validated against SPOVNOB's
  canonical lexicographic full-path sort.

Testable fully offline against the real sample at
`/home/user1/Documents/Audio_Diarization/session/batch01/pipeline_output.json`.

---

## 7. Time-alignment contract (the subtle part)

SPOVNOB `clean_segments` are in **local-PTS ms** (`start_local_ms` includes the file's
`audio_start_pts_ms`). The deception pipeline:
- builds incongruence features against **video frame timestamps** (`pose_rec["timestamp"]`,
  `main_pipeline.py:401`), and
- in `isolate_voice_channel` converts `start_ms/end_ms → sample = (ms/1000)*sample_rate` on the
  **canonical `_hubert.wav`** (data-relative, sample 0 origin).

The bridge must therefore reconcile SPOVNOB local-PTS ms with the deception data-relative/frame
clock — i.e. account for `audio_start_pts_ms` and any audio-vs-video PTS offset. With the old
whole-clip mock this never mattered; with sparse real segments it does. For most containers the
offset is ~0–25 ms (negligible vs 1 s blocks) but must be handled for correctness and **validated
empirically on one real clip** (§9).

---

## 8. Changes required (file-by-file, deception repo)

1. **`spovnob_bridge.py`** *(new)* — §6.
2. **`main_pipeline.py`**
   - `process_video_session` (`:261`): accept a `pyannote_segments` argument (default `None` →
     fall back to current mock for backward-compat / unit tests).
   - Seam (`:384`): replace mock with the passed-in `pyannote_segments`.
   - Optional cleanup: since SPOVNOB already anchored identity, the redundant
     `anchor_target_identity` call (`:387`) can be bypassed (target is known = `"TARGET"`); keep
     for now to minimize change.
   - Add `process_recording_session(clips: list, session_id, ...)` outer loop: canonicalize all
     clips → run SPOVNOB once (via bridge) → loop `process_video_session` per clip with its slice.
   - `__main__` (`:581`): wire the new batch entry.
3. **(Optional) front-door** — a `run_session.sh` (or reuse SPOVNOB's `run.sh`) that does:
   clips dir → SPOVNOB click+batch → deception loop.
4. **Pre-existing fix (their side, flag only):** `acoustic_extractor` needs `transformers`
   (HuBERT `facebook/hubert-base-ls960`); it's present in top-level `requirements.txt`
   (4.40.1) and in `spovnob_env`, but **absent from `Yolo_v8/deception.yml`** (nv05). Reconcile
   before nv05 deployment.

---

## 9. Phased build plan

- **Phase 0 — Prove coexistence (½ day).** Confirm SPOVNOB `.venv` runs a real batch on this
  box (self-tests + one `run.sh`); confirm `spovnob_env` runs `main_pipeline.py`. No code.
- **Phase 1 — `spovnob_bridge.py`.** Build + unit-test offline against `session/batch01/
  pipeline_output.json`. (Load-bearing new component.)
- **Phase 2 — Seam + batch loop.** Add `pyannote_segments` param, swap the mock, add
  `process_recording_session`.
- **Phase 3 — Front-door.** Session runner (clicks → SPOVNOB → deception loop).
- **Phase 4 — Validate (see below).**

---

## 10. Risks & validation

1. **Time-frame alignment** (§7) — verify SPOVNOB local-PTS ms vs deception frame ms on one real
   clip; confirm incongruence features flip sensibly vs the old mock.
2. **Clip → file_index mapping** — deception clip order must resolve to SPOVNOB's lexicographic
   full-path `file_index`; map by basename and assert coverage.
3. **Empty / INSUFFICIENT / `TARGET_SILENT`** — if SPOVNOB yields no target speech, the existing
   `TARGET_SILENT` fallback (`diarizer_engine.py:115`) attenuates the whole track and incongruence
   degrades to NaN where pose is NaN; confirm it triggers cleanly from a real empty result.
4. **Source media for SPOVNOB** — SPOVNOB ingests the **original** clips (its own PTS-true
   extraction + InsightFace need real frames), while the deception pipeline canonicalizes
   separately. Confirm (assumption, pending user sign-off).
5. **Production env (`nv05`)** — replicate SPOVNOB's pinned venv + model store there if that box
   is the deployment target.

---

## 11. Open questions for sign-off

- Confirm SPOVNOB ingests **original** clips (not the 30 fps CFR canonical re-encodes).
- Confirm the production target box and env (`nv05`/`deception.yml` vs this box/`spovnob_env`).
- Where should `clicks.json` + SPOVNOB `work-dir`/manifest live per session (inside the
  deception session dir, or SPOVNOB's own `session/`)? Affects the front-door wiring.
