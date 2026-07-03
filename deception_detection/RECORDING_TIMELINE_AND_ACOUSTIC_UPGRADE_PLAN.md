# Recording-Level Global Timeline, Baseline Calibration & Acoustic-Model Upgrade — Plan

> **Status:** Authored 2026-07-01. **Phase B (WavLM swap) implemented 2026-07-02** — see §4.3.
> **Phase A redesigned 2026-07-02** after the production data model was clarified (see §2) —
> baseline now comes from a dedicated calibration clip, and ELAN batch injection is dropped from
> scope. Phase A is next; not started.
> **Scope:** (A) baseline-clip-driven calibration + recording-level global timeline in
> `process_recording_session`. (B) Replace HuBERT with WavLM in the acoustic extractor. (C) Flag
> VideoMAE v2 as deferred future work.
> **Explicitly out of scope right now:** no dry runs on real footage. Everything here is designed
> to be buildable and unit-testable against synthetic/mocked CSVs (the same style as
> `tests/verify_diarization_bridge.py`), the way the merge seam was verified before any real GPU
> run happened.
> **Authority:** file:line references below were verified against the code at authoring time.

---

## 1. Current state (verified against code)

`process_recording_session` (`main_pipeline.py:591-634`) already loops over a recording's clips
and feeds each one its real diarization segments via `DiarizationBridge`, but every clip is
otherwise processed as if it were the *only* thing that ever happened:

- Each clip gets its own raw 30fps CSV, windowed CSV, and calibrated CSV (`process_video_session`,
  `main_pipeline.py:262-589`).
- `DynamicWindowEngine.compile_sliding_windows` (`analytics/dynamic_window_engine.py:41-286`)
  starts its time ruler at `start_time = 0` and its `window_id` at `0` for every call
  (`dynamic_window_engine.py:52,59`) — so "window 12" in clip 2 and "window 12" in clip 0 are
  unrelated windows on unrelated clocks.
- `BaselineCalibrator.calibrate` (`analytics/baseline_calibrator.py:29-151`) computes its baseline
  mean/std from `df['start_time_ms'] < calibration_duration_ms` (`baseline_calibrator.py:64`) —
  i.e. **the first 30 seconds of whatever CSV it's handed**. In batch mode today that means every
  clip after clip 0 calibrates its "neutral" baseline from its *own* opening seconds — which, for
  a multi-clip recording, is mid-interview, not neutral.
- `process_recording_session`'s call into `process_video_session` (`main_pipeline.py:623-628`)
  never passes `session_manifest_path` — so `ContextMapper` (`analytics/context_mapper.py`)
  silently defaults to NaN phase/question labels for every batch-mode run today. This is a live
  bug, not a future one.
- `batch_daemon.py`'s `process_recording` already knows all of this is unfinished — its own
  docstring says *"NOTE (parked): per-clip outputs only. ELAN injection, baseline calibration, and
  combined-per-recording assembly are deferred (coupled to the global-timeline design that is
  still pending)"* (`app/batch_daemon.py:496-498`, echoed again at `:567-568`). Single-clip mode
  (`process_recording`, non-batch path) already does ELAN injection via
  `inject_elan_labels`/`ELANAnnotationMapper` (`app/batch_daemon.py:159-258`) against the
  calibrated (or windowed) CSV — batch mode skips this entirely today.

So there are really **three** things parked behind "the global-timeline design," not one:
recording-level baseline calibration, recording-level ELAN ground-truth injection, and the
already-broken `ContextMapper` wiring in batch mode. (§2 re-scopes this list: ELAN batch
injection is dropped — it's a training-corpus concern, and no training is happening now.)

---

## 2. The production data model (clarified 2026-07-02 — supersedes the original §2/§3)

Two assumptions in the original plan were wrong, corrected by the user:

1. **There is no ground truth in production — ever.** Whether the subject is lying is the
   *output* the system produces, not an input it receives. ELAN-annotated, timestamped videos
   exist only as a small offline **training corpus**, and since there is no time for training
   right now (pretrained models only), ELAN batch-mode injection has **no current consumer** and
   is dropped from Phase A scope. Single-clip mode's existing ELAN path
   (`app/batch_daemon.py:159-258`) stays as-is for future training-corpus prep.
2. **Baseline is not "the first 30 seconds of the interview."** In actual execution the operator
   records a **separate, dedicated first video** in which the target gives generic information
   (name, background, neutral small talk). That clip *is* the subject's behavioral baseline. The
   interview clips that follow are scored as deviation from it.

Locked decisions from the 2026-07-02 session:

- **Baseline clip = file_index 0 of the same SPOVNOB batch.** One recording bucket, one operator
  click, one enrollment, one `pipeline_output.json` covering baseline + interview clips alike.
  (Convenient side effect: the click and enrollment happen on the baseline clip, where the target
  is reliably speaking solo.) Since SPOVNOB's canonical order is the lexicographic full-path sort,
  the operator must name the baseline clip so it sorts first; a `baseline_file_index` override
  hook is cheap to add for safety.
- **The whole baseline clip is the baseline pool** — no 30-second cap. Every valid window of the
  baseline video contributes to the mean/std.
- **Windows never cross clip boundaries** (hard break). Windowing runs per clip, which makes this
  automatic — no straddle special-casing needed.
- **Global timeline is still built** — not because calibration needs it (it no longer does), but
  for the continuous recording-level view: one assembled CSV with sequential window ids on one
  clock, for the dashboard and any downstream per-recording consumer.

Why this is *better* than the original global-assembly design, not just different: calibration no
longer depends on timeline stitching at all. Baseline stats are fit once on the baseline clip and
**applied** to each interview clip independently — a per-clip operation that cannot be corrupted
by clip-ordering or offset mistakes. The timeline assembly becomes a pure presentation/packaging
step, with no statistical consequences if it's wrong.

---

## 3. Design — baseline-clip calibration + recording-level assembly

### 3.1 `BaselineCalibrator`: split fit from apply

Today's `calibrate()` (`analytics/baseline_calibrator.py:29-151`) conflates two things: computing
baseline stats (from the first 30 s of whatever CSV it gets) and applying them (to that same CSV).
The original plan claimed no code change was needed here; **under the corrected model that's no
longer true.** The split:

- **`fit(baseline_windowed_csv) -> BaselineStats`** — per-feature mean/std over **all** valid
  windows of the baseline clip's windowed CSV (no duration cap), with the existing zero-std → NaN
  guard (`baseline_calibrator.py:100-110`). `BaselineStats` also carries the baseline window
  count and source path, and can be persisted as JSON next to the recording outputs so any later
  re-run or audit can reproduce the exact normalization.
- **`apply(windowed_csv, stats) -> calibrated_csv`** — Z-score every feature column against the
  fitted stats. Applied to every interview clip, **and to the baseline clip itself** (its
  deviations land near 0 by construction, which doubles as a sanity check — a baseline clip that
  deviates from its own stats signals a data problem).
- The existing single-CSV `calibrate()` stays untouched for the legacy single-clip path
  (`process_video_session` Phase 4), so nothing existing breaks.
- `deviation_magnitude` moves into `apply` unchanged; `deviation_percentile` is **removed from
  `apply` and computed once over the assembled recording-level CSV** (§3.3) — a percentile rank
  is only meaningful over one shared population, not per clip.
- **Failure policy:** if the baseline clip yields `< 2` valid windows, recording-mode calibration
  **fails loudly** (exception → recording marked FAILED in the ledger). Today's warn-and-output-raw
  fallback is fine for the legacy path but wrong for production recording mode: uncalibrated
  deviation scores presented as calibrated ones are exactly the kind of silent corruption this
  project's doctrine exists to prevent. (Flagged in §7 for sign-off.)

### 3.2 Recording flow (`process_recording_session`)

1. Per-clip cascade runs exactly as today — extraction, windowing (per clip → hard break at
   boundaries is automatic), per-clip windowed CSV. No change to `process_video_session`'s
   extraction phases; its Phase-4 self-calibration is **skipped in recording mode** (the
   orchestrator calibrates, not the clip).
2. `fit` on clip 0's (the baseline clip's) windowed CSV → `BaselineStats` → persisted JSON.
3. `apply(stats)` to every clip's windowed CSV (baseline included) → per-clip calibrated CSVs.
4. Assemble the recording-level CSV (§3.3).
5. Wire `session_manifest_path` through `process_recording_session` → `process_video_session` so
   `ContextMapper` stops silently defaulting to NaN in batch mode (the pre-existing bug from §1,
   bundled here).

### 3.3 Recording-level assembly (the global timeline, demoted to packaging)

Rebase each per-clip **windowed/calibrated** CSV onto the recording clock by adding that clip's
`file_offset_ms` (from `DiarizationBridge.file_offset_ms(file_index)` — reusing SPOVNOB's own
offsets so audio and video share one anchoring scheme) to `start_time_ms`/`end_time_ms`, add
`file_index` and `clip_window_id` provenance columns, concatenate in file_index order, renumber
`window_id` sequentially, then compute `deviation_percentile` over the whole assembled set. Output:
`<recording_id>_recording_calibrated.csv` — the primary downstream artifact. Per-clip CSVs stay on
disk as debug artifacts.

The time-alignment caveat from the original plan still applies but now with lower stakes: the
`offset_ms` reconciliation (audio local-PTS vs. video frame clock, `MERGE_INTEGRATION_PLAN.md` §7)
remains empirically unvalidated until real-footage runs resume — but since assembly is
presentation-only, an offset error can no longer corrupt calibration statistics.

### 3.4 Verification (no real footage, per constraint)

A new `tests/verify_recording_calibration.py` in the established style: synthetic per-clip
windowed CSVs with known distributions → assert fit stats match hand-computed values; assert
interview-clip z-scores against baseline stats (not their own); assert the baseline clip
z-scores to ~0 against itself; assert zero-std → NaN; assert the `< 2` windows hard-fail; assert
assembly rebases times by the right offsets, renumbers window ids, preserves provenance columns,
and computes percentiles over the combined population.

### 3.5 Implemented (2026-07-02)

Both §7 open questions were signed off (hard-fail on unusable baseline; `file_index 0` default
with a `baseline_file_index` profile override) and Phase A was built the same day:

- `analytics/baseline_calibrator.py` — added `BaselineCalibrationError`, `BaselineStats` (with
  JSON persistence; NaN stds encoded as null), a `NON_FEATURE_COLUMNS` constant, and
  `fit()`/`apply()` on `BaselineCalibrator`. Legacy `calibrate()` untouched. Note: the new
  `NON_FEATURE_COLUMNS` list *also* excludes `question_id`/`phase_elapsed_ms`/`context_phase` —
  the legacy path z-scores `question_id` and `phase_elapsed_ms` as if they were features, a
  pre-existing wart deliberately left alone for output parity but fixed in the recording path.
- `analytics/recording_assembler.py` *(new)* — `assemble_recording()` per §3.3, plus
  `RecordingAssemblyError` for empty input / missing per-clip CSVs.
- `main_pipeline.py` — `process_video_session` gained `calibrate: bool = True` (recording mode
  passes `False`; Phase 4 records `"status": "skipped"` in the clip manifest);
  `process_recording_session` rewritten per §3.2 (fit on baseline clip → persist
  `<recording_id>_baseline_stats.json` → apply to all clips → assemble
  `<recording_id>_recording_calibrated.csv`; a failed *interview* clip is excluded from assembly
  with a warning, a failed *baseline* clip raises). Its return type changed from a list to a dict
  carrying the recording-level artifact paths (the only existing caller,
  `_gpu_worker_batch_entrypoint`, ignores the return). `session_manifest_path` is now wired
  through (the §1 ContextMapper bug).
- `app/batch_daemon.py` — batch path reads `baseline_file_index` and `session_manifest_path`
  from `session_profile.json` (manifest resolved relative to the bucket, mirroring single-clip
  mode) and passes both through `_gpu_worker_batch_entrypoint`; stale "deferred" docstrings
  updated (ELAN note retained).
- `tests/verify_recording_calibration.py` *(new)* — 11 checks, all passing; full suite re-run
  green (`verify_end_to_end_pipeline` 375/375 confirms the legacy calibrate() path is
  byte-for-byte unaffected).

**Not validated (needs real footage, on hold):** the full recording flow through
`process_recording_session` itself (GPU imports make it un-importable in the test env) — the
fit/apply/assemble components are verified in isolation; the orchestration wiring is
compile-checked only. Same class of validation debt as the WavLM extractor (§4.3).

---

## 4. WavLM swap (replace HuBERT in `acoustic_extractor.py`)

### 4.1 Current implementation (verified)

`HuBERTAcousticExtractor` (`audio_isolation/core/acoustic_extractor.py:24-` ) loads
`facebook/hubert-base-ls960` via `transformers.HubertModel` + `Wav2Vec2Processor`, pulls Layer 7
hidden states (`HUBERT_LAYER_INDEX = 7`), and reduces them through a `MiniBatchKMeans` codebook
(`LATENT_CHANNELS = 16`, `CODEBOOK_SIZE = 64`) into the 20-column `ACOUSTIC_COLUMN_NAMES` schema
(`acoustic_extractor.py:18-22`): `acoustic_volatility`, `prosodic_velocity`, `hubert_latent_0..15`,
`vocal_entropy`, `acoustic_energy_rms`. Consumers: `dynamic_window_engine.py:18` (import),
`:192-198` (injection per window), `:267-269` (NaN fill on dropped windows); `main_pipeline.py:20`
(import), `:448` (instantiation), `:452-454` (manifest metadata).

### 4.2 Swap plan

- Model class → `transformers.WavLMModel`; feature front-end → `Wav2Vec2FeatureExtractor` (WavLM
  shares the Wav2Vec2-family input processor — `Wav2Vec2Processor`'s tokenizer half is unused for
  feature extraction either way, so this is a like-for-like swap, not a new dependency).
  `transformers==4.40.1` is already pinned (`requirements.txt:21`) and has shipped WavLM support
  since ~4.16, so no version bump should be needed — worth a one-line confirmation, not a
  re-pin.
- Checkpoint size is an open decision: `microsoft/wavlm-base-plus` (closer to HuBERT-base in
  size/latency, safer default given the TensorRT/latency-conscious posture already in
  `main_pipeline.py`) vs. `microsoft/wavlm-large` (better paralinguistic quality, more VRAM/latency
  on a box already running YOLO + InsightFace + OpenFace concurrently). See §7.
- Layer selection is a re-tuning item, not a straight port: `HUBERT_LAYER_INDEX = 7` was presumably
  chosen for HuBERT-base's layer/task profile. Published paralinguistic/speaker work on WavLM
  commonly uses a *learned weighted sum over all layers* rather than one fixed layer — notably the
  same reasoning SPOVNOB's own docs used to reject reusing a WavLM speaker-verification checkpoint
  (`SPOVNOB_MASTER_REFERENCE.md` §0.4: "published SV heads consume a learnable weighted sum over
  *all* transformer layers"). That specific SPOVNOB rejection was about *speaker verification*, a
  different task from this extractor's paralinguistic/deception-cue role — it doesn't block this
  swap, but it's a hint that a single fixed layer index may need re-validating, not just copying.
- Naming: `hubert_latent_*` should become something WavLM-appropriate (e.g. `wavlm_latent_*`, or a
  neutral `acoustic_latent_*` so a future model swap doesn't rename columns again). Recommend
  deciding this now, since **no real recorded CSVs exist yet to break** (per the no-dry-run
  constraint above) — this is the cheapest possible time to rename. Touches: the
  `ACOUSTIC_COLUMN_NAMES` tuple (`acoustic_extractor.py:18-22`) and its two consumers in
  `dynamic_window_engine.py` (`:192-198`, `:267-269`); consider also renaming the class itself
  (`HuBERTAcousticExtractor` → e.g. `WavLMAcousticExtractor`) and its import sites
  (`main_pipeline.py:20,448`) for consistency.

### 4.3 Implemented (2026-07-02)

All three §4.2 decisions were made and built:

| Decision | Chosen |
|---|---|
| Checkpoint | `microsoft/wavlm-large` (`WAVLM_MODEL_NAME`) |
| Layer selection | Fixed single layer, not a learned weighted sum |
| Column naming | `wavlm_latent_0..15` (renamed from `hubert_latent_*`) |

`acoustic_extractor.py` was rewritten: `HuBERTAcousticExtractor` → `WavLMAcousticExtractor`,
`HubertModel`/`Wav2Vec2Processor` → `WavLMModel`/`Wav2Vec2FeatureExtractor`. `WAVLM_LAYER_INDEX =
14` — chosen by proportional depth (wavlm-large has 24 transformer layers vs. HuBERT-base's 12;
`24 * 7/12 = 14` matches HuBERT's ~58%-depth choice). This is a reasoned placeholder, **not
empirically re-tuned** — no dry runs happened, per the no-real-footage constraint, so this index
should be revisited once real-audio validation resumes. The latent-profile reshape (`layer_size /
LATENT_CHANNELS`) is now computed from `model.config.hidden_size` at load time instead of a
hardcoded `48`, since wavlm-large's hidden size (1024) differs from HuBERT-base's (768).

Rippled through every consumer of the schema: `main_pipeline.py` (import, instantiation, manifest
metadata `model`/`layer` fields), `dynamic_window_engine.py` and `temporal_window_generator.py`
(comments only — they import `ACOUSTIC_COLUMN_NAMES` generically, no hardcoded copy), and
`analytics/predictive_engine.py` — which turned out to hold its **own hardcoded duplicate** of the
20-column schema (`ACOUSTIC_COLUMNS`, used by `CalibratedSessionLoader._validate_schema`'s FATAL
gate). Fixed by importing `ACOUSTIC_COLUMN_NAMES` from `acoustic_extractor.py` instead of
duplicating it, so the next model swap won't need to touch this file at all. All six
`tests/verify_*.py` scripts that touch the acoustic schema were updated to match and re-verified
passing (`verify_diarization_bridge`, `verify_merge_seam`, `verify_recording_intake`,
`verify_end_to_end_pipeline` at 375/375, `verify_behavioral_periodicity`); `verify_confidence_fusion`
has a pre-existing, unrelated `NameError` (`macro_motion_energy` test scaffolding) confirmed present
before this change too — not fixed, out of scope.

Validated without a dry run: `WavLMModel`/`Wav2Vec2FeatureExtractor` import cleanly under
`spovnob_env`'s pinned `transformers==4.40.1`; the rewritten module imports and its schema
(`ACOUSTIC_COLUMN_NAMES`, 20 cols, `wavlm_latent_0..15`) is correct; `predictive_engine.py`'s
`ACOUSTIC_DIM`/`VISUAL_DIM` compute correctly under the real CUDA-present env. **Not validated:**
actually instantiating `WavLMAcousticExtractor` and running inference — that downloads the
`wavlm-large` checkpoint and requires a real isolated WAV, which is exactly the dry-run class of
work still on hold.

Left untouched (deliberately, different concern — `_hubert`/`_hubert.wav` is a **filename-suffix
convention** for "the canonical audio track prepared for acoustic extraction," not a reference to
the HuBERT model): `diarization_bridge.py`'s `CANONICAL_SUFFIXES`, `recording_intake.py`,
`ffmpeg_ingestion/core/canonicalizer.py` + `batch_canonicalizer.py`, and the `_hubert`-suffix
fixtures in `verify_diarization_bridge.py`/`verify_recording_intake.py`. Also untouched:
`generate_manual_docx.py`'s prose (manual-generation content, cosmetic, lower priority — flagged as
a follow-up, not done).

**Found but not fixed (pre-existing, unrelated):** `tests/verify_confidence_fusion.py` has a
`NameError: name 'macro_motion_energy' is not defined` at line 120, confirmed present on the
unmodified file too (git-stash-verified). A local `.venv/bin/pip` symlink in this repo points at
the pre-restructure path `/home/user1/Documents/Audio_Diarization/.venv/bin/python` (broken since
the mono-repo rename, commit `ae2fc97`) — `pip` is unusable in that venv until repointed.

---

## 5. VideoMAE v2 — flagged for future work, not scheduled

Understood intent: a self-supervised video-transformer encoder for holistic spatiotemporal visual
features, as a third parallel extractor alongside MediaPipe pose and OpenFace AUs — capturing
subtler visual dynamics than the current hand-engineered kinematic/AU features.

No design commitment here beyond noting the integration shape, by analogy with
`acoustic_extractor.py`'s existing pattern: a new extractor class consuming per-clip frames (full
frame vs. the FaceLock crop already used for OpenFace, `main_pipeline.py:190-215`) → per-window
embeddings → new window columns, wired into `DynamicWindowEngine` the same way `acoustic_extractor`
is injected today (`dynamic_window_engine.py:191-198`). Open questions for whenever this is picked
up: input granularity (full frame vs. crop), checkpoint size (VideoMAEv2 base/large/giant), and GPU
budget alongside the already-heavy concurrent stack (YOLO + InsightFace + OpenFace + WavLM) on one
box.

---

## 6. Phased plan

- **Phase A (redesigned 2026-07-02) — Baseline-clip calibration + recording-level assembly.**
  Per §3: split `BaselineCalibrator` into `fit`/`apply` (fit on the whole dedicated baseline clip
  = file_index 0, apply to every clip); rebase+concatenate per-clip calibrated CSVs into
  `<recording_id>_recording_calibrated.csv` with recording-level percentiles; wire
  `session_manifest_path` through the batch path; hard-fail on an unusable baseline clip. ELAN
  batch injection is **out** (no training now; no ground truth in production, ever). Verify with
  synthetic per-clip CSVs (`tests/verify_recording_calibration.py`, §3.4) — no GPU, no real
  footage. **Done 2026-07-02** — see §3.5.
- **Phase B — WavLM swap.** Replace the model/processor in `acoustic_extractor.py`, resolve the
  checkpoint-size and layer-index decisions (§7), rename the schema columns, update consumers.
  Verify against a short local/synthetic WAV buffer, not a forensic clip. **Done 2026-07-02** — see
  §4.3.
- **Phase C — VideoMAE v2 (later).** Revisit when scheduled; no work planned now.

---

## 7. Open questions — resolved vs. still open

**Resolved (2026-07-02):**

1. ~~**Cross-clip-boundary windows**~~ → **Hard break.** Automatic under §3's per-clip windowing.
2. ~~**WavLM checkpoint**~~ → **`microsoft/wavlm-large`.** Built (§4.3).
3. ~~**WavLM layer**~~ → **Fixed single layer** (`WAVLM_LAYER_INDEX = 14`, proportional-depth
   placeholder pending future re-validation). Built (§4.3).
4. ~~**Column naming**~~ → **`wavlm_latent_0..15`.** Built (§4.3).
5. ~~**Baseline source**~~ → **Dedicated baseline clip = file_index 0 of the same SPOVNOB batch;
   the whole clip is the baseline pool** (no 30 s cap). See §2.
6. ~~**ELAN in batch mode**~~ → **Dropped from scope.** Training-corpus concern only; no training
   now, no ground truth in production ever. Single-clip path untouched.
7. ~~**Global timeline**~~ → **Still built**, but demoted to presentation/packaging (§3.3) —
   calibration no longer depends on it.
8. ~~**Intermediate CSVs**~~ → keep per-clip CSVs on disk as debug artifacts.
9. ~~**`session_manifest_path` fix**~~ → bundled into Phase A (§3.2 step 5).

10. ~~**Baseline hard-fail policy**~~ → **Fail the recording.** `BaselineCalibrationError` raises
    out of the subprocess → non-zero exit → ledger FAILED. Built (§3.5).
11. ~~**Baseline-clip identification**~~ → **file_index 0 default + `baseline_file_index`
    override in `session_profile.json`.** Built (§3.5).

**Nothing currently open.** Remaining work is all validation debt gated on real footage: the
`offset_ms` time-alignment measurement (§3.3), the WavLM layer-index re-tune (§4.3), and a first
real end-to-end recording run through `process_recording_session` (§3.5).
