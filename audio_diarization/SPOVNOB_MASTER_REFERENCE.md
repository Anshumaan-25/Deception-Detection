# SPOVNOB — The Complete Pipeline Reference (Ultimate Edition)

> **What this document is.** A single, self-contained, code-grounded reference for the entire
> SPOVNOB forensic audio-diarization pipeline. It supersedes and replaces the four earlier
> documents (`Audio_Diarization.md`, `SPOVNOB_COMPLETE_CONTEXT.md`,
> `SPOVNOB_TECHNICAL_DEEP_DIVE.md`, `SPOVNOB_PIPELINE_SUMMARY.md`). It is written *from the code at
> repository head*, module by module, function by function, constant by constant. Where any earlier
> narrative disagreed with the implementation, the implementation wins and the divergence is noted.
>
> **Authority rule.** The Python modules are the ground truth. Every threshold below is quoted with
> the exact comparison operator the code uses — in a forensic gate `>` and `≥` are never
> interchangeable.
>
> **Status:** All seven pipeline modules complete (`session_manifest`, `environment_gate`, Layers
> 0–3, `pipeline_runner`) plus operator tooling (`click_ui.py`, `audit_visualizer.py`,
> `diag_windows.py`, `env.sh`, `run.sh`). Behavioral / paralinguistic analysis is deferred to a
> separate design phase and is out of scope.
>
> **Deployment target:** Ubuntu 22.04 LTS · NVIDIA RTX 6000 Ada (48 GB VRAM) · Python 3.10.x ·
> PyTorch `+cu121` (CUDA 12.1 runtime) on a CUDA-13.x / driver-580 host.

---

## Table of Contents

- **Part 0 — Orientation:** mission, doctrine, constraints, the refusals, environment, repo map, the time frames
- **Part 1 — The Substrate:** `session_manifest.py` (0a) and `environment_gate.py` (0b)
- **Part 2 — Layer 0:** `layer0_preprocessor.py` — PTS-true extraction + the Silero speech map
- **Part 3 — Layer 1:** the `layer1_enrollment/` package — visual-anchored (and audio-anchored) enrollment
- **Part 4 — Layer 2:** `layer2_tracker.py` — calibrated sliding-window tracking
- **Part 5 — Layer 3:** `layer3_contamination.py` — overlap exclusion + final output
- **Part 6 — The Runner:** `pipeline_runner.py` — orchestration + audit closure
- **Part 7 — Operator tooling:** `env.sh`, `run.sh`, `click_ui.py`, `audit_visualizer.py`, `diag_windows.py`
- **Part 8 — Determinism, time, hashing** (cross-cutting deep dive)
- **Part 9 — The five models** (ECAPA-TDNN, InsightFace, YOLOv8, Silero VAD, PyAnnote OVD)
- **Part 10 — Complete parameter reference** (every constant, value, operator, rationale)
- **Part 11 — Manifest operation vocabulary** (every entry type)
- **Part 12 — Setup & deployment** (Ubuntu guide + the seven gotchas)
- **Part 13 — The self-test architecture**
- **Part 14 — Bench-validation register & known limitations**
- **Appendix A — File-by-file line/function map**
- **Appendix B — Glossary**

---

# Part 0 — Orientation

## 0.1 What SPOVNOB does (one paragraph)

SPOVNOB ingests one **batch** of interview videos — 5 to 15 files of 5–10 minutes each, all cut
from **one continuous recording session** (same subject, same room, same microphone, same day) —
plus one mandatory operator click (and one optional one) on the first video. It emits
**PTS-timestamped WAV segments containing only the visually-verified target speaker's speech,
verified free of overlapped speech**, each segment SHA-256-hashed, plus a hash-chained audit log
recording every parameter, decision, acceptance, discard, warning, and halt that produced them. The
output is the input contract for a future, deferred behavioral-analysis phase.

The name is treated throughout the code simply as the project identifier (manifest schema
`spovnob-manifest-v1`, output schemas `spovnob-layer2-output-v1`, etc.).

## 0.2 The two properties that define the system

1. **Every output sample existed in the source.** The pipeline never synthesizes, interpolates,
   reconstructs, or separates audio. Output WAVs are byte slices of the original 16 kHz extraction
   (`pcm_slice` in `encoding.py` is the only slicer, and `layer3_contamination._write_segment_wav`
   the only writer). Contaminated audio is **excluded, never repaired**.
2. **The whole run is a deterministic function of its inputs.** Re-running the same batch with the
   same model store produces **bit-identical decision payloads and output hashes**. "Same inputs →
   same SHA-256s, today or in two years" is the engineering standard, not an aspiration.

## 0.3 The non-negotiable constraints, and why each exists

| # | Constraint | Forensic rationale |
|---|---|---|
| 1 | **Bit-identical determinism** — CUDA deterministic mode, float32-only, fixed batch shapes, order-independent reductions | An auditor must reproduce the run and match hashes. Verification becomes a string comparison. "Statistically similar" output invites the unanswerable question *which run is the evidence?* |
| 2 | **Zero synthesis / zero separation** — HTDemucs, SepFormer, SpeakerBeam and all source-separation models categorically forbidden | Separation models *hallucinate* plausible source audio by construction. No witness can testify which emitted sample energies existed in the room. Exclusion can lose information but cannot invent it. |
| 3 | **Time = absolute integer milliseconds from container PTS** — never frame indices, never floats | Frame-index timing drifts on variable-frame-rate footage (`index × nominal_rate` diverges). Float time accumulates representation error and breaks both equality comparison and hashing. Enforced mechanically by manifest Rule 6. |
| 4 | **Fully offline / air-gapped, checksum-pinned models** | A silently-updated hub checkpoint changes outputs with no code change. Every weight's SHA-256 is verified before anything runs; all hub-download paths are disabled at import time (`HF_HUB_OFFLINE=1`). |
| 5 | **Append-only, hash-chained manifest, written *before* destructive ops** | Chain of custody. Any retroactive edit/delete/reorder is detectable by re-hashing. The record of an action exists durably *before* the action does. |
| 6 | **Minimal, validated, recorded human input** | Two clicks maximum; every operator override needs an identity + stated reason, recorded append-only. The human is in the loop but cannot be invisibly in the loop. |

## 0.4 Why not conventional deep-learning diarization (a sequence of refusals)

SPOVNOB's architecture is best understood as a series of explicit rejections:

- **End-to-end neural diarization (EEND) rejected** for two structural hazards. *Label permutation:*
  EEND assigns speaker labels that permute freely between chunks; stitching a coherent multi-file
  identity timeline out of permutation-invariant outputs is mathematically unstable and indefensible
  under cross-examination. *VFR tensor desync:* audio-visual EEND needs paired audio/video tensors,
  and pairing variable-frame-rate footage requires interpolation — violating constraint 2.
- **Clustering diarizers rejected** (embedding + spectral/AHC clustering). They answer "how many
  voices and which segments group," but still leave *which cluster is the target* to post-hoc
  inference. SPOVNOB replaces that inference with an **operator-witnessed visual identity anchor**.
- **WavLM-based speaker verification reuse rejected** on four audited grounds: the cached final-layer
  features are the wrong features (published SV heads consume a learnable weighted sum over *all*
  transformer layers); the turnkey checkpoint's backbone has diverged from the vanilla backbone; the
  accuracy delta is marginal at this operating point (two known speakers, same room, same mic, ≥45 s
  verified enrollment); and forensic defensibility favors the most independently replicated open
  speaker encoder in existence. WavLM had **zero consumers** under the pure-ECAPA architecture, so it
  was deleted entirely (and its `transformers` / `h5py` dependencies removed from `requirements.txt`).
- **Probability-style scoring rejected** after a concrete near-miss: early drafts treated ECAPA
  cosines as `P(Target)` with fixed `0.85 / 0.65 / 0.30` tiers. Cosines are **not calibrated
  probabilities**; the fix was per-session calibration (Layer 2) + strict naming discipline (the
  scores are `S_target` / `S_interviewer`, never `P(·)`).

**The remaining doctrine:** *frozen, widely-replicated encoders, surrounded by deterministic
arithmetic — cosines, medians, quantiles, duration-weighted means — and explicit, logged decision
rules. Nothing learns at runtime.* The Zero-Training Mandate is absolute: no model weight changes
anywhere, ever; "enrollment" is a weighted average an auditor can recompute by hand.

## 0.5 Execution environment (production, fixed)

- **Hardware:** NVIDIA RTX 6000 Ada (48 GB VRAM) · 44 cores / 88 threads · 512 GB DDR5 · 2 TB NVMe.
- **OS / runtime:** Ubuntu 22.04 LTS · Python 3.10.x · PyTorch-only stack (sole sanctioned exception:
  InsightFace's internal ONNXRuntime).
- **CUDA topology:** host driver 580 (CUDA 13.x). PyTorch wheels are `+cu121` (CUDA toolkit 12.1,
  bundled inside the venv). ONNXRuntime is the CUDA-12 build. The 13.x driver is backward-compatible;
  the only filesystem dependency ORT's CUDA provider needs (`libcufft.so.11`, `libcudart.so.12`,
  `libcublasLt.so.12`) is supplied by the `nvidia-cufft-cu12` / `nvidia-cuda-runtime-cu12` /
  `nvidia-cublas-cu12` pip packages and wired up by `env.sh` (Part 7, Part 12).
- **Resident Model Policy:** all five models are loaded **once** at batch start by the environment
  gate and held resident for the entire batch. There is no `torch.cuda.empty_cache()` anywhere and no
  load/unload state machine; combined footprint is far under 48 GB, and unload/reload cycles are both
  wasted time and a reproducibility risk surface.
- **Memory policy:** the entire batch's 16 kHz PCM (~1–2 GB for 15 files) is preloaded into RAM as raw
  `bytes` buffers at Layer 0. Every later layer slices these buffers; no audio is re-read from disk
  mid-pipeline.
- **Parallelism with determinism guardrails:** FFmpeg extraction fans out across CPU threads
  (`EXTRACT_WORKERS = 8`). Layer 1 is **strictly sequential in canonical file order** (the cumulative
  enrollment pool is order-dependent *by design*). Layers 2 and 3 currently run their GPU stages
  sequentially but are written through per-file `WorkerLog`s merged under the Canonical Merge Rule, so
  their records are byte-identical regardless of scheduling — fan-out is permitted without ever letting
  scheduling reach the evidence.
- **Session topology guarantee:** a batch *is* a session. Per-batch threshold calibration is therefore
  per-session by construction; no session-grouping machinery exists because none is needed.

## 0.6 Repository map (every project file)

```
Audio_Diarization/
├── session_manifest.py          (0a)  492 lines  append-only hash-chained audit log
├── environment_gate.py          (0b)  713 lines  determinism + vendoring gate + resident loaders
├── layer0_preprocessor.py       (1)   648 lines  PTS-true extraction + Silero speech map
├── layer1_enrollment/           (2)   package, ~2.4k lines total
│   ├── __init__.py                72   public exports
│   ├── params.py                 111   the EnrollmentParams table (schema v2)
│   ├── errors.py                  25   Layer1Error / Layer1ReclickError
│   ├── geometry.py                78   MAR, causal EMA, yaw suspension          [pure]
│   ├── window_machine.py         253   E_window capture FSM                     [pure]
│   ├── gates.py                  148   Triple Gate, M-Trap, variance, VAD lookup [pure]
│   ├── quality.py                 48   STRONG / MARGINAL / INSUFFICIENT          [pure]
│   ├── encoding.py               158   pooling/cosine/slice [pure] + ECAPA (torch lazy)
│   ├── vision.py                 194   frame PTS, YOLO/InsightFace scan (cv2/GPU lazy)
│   ├── enrollment.py            1129   the orchestrator (sequential, canonical order)
│   ├── selftest.py               361   stdlib-only self-test over every pure module
│   └── __main__.py                75   CLI entrypoint
├── layer2_tracker.py            (3)  1197 lines  calibrated sliding-window scanning
├── layer3_contamination.py      (4)   808 lines  OVD exclusion (NaN-only) + smoothing + output
├── pipeline_runner.py           (5)   403 lines  batch orchestrator + audit closure (entrypoint)
├── click_ui.py                       1872 lines  operator clicking web UI (Flask)
├── audit_visualizer.py               1367 lines  read-only forensic HTML dashboard (stdlib only)
├── diag_windows.py                    138 lines  throwaway window-machine diagnostic
├── env.sh                              11 lines  venv + LD_LIBRARY_PATH bridge (source before any run)
├── run.sh                              90 lines  one-command batch runner
├── requirements.txt                   pinned dependency stack + the five compatibility flags
├── README.md / UBUNTU_SETUP_GUIDE.md / Ubuntu_Setup_Gotchas.md / implementation_order.md
├── CLICK_UI_IMPLEMENTATION_PLAN.md
└── SPOVNOB_MASTER_REFERENCE.md  ← THIS FILE (replaces the four old docs)
```

## 0.7 Module dependency graph (strictly acyclic; arrows = importer → imported)

```
session_manifest ◄ environment_gate ◄ layer0_preprocessor ◄ layer1_enrollment ◄ layer2_tracker ◄ layer3_contamination ◄ pipeline_runner
   (0a, stdlib)      (0b)               (1)                   (2, package)         (3)              (4)                     (5)
```

Every module from 0b onward begins with `import environment_gate` as its **first import**, because
that import fixes the process environment (CUDA determinism env var + offline switches) *before*
anything else can observe it. No module imports a later one. Every module from 0a onward ships a
**stdlib-only self-test** runnable on a bare Python 3.10 with zero installed packages — the injectable
seams that make that possible are part of the architecture, not test scaffolding (Part 13).

`audit_visualizer.py` is deliberately **outside** the graph: it imports nothing from the pipeline
(not even `session_manifest`) so a read-only forensic tool never perturbs process state and runs on a
plain laptop. `click_ui.py` and `diag_windows.py` *do* import pipeline modules (by design — parity
with production), but write no manifest entries.

## 0.8 The three time coordinate frames (integer milliseconds everywhere)

All timing lives in exactly one of three integer-ms frames, and every conversion site is fixed:

| Frame | Zero point | Conversion |
|---|---|---|
| **Data-relative ms** | first audio sample of one file's extraction | — |
| **Local PTS ms** | the file's own container clock | `local = data_relative + audio_start_pts_ms` |
| **Global session ms** | start of the batch timeline | `global = local + file_offset_ms` |

- `audio_start_pts_ms` is the audio stream's `start_time` from ffprobe — containers routinely start
  audio at a small nonzero PTS (e.g. 23 ms); ignoring it would shift every downstream timestamp.
- `file_offset_ms` is the cumulative duration of all preceding files in canonical order.
- At 16 kHz, **1 ms = 16 samples exactly**, so `ms × 16` is exact integer arithmetic in both
  directions; `ms_from_samples` floors (`samples * 1000 // 16000`).
- Layer 1 operates in **local PTS** (its PCM slicer subtracts `audio_start_pts_ms`). Layer 2 plans
  and scores windows in **data-relative ms** (scorer slices with start offset 0), then logs local and
  global. Layer 3's overlap regions are produced in **local PTS** (the provider adds
  `audio_start_pts_ms`). Manifest Rule 6 enforces integerness at every payload boundary, so a float
  can never leak into the timeline.

## 0.9 End-to-end flow at a glance

```
Pre-flight: manifest init → environment gate (determinism + vendoring + load 5 resident models)
Layer 0:    ffmpeg PTS-true 16 kHz mono extraction (parallel) → Silero speech MAP (non-destructive) → RAM preload
Layer 1:    operator clicks → YOLO+InsightFace lock F_target → window machine finds speaking windows
            → E_seed → anti-profile (Track B auto + Track C click, M-Trap) → Triple Gate (A/B/C)
            → duration-weighted pool → E_composite (recomputed per video) → quality states → FREEZE
            (alternate audio-anchored path for bearded / unreliable-MAR subjects)
Layer 2:    per-session threshold calibration → 5 s/1 s sliding window ECAPA cosine scoring (skip <20% speech)
            → median-pool onto 1 s blocks → tier (HIGH/MEDIUM/SUB/REJECT) + margin demotion
            → edge-trim (trim-only) → activity ratio + drift → single authoritative pass → hashed output
Layer 3:    PyAnnote OVD per full file → NaN-only whole-block voiding on overlap → bridge clean gaps <400 ms
            (overlap-free + dominance guard) → slice ORIGINAL pcm → WAV + sidecar, hashed → final output
Runner:     chain all → write summary → re-verify the entire hash chain from disk
```

**One-sentence mental model:** *Layer 0 says **where** speech is; Layer 1 learns **who** the target
is (by sight) and freezes an acoustic fingerprint; Layer 2 finds **where the target speaks**; Layer 3
removes **anything overlapped** — and every byte out is an unmodified slice of what came in.*

---

# Part 1 — The Substrate

Two modules underpin everything: the **session manifest** (the chain of custody as a data structure)
and the **environment gate** (the startup proof the runtime is the one the architecture demands).

## 1.1 `session_manifest.py` — Module 0a (492 lines)

Pure standard library. Imports nothing from any other SPOVNOB module and must never import torch,
numpy, or any framework. Implemented *first* (the "FLAG 5 ruling"): the gate and Layers 0–3 all need
to write the manifest, and a manifest module that imported a later module would break the acyclic
import rule.

### 1.1.1 Module-level constants and helpers

- `SCHEMA = "spovnob-manifest-v1"` — stamped into every entry.
- `GENESIS_SHA256 = "0" * 64` — the chain's root `prev_sha256` (64 zero hex chars).
- `_HASH_CHUNK_BYTES = 1024 * 1024` — streaming read size for file hashing.

**`canonical_json(obj) -> str`** — the *only* serializer the module uses, with four locked flags,
each load-bearing:
```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
```
`sort_keys` kills dict-insertion-order variance; fixed `separators` kill whitespace variance;
`ensure_ascii` makes the byte stream pure ASCII regardless of platform; `allow_nan=False` makes IEEE
NaN/Inf *unserializable* — which is exactly why Layer 3's "NaN block" is the **string** `"NaN"`, never
a float NaN (a float NaN also breaks equality and hashing, so the serializer refusing it is
defense-in-depth).

**Hash utilities:** `sha256_hex(bytes)`, `sha256_of_obj(obj)` (= sha256 of `canonical_json(obj)` UTF-8
bytes — the cross-run-comparable quantity), and `sha256_of_file(path)` (streaming, 1 MiB chunks; used
for model weights, worker logs, output corpora).

**`validate_time_fields(obj, _path="payload")`** — **Rule 6**, recursive. Every dict key ending in
`_ms`, at any nesting depth (recurses through dicts, lists, tuples), must hold a Python `int`. The
check is `isinstance(value, bool) or not isinstance(value, int)` — `bool` is rejected **explicitly**
because `bool` subclasses `int` in Python, so without the explicit clause `"start_ms": True` would
validate. The rule is **suffix-driven**, which is why payload *shape* matters (see Layer 0's nested
`silero_segments` design in §2.4).

**`_utc_now_iso_ms()`** — wall-clock UTC formatted to millisecond precision (`...Z`). Used **only** in
the `audit` block, never in a payload.

### 1.1.2 The `Operation` vocabulary

A class of canonical operation-name string constants (modules may extend, but these cover everything
named in the architecture): `batch_init`, `model_checksum`, `determinism_check`,
`parameter_modified`, `enrollment_vector`, `enrollment_discard`, `calibration`, `video_gap`,
`drift_notice`, `warning`, `blocking_halt`, `destructive_op`, `output_hash`, `worker_log_merged`. (See
Part 11 for the full vocabulary including layer-specific operations.)

### 1.1.3 Exception hierarchy

`ManifestError` (base) → `ManifestTimeError` (a `*_ms` field was not an int — Rule 6), `ManifestChainError`
(hash/seq/content verification failed), `ManifestLockError` (a second writer tried to open the
manifest — the single-writer rule).

### 1.1.4 Entry anatomy and the payload/audit split

Each line of the append-only JSON-Lines file is one entry:
```json
{
  "schema":         "spovnob-manifest-v1",
  "seq":            17,
  "operation":      "calibration",
  "payload":        { "...deterministic content only..." },
  "payload_sha256": "sha256 of canonical-JSON payload",
  "prev_sha256":    "entry_sha256 of the previous line (or 64 zeros = GENESIS)",
  "audit": { "timestamp_utc": "2026-06-12T03:41:07.221Z",
             "operator_id": "operator-7", "stated_reason": null },
  "entry_sha256":   "sha256 of this entry minus this field"
}
```
**The decisive design choice is the payload/audit split.** Payloads contain *only* deterministic
content — no wall-clock time, no hostnames, no operator identity — so `payload_sha256` is
**bit-reproducible across re-runs**: an auditor replaying the batch months later produces the same
payload-hash stream. Wall-clock time and operator identity live exclusively in `audit`, which exists
for custody (when/who/why). `entry_sha256` seals the *whole* entry including the audit block and the
chain link, so chain hashes are run-specific while payload hashes are the cross-run comparable
quantity. *What happened* is reproducible; *when and by whom* is tamper-evident.

### 1.1.5 `SessionManifest` — the writer

- **`__init__(path, operator_id=None, verify_on_open=True)`** opens the file in append mode (`"a+"`,
  never truncated) and immediately takes an exclusive **`fcntl` advisory lock**
  (`LOCK_EX | LOCK_NB`); a second concurrent writer fails fast with `ManifestLockError`. If
  `verify_on_open` (default), it re-walks and re-verifies the entire existing chain via
  `_resume_from_verified_chain()` before resuming the sequence — a crashed run cannot be silently
  continued atop a corrupted log. (`_resume_from_tail()` is the unverified fast path used only with
  `verify_on_open=False`.)
- **`append(operation, payload, operator_id=None, stated_reason=None)`** is the heart. It (1) rejects
  non-dict payloads; (2) runs `validate_time_fields(payload)` (Rule 6); (3) builds the entry with
  `seq`, `payload_sha256`, `prev_sha256`, and the `audit` block; (4) computes `entry_sha256` over the
  entry-minus-that-field; (5) writes the canonical line; (6) **`flush()` then `os.fsync()`** before
  returning; (7) advances `_seq` and `_prev_sha256`. *Returning from this method is the durability
  guarantee Rule 7 relies on.*
- **`guard_destructive(action, payload, ...)`** — **Rule 7**. Appends a `destructive_op` entry and
  returns only after fsync. The caller must invoke this and receive the entry back **before**
  performing the irreversible action. The inverse ordering (act, then log) is how chain-of-custody
  gaps are born.
- **`record_parameter_change(parameter, default_value, operator_value, modified_by, stated_reason)`** —
  the "Operator Threshold Manifest Format": logs `parameter_modified` with the operator's identity and
  stated reason carried in the audit block.
- **`close()`** releases the lock (`LOCK_UN`) and closes; context-manager methods wrap this.

### 1.1.6 `verify_chain(path)` — the auditor's entry point (static)

Re-walks the full manifest, and for every line re-derives and checks four things:
1. `entry_sha256` equals the hash of the entry minus that field;
2. `payload_sha256` equals the hash of the payload;
3. `prev_sha256` equals the previous entry's seal (chain link from `GENESIS`);
4. `seq` equals the line's position (0-based, dense).

The first inconsistency raises `ManifestChainError(line_number)`. What each tampering class trips:

| Tampering | Detected by |
|---|---|
| Edit any byte of any payload | checks 1 and 2 on that line |
| Edit the audit block (e.g. backdate a timestamp) | check 1 on that line |
| Delete a line | check 3 on the next line (and check 4 thereafter) |
| Reorder lines | checks 3 and 4 |
| Insert a forged line | check 3 on the following original line |
| **Truncate the tail** | **not** detectable by the chain alone — a valid prefix is a valid chain. Mitigations: the runner's terminal `pipeline_complete` entry is the designated last operation; the runner re-verifies and records the entry count after close; recommended practice retains the final `entry_sha256` off-box (converting truncation + full-rewrite into detectable attacks). |

### 1.1.7 `WorkerLog` and the Canonical Manifest Merge Rule

Parallel (or parallelizable) stages never write the main manifest directly. Each per-file worker
writes a **`WorkerLog`** — plain JSON-Lines, *not* hash-chained (it is an intermediate artifact),
each record carrying its canonical sort fields:
```json
{"file_index": 3, "start_ms": 127000, "operation": "layer2_block", "payload": {...}}
```
`WorkerLog.append(operation, payload, start_ms=-1)` validates that `start_ms` is an int (and Rule-6
validates the payload). **`start_ms = -1` marks file-level records** (e.g. activity-ratio summaries),
which deliberately sort *before* all timed records of their file.

**`merge_worker_logs(manifest, log_paths)`** reads every record from every log and sorts by the
four-tuple
```
(file_index, start_ms, operation, sha256(canonical_json(payload)))
```
— a **deterministic total order independent of worker count, scheduling, and arrival time** (the
payload hash as final tiebreaker makes the order well-defined even for records identical in the first
three fields). It then appends the sorted records through the single manifest writer and finishes
with a `worker_log_merged` summary entry carrying each source log's file SHA-256 and record count —
so the intermediate artifacts are integrity-pinned even though they aren't chained. *Consequence,
verified in the self-test: two workers finishing in any order produce a byte-identical payload stream
in the merged manifest.*

### 1.1.8 Self-test (`python3 session_manifest.py`)

Stdlib-only. Builds a temp manifest, asserts a float `_ms` is rejected (Rule 6), drives two workers
finishing out of order and asserts the merged order is `[(0,9000),(1,2000),(1,5000)]`, exercises
`guard_destructive`, then flips one byte (`"HIGH"`→`"MEGA"`) and asserts `verify_chain` raises
`ManifestChainError`.

## 1.2 `environment_gate.py` — Module 0b (713 lines)

Imports only Module 0a. Every entrypoint imports this module **first** and calls `run_gate()` before
touching any model. The module is the **source of all four CUDA determinism constants**.

### 1.2.1 Step 0 — import-time environment fix (before any heavy import)

At module top, *before* `import torch` is even possible, it sets three process env vars:
```python
DETERMINISM_ENV = {
  "CUBLAS_WORKSPACE_CONFIG": ":4096:8",   # CUDA determinism constant (must precede CUDA context)
  "HF_HUB_OFFLINE": "1",                  # air-gap switch (must precede any HuggingFace import)
  "TRANSFORMERS_OFFLINE": "1",            # air-gap switch
}
```
Only after this does it do its stdlib-only imports (`torch` is **deliberately not imported at module
level** — it is imported lazily inside `run_gate()` / loaders, which is what keeps the self-test
runnable on a bare Python 3.10).

### 1.2.2 Architectural constants (manifest-logged by `run_gate`)

```python
ECAPA_BATCH_WINDOWS = 256        # fixed inference batch — FP reduction order is part of the function
VISUAL_BATCH_FRAMES = 32         # fixed YOLO batch
TORCH_NUM_THREADS   = 8          # fixed intra-op threads — CPU reduction order
GLOBAL_SEED         = 20260611   # defense-in-depth; nothing samples at runtime
INSIGHTFACE_DET_SIZE = (640, 640)
PYANNOTE_OVD_HYPERPARAMS = {"min_duration_on": 0.0, "min_duration_off": 0.0}
EXPECTED_CUDA_VERSION = "12.1"
EXPECTED_PYTHON_PREFIX = "3.10."
HASHES_FILENAME = "expected_hashes.json"
HASHES_SCHEMA   = "spovnob-model-hashes-v1"
```
`PINNED_VERSIONS` is a dict of **23 packages** at exact pins (mirrors `requirements.txt`).
`FORBIDDEN_IMPORTS = ("pyannote.audio.pipelines.speaker_verification",)` is the FLAG 1 policy
constant. `REQUIRED_MODEL_DIRS` maps the five model keys to their store directory names:
`silero-vad`, `speechbrain-spkrec-ecapa-voxceleb`, `yolov8`, `insightface`,
`pyannote-segmentation-3.0`.

`ResidentModels` is the dataclass registry holding `device`, `silero`, `ecapa`, `yolo`,
`insightface`, `ovd_pipeline`, and `loaded_names` — loaded once, never unloaded or replaced.

### 1.2.3 The four CUDA determinism constants, mechanism by mechanism

1. **`CUBLAS_WORKSPACE_CONFIG=":4096:8"`** (set at import time) — cuBLAS picks GEMM reduction
   strategies (e.g. split-K) partly from available workspace; different strategies sum partial
   products in different orders, and FP addition is **not associative**. Fixing the workspace pins
   the strategy and therefore the summation order. It must exist before the CUDA context is created,
   which is why it is module-level.
2. **`torch.use_deterministic_algorithms(True)`** — forces deterministic kernels and *raises* on ops
   with no deterministic implementation rather than silently varying.
3. **`torch.backends.cudnn.deterministic = True`** — restricts cuDNN to deterministic convolution
   algorithms.
4. **`torch.backends.cudnn.benchmark = False`** — disables cuDNN autotuning (which times candidate
   algorithms at runtime and would pick by transient machine state). Off ⇒ the algorithm choice is a
   pure function of the operation shape.

Constants 2–4 are applied inside `enforce_torch_determinism()`, alongside
`torch.set_num_threads(8)`, `torch.manual_seed(20260611)`, float32 throughout (no AMP, no TF32), and
`torch.no_grad()` around every forward pass in every module.

### 1.2.4 The gate sequence (`run_gate`) — nine steps, fail-closed

`run_gate(model_store, manifest, load_models=True)` runs the checks in order; any failure records a
`blocking_halt` entry **first** (via the `_halt` helper) and then raises `EnvironmentGateError`:

| Step | Function | Halts on |
|---|---|---|
| 1 | `check_runtime_platform` | non-Linux OS; Python not `3.10.x`; any of the three import-time env vars drifted |
| 2 | `check_ffmpeg` | ffmpeg **or** ffprobe missing/unusable on PATH (both shelled out to in Layers 0–1) |
| 3 | `check_versions` | any of the 23 pinned packages not at its exact pin |
| 4 | `check_forbidden_imports` | `pyannote.audio.pipelines.speaker_verification` present in `sys.modules` |
| 5 | `verify_model_store` | any weight file missing, **unexpected**, or hash-mismatched |
| 6 | `enforce_torch_determinism` | `torch.version.cuda != "12.1"`; CUDA unavailable |
| 7 | `check_onnxruntime_cuda` | CUDAExecutionProvider missing, probe model absent, or a real session fell back to CPU |
| 8 | `gpu_determinism_selftest` | the two workload hashes differ |
| 9 | `load_resident_models` | any loader failure (skipped if `load_models=False`) |

Each successful check appends a `determinism_check` (or `model_checksum`) entry, so the entire gate
is auditable.

### 1.2.5 Three steps that encode lessons validated in practice

- **Step 4 (FLAG 1).** PyAnnote's SpeechBrain speaker-verification wrapper imports the
  `speechbrain.pretrained` module that SpeechBrain 1.x removed (pyannote #1661/#1677). SPOVNOB never
  uses that path — Layer 3 uses `OverlappedSpeechDetection` only. The check runs *before* the resident
  loader; pyannote's own `__init__` later pulls the module into `sys.modules` transitively (guarded by
  pyannote's optional-backend try/except, never instantiated), so post-load presence is *expected*.
  What the gate actually enforces is the realistic violation vector: a **direct** import by SPOVNOB
  code.
- **Step 7 (FLAG 2).** `onnxruntime-gpu==1.17.1` exists under the *same version number* on default
  PyPI (built for CUDA 11.8) and on the Microsoft CUDA-12 feed. On this CUDA-13 host the wrong wheel
  doesn't crash at import — InsightFace **silently falls back to CPU**, and crucially
  `ort.get_available_providers()` *still returns* `CUDAExecutionProvider` (registered ≠ loadable). So
  the hardened check goes further: it instantiates a real `ort.InferenceSession` on
  `insightface/models/buffalo_l/det_10g.onnx` (the exact model InsightFace uses), requests
  `["CUDAExecutionProvider"]`, and asserts `session.get_providers()[0] == "CUDAExecutionProvider"`. Failure
  modes → halt codes: provider not registered → `onnxruntime_cuda_provider_missing`; probe model
  absent → `onnxruntime_cuda_probe_missing`; session built on CPU or raised →
  `onnxruntime_cuda_provider_inactive`. On success it records `active_session_provider` and
  `probe_model`. This exact failure was reproduced on the bench (`libcublasLt.so.11 not found` → CPU
  fallback; the old check passed, the new one catches it).
- **Steps 5 & 9 (Model Vendoring Mandate).** On the staging box, `freeze_model_hashes(store)` runs
  **once**: it refuses to freeze if any of the five model dirs is missing/empty, hashes every regular
  file under the store (sorted `rglob` walk, the registry file itself excluded) via `_walk_store_files`,
  writes `expected_hashes.json` as canonical JSON, and `chmod 444`s it (re-freezing requires a
  deliberate manual delete — intentional friction; Gotcha #4). On the air-gapped box,
  `verify_model_store` re-hashes everything and computes a three-way set difference: `missing`,
  `unexpected`, `mismatched`. **`unexpected` is load-bearing** — a smuggled extra file halts the gate
  even though no expected file changed (the store is closed-world). Every verified file gets its own
  `model_checksum` entry.

### 1.2.6 The ~10-second GPU determinism self-test

`gpu_determinism_selftest()` is the startup *proof* (a measurement, not an assertion) that the
constants are actually in force on *this* machine/driver/now:
```python
generator = torch.Generator(device="cuda").manual_seed(20260611)
x = randn(2048, 2048); w = randn(2048, 2048); kernel = randn(8, 1, 7, 7)   # float32, seeded
for _ in range(200):
    x = torch.tanh(x @ w) * 0.5                                            # cuBLAS GEMM chain
feature_map = conv2d(x.view(1, 1, 2048, 2048), kernel, padding=3)          # cuDNN
payload = first 65,536 elems of x  +  first 65,536 of feature_map
return sha256(payload)
```
The workload runs **twice from scratch**; the two SHA-256s must match or the gate halts
(`gpu_determinism_failure`). The 200-iteration GEMM chain amplifies any reduction-order divergence
exponentially (a single differing bit in iteration 1 avalanches); `tanh(·)·0.5` keeps values bounded
so the chain cannot saturate and mask divergence; the conv2d exercises the cuDNN
algorithm-selection path that constants 3/4 govern. The passing hash (`gpu_workload_checksum`) is
recorded, converting an *intra-machine* proof into an *inter-machine* check. What it cannot prove:
that a *different* GPU architecture would produce the same bits — deterministic-per-machine is the
guarantee; cross-architecture identity is checked (via the recorded hash), not assumed.

### 1.2.7 `load_resident_models` — the five loaders (local-only by construction)

1. **Silero VAD** — `torch.jit.load(store/"silero-vad"/"files"/"silero_vad.jit", map_location="cpu").eval()`.
   CPU-resident TorchScript, loaded from the vendored snapshot (commit-pinned
   `915dd3d639b8333a52e001af095f87c5b7f1e0ac`, *not* the upstream `v4.0` tag, which was observed to
   move). `torch.hub` is bypassed entirely.
2. **ECAPA-TDNN** — `EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir, run_opts={"device":"cuda"})`
   from the local SpeechBrain dir.
3. **YOLOv8m** — `YOLO(store/"yolov8"/"yolov8m.pt")`.
4. **InsightFace** — `FaceAnalysis(name="buffalo_l", root=store/"insightface", providers=["CUDAExecutionProvider"])`
   then `.prepare(ctx_id=0, det_size=(640,640))`. CUDA-only: CPU fallback is forbidden (FLAG 2 already
   proved the provider live).
5. **PyAnnote OVD** — `Model.from_pretrained(store/"pyannote-segmentation-3.0"/"pytorch_model.bin")`
   wrapped in `OverlappedSpeechDetection`, instantiated with `{"min_duration_on":0.0,"min_duration_off":0.0}`
   and `.to(cuda)`.

A final `determinism_check` entry records the loaded model names, device, det size, OVD hyperparams,
and the `resident_for_entire_batch_no_unload` policy.

### 1.2.8 CLI and self-test

`_main` supports `--run` (full gate), `--freeze-hashes` (staging box, run once), and `--selftest`
(stdlib-only). The self-test (`_selftest_stdlib`) fabricates a fake model store, freezes + verifies it
(pass), tampers a weight (`model_checksum_failure` with `mismatched == ["yolov8/weights.bin"]`),
smuggles an extra file (halt), exercises `check_versions` with injected metadata (pass then halt), and
`check_ffmpeg` with injected probes (pass then `ffmpeg_missing` with `missing == ["ffprobe"]`) — all
while asserting `torch` was never imported.

---

# Part 2 — Layer 0: PTS-True Extraction & the Speech Map (`layer0_preprocessor.py`, 648 lines)

**Purpose:** convert raw videos into PTS-true 16 kHz mono audio plus a Silero speech segment map,
preloaded into RAM for the whole batch. No identity decisions, no signal modification beyond the
mandated 16 kHz mono resample. Audio is **never** zeroed, padded, trimmed, or interpolated in storage.
CPU-only module (relies on the gate's import-time env fixing and the fixed torch thread count for
Silero reproducibility).

## 2.1 Constants

```python
SAMPLE_RATE = 16000              # architectural — the whole pipeline is 16 kHz
SILERO_WINDOW_SAMPLES = 512      # 32 ms windows (Silero v4 @ 16 kHz)
SILERO_THRESHOLD = 0.50          # window is speech iff prob >= this
SILERO_MIN_SPEECH_MS = 250       # drop speech runs shorter than this
SILERO_MIN_SILENCE_MS = 100      # merge speech runs across gaps under this
SILERO_SPEECH_PAD_MS = 30        # widen kept segments by this per side
EXTRACT_WORKERS = 8              # parallel ffmpeg extraction threads
OP_LAYER0_FILE = "layer0_file"
```

## 2.2 Pure time / parsing helpers (stdlib-only, self-tested)

- **`ms_from_samples(n) -> int`** = `(n * 1000) // SAMPLE_RATE` (integer floor).
- **`decimal_seconds_to_ms(value: str) -> int`** = `int((Decimal(value) * 1000).quantize(Decimal(1),
  rounding=ROUND_HALF_EVEN))`. Parsed *as a string* into `Decimal` (exact, no binary-float
  re-rounding); **banker's rounding** is bias-free under accumulation. Examples from the self-test:
  `"0.023220"→23`, `"2.0005"→2000`, `"2.0015"→2002` (both ties round to the even ms).
- **`parse_probe(probe_json) -> ProbeInfo`** — a pure function over an ffprobe JSON document (testable
  without the binary). Finds the first audio stream (raises `Layer0Error` if none). Reads
  `start_time` → `audio_start_pts_ms` (missing/`"N/A"` → `0` + `audio_start_missing=True`). Computes
  `vfr_suspected` by parsing the video stream's `avg_frame_rate` and `r_frame_rate` as exact
  `Fraction`s and comparing them — inequality is the standard VFR fingerprint (`"30000/1001"` parses
  exactly; string comparison would false-positive on equivalent spellings, float on precision). VFR is
  **diagnostic only**: audio processing is index-free, so VFR cannot corrupt the audio timeline.

## 2.3 `segments_from_window_probs` — the non-destructive speech map

Converts per-window speech probabilities to integer-ms segments (data-relative; the caller adds
`audio_start_pts_ms`) in a **fixed four-step order**:
```
1. threshold:  window is speech iff prob >= 0.50              (SILERO_THRESHOLD)
2. merge:      join speech runs separated by silence < 100 ms (SILERO_MIN_SILENCE_MS)
3. drop:       discard runs shorter than 250 ms — measured BEFORE padding
4. pad:        widen each survivor by 30 ms per side, clamp to [0, audio_end], re-merge overlaps
```
**Order is load-bearing:** pad-before-drop would inflate sub-250 ms blips over the survival threshold.
Worked self-test cases: 10 speech windows (320 ms) → `(0,350)`; 5 speech windows (160 ms) → dropped
(`[]`); a 64 ms gap (< 100 ms) merges two runs → `(0,606)`; a 128 ms gap (≥ 100 ms) keeps them apart →
`[(0,286),(354,640)]`. This single map has **four consumers**: Layer 1 window starts + Gate A
coverage, Layer 2's skip rule, and the activity-ratio denominator.

**`read_wav_pcm16(path)`** re-validates the 16 kHz/mono/16-bit contract on read-back via the stdlib
`wave` module (any mismatch → `Layer0Error`) and returns `(num_samples, raw int16 bytes)`.

## 2.4 Data structures

- **`FileAudio`** — one file's preloaded audio + PTS metadata: `file_index`, `source_path`,
  `wav_path`, `source_sha256`, `wav_sha256`, `num_samples`, `duration_ms`, `audio_start_pts_ms`,
  `audio_start_missing`, `vfr_suspected`, `file_offset_ms`, the in-RAM `pcm` bytes, and
  `silero_segments_local_ms`. Conversion helpers: `to_global_ms(local) = file_offset_ms + local`;
  `sample_to_local_pts_ms(i) = audio_start_pts_ms + ms_from_samples(i)`.
- **`BatchAudio`** — a list of `FileAudio`, plus `total_speech_ms()`.
- **`layer0_file_payload(file_audio)`** — the **pure** payload builder (self-testable without ffmpeg).
  Critically, it nests segments as `[{"start_ms": s, "end_ms": e}, ...]` under key `silero_segments`
  (not a bare pair-list under a `*_ms` key) **specifically so Rule 6 actively validates every
  boundary**. This is the fix shipped in commit `7dc3daa` for a bug where the original
  `silero_segments_local_ms` list would have died with `ManifestTimeError` at the first real Layer 0
  write (the zero-pip self-tests couldn't reach that path — it needs ffmpeg + Silero). The general
  doctrine: *when a validation rule and a data shape collide, reshape the data into the rule's
  coverage, never around it.* The self-test even corrupts one boundary to a float and asserts the
  validator catches it.

## 2.5 Subprocess wrappers and Silero inference

- **`extract_audio(source, dest)`** runs, with a fully explicit, auditable argument list:
  `ffmpeg -hide_banner -nostdin -y -i <source> -vn -map 0:a:0 -acodec pcm_s16le -ar 16000 -ac 1 -f wav <dest>`.
  `-map 0:a:0` selects the first audio stream **deterministically** (no ffmpeg default-stream
  heuristics); `-vn` no video decode. Returns the exact command list (manifest-recordable).
- **`run_ffprobe(path)`** runs `ffprobe -v error -print_format json -show_streams -show_format <path>`.
- **`ffmpeg_version()`** captures the first version line (logged in `batch_init`).
- **`silero_window_probs(silero_model, pcm)`** converts the int16 buffer to float32 `/ 32768.0`,
  `reset_states()` (stateful model — per-file reset is itself a determinism requirement), then iterates
  **512-sample windows in order**, zero-padding only the final partial window **for inference** (the
  stored buffer is untouched), producing one probability per window under `torch.no_grad()`.

## 2.6 `preprocess_batch` — the entrypoint

`preprocess_batch(manifest, video_paths, work_dir, silero_model, extract_workers=8)`:
1. **Canonical file order** = lexicographic full-path sort.
2. Missing-file check → blocking halt listing the absent paths.
3. Source hashing → a `batch_init` entry recording canonical order, ffmpeg version, sample rate,
   worker count, all five Silero params, and each source's SHA-256.
4. **Parallel extraction** (`ThreadPoolExecutor`, 8 workers): `_extract_one` probes, extracts, reads
   back PCM, and builds a `FileAudio` with no manifest IO. `pool.map` preserves index order.
5. **Sequential stage** in canonical order: accumulate `file_offset_ms` (file *k*'s offset = sum of
   durations of files 0..k−1); run Silero and convert to `silero_segments_local_ms` (adding
   `audio_start_pts_ms` to each pair); write a `video_gap` entry at each file boundary after the first;
   write one `layer0_file` entry per file.

The split is the parallelism doctrine applied: the expensive, side-effect-free stage fans out;
everything that writes the record runs in canonical order, so the manifest's payload content is
identical no matter how the extraction pool was scheduled.

---

# Part 3 — Layer 1: Visual-Anchored Enrollment (`layer1_enrollment/` package, ~2.4k lines)

This is the most intricate layer. Its job: produce the **frozen** Target Enrollment Profile
`E_composite` and interviewer anti-profile `E_anti` from operator-anchored, triple-validated,
visually-confirmed speech windows — using zero training, only frozen models and deterministic
arithmetic. It also implements an **audio-anchored** alternate path for bearded / unreliable-MAR
subjects (schema v2). The package is "one logical module" split into pure leaves and two GPU-facing
modules.

## 3.1 Philosophy: replace identity *inference* with identity *witness*

The hard problem in diarization is not "where is speech" but "*whose* speech." Statistical systems
answer by clustering + post-hoc assignment — inference an auditor takes on faith. SPOVNOB answers with
a **witnessed anchor**: the operator clicks the target *while watching them speak*; the clicked face
becomes a biometric lock (`F_target`, an ArcFace embedding); and from then on **audio enters the
enrollment pool only when the locked face is visibly producing speech at the same PTS, corroborated by
VAD**. Everything downstream — `E_seed`, `E_composite`, `E_anti` — is **arithmetic over frozen encoder
outputs** (duration-weighted means and cosines), recomputable by hand from the persisted per-window
WAVs and vectors.

## 3.2 `params.py` — the `EnrollmentParams` table (schema v2, 111 lines)

A frozen dataclass; the full set is written to the manifest at Layer 1 init (so any change is in the
chain of custody). `PARAM_SCHEMA_VERSION = 2` (bumped when the audio-anchored path was added).
`manifest_payload()` returns `{"schema_version": 2, **asdict(self)}`. The complete value list is in
Part 10; the key fields and their *why*:

- **Identity/ReID:** `face_reid_threshold = 0.40` (ArcFace cosine to keep the target lock);
  `reid_warning_floor = 0.50` (guardrail 6 running-mean warning).
- **MAR hysteresis:** `mar_on = 0.15` (lips clearly open), `mar_off = 0.10` (lips closing) — the 0.05
  dead band stops boundary chatter; both bench-recalibrated 2026-06-12 for the corrected outer-lip
  formula's range (~0.10–0.25, resting ~0.13).
- **Window machine:** `plosive_ms = 500`, `vad_tol_ms = 50`, `min_enroll_len_ms = 2000`,
  `seed_min_ms = 3000` (E_seed minimum; **no maximum** — supersedes the legacy "3–8 s" wording).
- **Triple Gate:** `int_lips_closed_frac = 0.80` (Gate A), `threshold_target = 0.70` (Gate B),
  `threshold_anti = 0.50` (Gate C ceiling), `margin_minimum = 0.15` (Gate C margin),
  `gate_a_vad_min_coverage = 0.50` (Gate A VAD coverage — the implementation's quantification of "Silero
  confirms speech").
- **Anti-collection:** `mtrap_sim_max = 0.60` (Track B M-Trap), `trackb_window_ms = 2000`,
  `trackb_min_spacing_ms = 2000`, `click_overlap_max_frac = 0.20` (guardrail 1 visual proxy).
- **Sanity:** `anti_contam_warning = 0.45`, `anti_contam_halt = 0.60`, `pool_var_warning = 0.05`.
- **Geometry/vision:** `yaw_max_degrees = 35.0`, `ema_span = 5`, the landmark index tuples
  `upper_inner_lip = (71, 63, 68)`, `lower_inner_lip = (62, 54, 57)`, `mouth_width_pair = (52, 61)`
  (bench-corrected — see §3.3), `insightface_min_det_score = 0.50` (guardrail 5), `yolo_min_conf = 0.30`,
  `silence_stride = 1` (off).
- **Encoding:** `encode_max_ms = 60000` (single-pass sanity cap), `encode_overlap_ms = 2000`.
- **Quality:** `strong_ms = 45000`, `strong_ms_no_anti = 60000`, `marginal_ms = 20000`,
  `variance_high = 0.05`.
- **Audio-anchored (v2):** `audio_anchor_accept_sim = 0.78`, `audio_anchor_collect_sim = 0.55`,
  `audio_anchor_consistency_min = 0.65`, `audio_solo_min_ms = 2000`, `audio_solo_face_max_others = 0`.

`errors.py` (25 lines) defines `Layer1Error` and its subclass `Layer1ReclickError` (an operator click
failed validation — the run stops and asks for a corrected click; recorded as a `warning` entry
before raising).

## 3.3 `geometry.py` — MAR, EMA, yaw suspension (78 lines, pure)

**MAR (Mouth Aspect Ratio)** — `compute_mar(landmarks, params)`:
```
vertical = fsum([ d(71,62), d(63,54), d(68,57) ]) / 3       # mean inner-lip vertical gap
MAR      = vertical / d(52, 61)                              # divided by mouth width
```
Returns `None` if the landmark array is too short (`len <= max index needed`) or the width is
degenerate (`<= 1e-6`). Division by width makes MAR **face-scale invariant** (a face twice as close
doubles numerator *and* denominator). `euclidean` is `math.hypot`; the vertical average uses
`math.fsum`.

**The landmark-index story (Gotcha #7, bench-validated 2026-06-12 on NT-clip27 at 42000 ms).**
InsightFace `buffalo_l` 2d106det provides the **outer lip contour only** (indices 52–71); indices
72–86 are **nose**, not inner lip. The MAR therefore uses inner-edge points of the *outer* contour —
the bottom rim of the upper arc (`upper_inner_lip = 71, 63, 68`) and the top rim of the lower arc
(`lower_inner_lip = 62, 54, 57`), which are nearly vertically aligned (Δx < 8 px each), divided by the
corner-to-corner width `d(52, 61)` (~80 px). The **original doc indices `(52,53,54)/(61,62,63)` were
wrong**: `d(52,61)` is a horizontal corner distance that ended up in the numerator, pinning MAR
≈ constant at 0.44–0.57 regardless of mouth state (the window machine could never close and produced a
single 66-second mega-window). Corrected range: **~0.10 (closed) → ~0.25 (open)**, resting ~0.13.
Outcome of the fix on the NT-clip batch: **+3 enrollment windows, +7 250 ms of clean audio.**

**`CausalEMA`** — a 5-frame causal EMA, `alpha = 2/(span+1) = 1/3`, update `v ← αx + (1−α)v`,
**pre-seeded with the first observed value** (`self.value is None` → seed). A zero-init EMA needs ~4
frames to converge and could suppress a head-of-video start trigger; pre-seeding kills that warm-up
artifact.

**`yaw_suspends_mar(yaw, params)`** = `yaw is not None and abs(yaw) > 35.0`. Beyond ±35°, lips project
into foreshortened geometry and MAR crashes artificially, so MAR checking is suspended entirely (no
transitions fire). **Unknown yaw does *not* suspend** — that case is logged once per video as
`pose_unavailable` instead (so an InsightFace build without pose degrades visibly, not silently).

## 3.4 `window_machine.py` — the `E_window` capture FSM (253 lines, pure)

A pure, deterministic finite state machine over per-frame `FrameObs` (`pts_ms`, `target_present`,
`target_mar` raw, `target_suspended`, `interviewer_present`, `interviewer_mar`, `vad_speech`). States:
`IDLE → ACTIVE → PLOSIVE_BUFFER → (ACTIVE | emit)`. End reasons: `plosive_expiry`,
`interviewer_interjection`, `end_of_video`.

The machine maintains a target EMA and an interviewer EMA (both `CausalEMA`), plus per-window stats
(`frames`, `interviewer_present_frames`, `interviewer_closed_frames`, and the full
`(pts, smoothed MAR | None-while-suspended)` `mar_trace`). EMA update rules each frame:

- **Interviewer EMA:** updated when the interviewer is present *and* MAR is available; **decayed to
  `None`** when the interviewer is absent (a stale pre-gap value cannot survive an absence and
  false-trigger the Early Stop Rule on re-entry); left untouched on a transient present-but-MAR-missing
  frame.
- **Target EMA:** **frozen** while suspended (a brief head-turn must not lose the smoothed value), but
  **decayed to `None`** while the target is *absent* (so a stale value cannot drive a spurious
  start/resume; it reseeds to the returning frame's raw MAR).

The transition table (exact):

| State | Condition (exact) | Action |
|---|---|---|
| IDLE | target present ∧ ¬suspended ∧ smoothed MAR > `mar_on` ∧ `vad_speech` | `T_start = pts`; → ACTIVE |
| ACTIVE | suspended | accumulate; EMA frozen; wait for head return |
| ACTIVE | target absent **or** smoothed MAR < `mar_off` **or** `not vad_speech` | → PLOSIVE; `deadline = pts + 500 ms` |
| PLOSIVE | suspended | **pause the timer**: `deadline += pts − prev_pts` |
| PLOSIVE | `pts ≥ deadline` | emit with **`T_stop = deadline`** (`plosive_expiry`) |
| PLOSIVE | interviewer present ∧ interviewer EMA > `mar_on` | emit **immediately**, `T_stop = pts` (`interviewer_interjection`) |
| PLOSIVE | target present ∧ EMA > `mar_on` ∧ `vad_speech` | cancel timer; → ACTIVE |
| any open | end of video (`finalize`) | emit at last frame PTS (`end_of_video`) |

The rules and their *why*:
- **Plosive buffer (500 ms):** plosives/brief closures (/p/, /b/, swallows) close the lips
  mid-utterance; without the buffer every "p" splits the window.
- **Early Stop Rule (the single most important transition):** if the *interviewer's* lips open while
  the target's closure timer runs, the window ends **now** and the buffer is discarded — otherwise a
  rapid interviewer interjection landing inside the 500 ms grace would be captured into enrollment
  audio. This is the contamination firewall at capture time.
- **Clean expiry stops at the deadline PTS, not the noticing frame.** Frames arrive ~33 ms apart;
  using the observation frame's PTS would leak up to one frame of post-closure audio. The deadline
  itself is the correct, frame-rate-independent stop.
- **Suspension pauses the timer** (deadline extended by the suspended wall time): a turned head removes
  the *evidence* of closure, so the closure clock should not run while the evidence is absent.
- **Target absence is treated as a closure** (same plosive semantics). **[Code-vs-old-doc divergence,
  now resolved in this doc]:** an earlier deep-dive said the target EMA is *frozen* on absence; the
  **code decays it to `None`** on absence (only yaw suspension freezes it). Trust the code — and this
  reference reflects the code.
- **Resume from PLOSIVE additionally requires `vad_speech`** (bench-corrected 2026-06-12): the
  corrected MAR's narrow band (~0.10–0.25) cannot alone distinguish a resting-open mouth in silence
  from speech, so Silero is required alongside `MAR > mar_on`. Likewise the ACTIVE→PLOSIVE close is
  driven primarily by `not vad_speech` (with `MAR < mar_off` as a secondary trigger), so a window
  cannot span a non-speech segment while the lips happen to stay open.

`WindowMachine.step(obs)` returns a finished `CandidateWindow` when one ends at this frame (else
`None`); `finalize(last_pts_ms)` closes any open window. A `CandidateWindow` exposes
`duration_ms = t_stop_ms − t_start_ms`.

## 3.5 `gates.py` — the Triple Validation Gate + helpers (148 lines, pure)

**`evaluate_triple_gate(...)`** evaluates **A → B → C, first failure wins**, and returns a
`GateResult(accepted, failed_gate, detail)`:

| Gate | Pass condition (exact) | Fail reasons |
|---|---|---|
| **A** | VAD coverage ≥ `gate_a_vad_min_coverage` (= 0.50) **and**, if the interviewer was ever visible, interviewer-closed fraction ≥ `int_lips_closed_frac` (= 0.80); never-visible ⇒ vacuous pass (`closed_frac = None`) | `vad_coverage_low`, `interviewer_lips_open` |
| **B** | `cos(window, E_seed) ≥ threshold_target` (= 0.70); fails on `<` | `low_sim_to_seed` |
| **C** | only if anti pool non-empty: `cos(window, E_anti) ≤ threshold_anti` (= 0.50) **and** `cos_seed − cos_anti ≥ margin_minimum` (= 0.15) | `sim_anti_missing`, `high_sim_to_anti`, `margin_too_small` |

**Gate C fails open:** with no anti profile (`anti_available=False`) it is *skipped, never failed*
(`anti_applied=False` recorded). Other helpers:
- **`mtrap_discard(sim_to_seed, params)`** = `sim_to_seed > mtrap_sim_max` (0.60) — the M-Trap (§3.9).
- **`contamination_level(sim, params)`** returns `OK` / `WARNING` (> 0.45) / `HALT` (> 0.60) —
  guardrail 8.
- **`pairwise_cosine_variance(vectors)`** — population variance of all `C(n,2)` pairwise cosines (fixed
  i<j order, `fsum`), 0.0 for < 2 vectors. A self-consistency measure: a clean single-speaker pool has
  high mutual similarity and low variance.
- **`segment_overlap_ms(start, stop, segments)`** — total overlap of `[start,stop)` with the segment
  list (used by Gate A coverage *and* Layer 2's skip rule).
- **`vad_near(pts, segments, tol)`** — `True` iff Silero marks speech within ±`tol` of `pts`.

## 3.6 `quality.py` — progressive enrollment states (48 lines, pure)

`assess_quality(verified_ms, pool_variance, anti_available, params)`:

| State | Condition |
|---|---|
| **STRONG** | `verified_ms ≥ strong_ms` (45 000; **60 000 when no anti pool** — the NO_ANTI escalation, since Gate C is inert) **and** `pool_variance ≤ variance_high` (0.05) |
| **MARGINAL** | `verified_ms ≥ marginal_ms` (20 000) — absorbs the "met seconds bar but variance > 0.05" case |
| **INSUFFICIENT** | below 20 000 ms |

CRITICAL FAILURE is a batch-end condition enforced by the orchestrator (guardrail 9), not here.

## 3.7 `encoding.py` — ECAPA encoding + pure arithmetic (158 lines)

The pure half is stdlib-only and self-tested; torch is imported lazily in the two functions that touch
the model. `SAMPLE_RATE = 16000`, `BYTES_PER_SAMPLE = 2`.

- **`l2_normalize(vector)`** — `norm = sqrt(fsum(x*x))`; returns zeros if `norm <= 1e-12`.
- **`cosine(a, b)`** — `fsum`-based dot over `zip(a,b)` divided by the product of `fsum`-based norms;
  returns 0.0 if either norm `<= 1e-12`. This is the native metric of ECAPA's embedding space.
- **`duration_weighted_mean(vectors, durations_ms)`** — `Σᵢ vᵢ·dᵢ / Σ dᵢ` per dimension with `fsum`
  (inputs are L2-normalized d-vectors). The enrollment pool's core arithmetic; duration weighting is
  the defense against short-window noise (a marginal 2 s window cannot outvote a clean 10 s window —
  votes are milliseconds). Raises on empty/mismatched inputs or non-positive total duration.
- **`plan_chunks(duration_ms, max_ms, overlap_ms)`** — the Single-Pass rule: one chunk up to the 60 s
  cap; above it, split at `max_ms` boundaries with `overlap_ms` overlap.
- **`pcm_slice(pcm, audio_start_pts_ms, num_samples, start_local_ms, stop_local_ms)`** — slices raw
  int16 PCM by **local PTS** ms (subtracts `audio_start_pts_ms` to reach data-relative samples; ×16
  samples/ms; clamped to `[0, num_samples]`). This is the **only** audio-slicing primitive in the
  pipeline — reused by Layer 2's scorer and Layer 3's WAV writer, so "every output sample existed in
  the source" reduces to one function.
- **`ecapa_encode_pcm(model, pcm)`** — one ECAPA forward pass over one PCM slice under
  `torch.no_grad()`; returns the L2-normalized 192-dim d-vector as a plain Python float list
  (framework-free above this boundary, `fsum`-reduced thereafter).
- **`encode_window(...)`** — single pass below the 60 s cap, else 60 s/2 s-overlap chunks pooled with
  duration weighting. (On 48 GB even a 60 s window is trivial in one pass, so the earlier 10 s
  sub-chunking was deleted.)

All d-vectors are **L2-normalized at the encoding boundary** and handled as plain float lists above it.

## 3.8 `vision.py` — the GPU-facing visual scan (194 lines)

cv2 / ultralytics / insightface are imported lazily, so the module's *import* is stdlib-safe but its
*functions* run only on the bench with the resident models. Data: `FaceObs` (`bbox`, `det_score`,
`embedding` L2-normalized, `mar` raw, `yaw_degrees` or `None`) with `contains_point` and `center`;
`FrameFaces` (`pts_ms`, `faces`).

- **`video_frame_pts_ms(video_path)`** reads every video packet's `pts_time` via
  `ffprobe -select_streams v:0 -show_entries packet=pts_time -of csv=p=0`, converts each with the
  half-even rule, and **sorts ascending** (packets arrive in decode order; presentation order is sorted
  PTS). Frame timing comes from the container, never from frame counting.
- **`scan_video(models, video_path, frame_pts, speech_segments_local_ms, params)`** decodes the video
  **once, sequentially** with OpenCV (presentation order, so decoded frame *i* pairs with `pts[i]`),
  batches frames into `VISUAL_BATCH_FRAMES = 32`, and `_flush_pending()` runs **YOLOv8m as a person
  gate** (`conf=yolo_min_conf=0.30`, `classes=[0]`); frames with no person **skip InsightFace
  entirely**. For person frames, InsightFace yields faces; any face with `det_score < 0.50` is treated
  as **not detected** (guardrail 5 — a low-confidence face must not be allowed to *fail* an identity
  check, it simply doesn't exist). Each surviving face carries its ArcFace embedding
  (`normed_embedding` preferred, else `embedding`), MAR from `landmark_2d_106`, and
  `yaw = face.pose[1]` when pose exists. The optional silence-stride rule (`silence_stride > 1`) analyzes
  frames outside Silero speech only every Nth frame (default 1 = off; still input-deterministic).
  Returns `(analyzed_frames, stats)` where `stats` carries `decoded_frames`, `listed_pts`,
  `analyzed_frames`, `pts_mismatch` (a count mismatch sets this; pairing stops at the shorter length —
  on mismatch the tail is unanalyzed rather than mis-timed), and `silence_stride`.

## 3.9 `enrollment.py` — the orchestrator (1129 lines)

The sequential, canonical-order conductor. Operation entry names: `layer1_init`, `layer1_seed`,
`layer1_video_scan`, `layer1_quality`, `layer1_freeze` (plus `enrollment_vector` /
`enrollment_discard` from the shared vocabulary). Pool-entry kinds: `seed`, `track_a_window`,
`anti_track_b`, `anti_track_c`. `CLICK_MATCH_MAX_GAP_MS = 200`.

### Operator click input

`SpeakingClick` / `AntiClick` (`file_index`, `pts_ms`, `x`, `y`) and `OperatorClicks` (`speaking`,
optional `anti`, plus schema-v2 fields `extra_seeds`, `target_bearded`, `interviewer_bearded`).
**`load_clicks(path)`** parses the JSON:
```json
{"speaking_click": {"file_index": 0, "pts_ms": 41250, "x": 812, "y": 440},
 "anti_click":     {"file_index": 0, "pts_ms": 95000, "x": 300, "y": 400}}
```
`file_index`/`pts_ms` must be integers (bool rejected); `anti_click` is optional (Track C is optional
by design). Schema-v2 keys: `speaking_clicks` (a list of additional seed clicks → `extra_seeds`),
`target_bearded`, `interviewer_bearded` (must be booleans).

### Result structures

`PoolEntry` (`vector`, `duration_ms`, `kind`, `file_index`, `t_start_local_ms`, `t_stop_local_ms`) and
`EnrollmentResult` (`f_target`, `e_seed`, `e_composite` + sha, `e_anti` + sha, `no_anti_profile`,
`pool`, `anti_pool`, `total_verified_ms`, `quality_history`).

### Internal helpers

- `_mean_embedding(embeddings)` — L2-normalized mean (`fsum` per dim) — used to refine the biometric
  lock.
- `_face_at_click(frames, pts, x, y)` — the clicked face: nearest analyzed frame with faces (within
  `CLICK_MATCH_MAX_GAP_MS = 200`, else re-click), preferring a bbox containing the point, else the face
  whose center is nearest.
- `_match_face(faces, anchor, threshold)` — best face whose embedding cosine ≥ `threshold`.
- `_build_obs(frames, f_target, f_interviewer, file_audio, params)` — assigns identities per frame
  (target by best-cosine ≥ `face_reid_threshold`; interviewer = matched `f_interviewer` if known, else
  the highest-confidence non-target face) and assembles the `FrameObs` stream + Track B candidate
  centers (a non-target face with **raw** MAR < `mar_off` and Silero energy within ±`vad_tol`) +
  `any_pose` + `reid_sims`. Returns an `_ObsBundle`.
- `_run_machine(obs, params)` — drives a fresh `WindowMachine` over the obs and finalizes.
- `_persist_window(...)` — writes the raw WAV slice + a canonical JSON sidecar (PTS range, global
  range, end reason, MAR trace) for one window and returns paths + SHA-256s.
- `_vector_payload(...)` — the `enrollment_vector` payload (kind, file index, local+global PTS,
  duration, `vector_dim`, `vector_sha256`, persistence hashes, extra).
- `_reclick(manifest, reason, detail)` — appends a `reclick_required` warning, then raises
  `Layer1ReclickError`.
- `_clamp_local(file_audio, pts)` — clamps a PTS to `[audio_start_pts_ms, audio_start_pts_ms + duration_ms]`.

### `run_layer1` — the entrypoint

Opens `<work_dir>/enroll/`, writes `layer1_init` (params, click summary, beard flags,
`enrollment_mode = "audio_anchored" if target_bearded else "mar"`), and enforces that **both clicks
target the first video** (blocking halt otherwise — anchors propagate forward, never backward). It then
defines closures over mutable state (`f_target`, `f_interviewer`, `e_seed`, `seed_span`, `pool`,
`anti_pool`, `e_anti`, `e_composite`, `verified_ms`, `pool_variance`, ...):

- `_encode(file_audio, t0, t1)` — `encode_window` against the resident ECAPA.
- `_recompute_anti()` — `E_anti = L2(duration_weighted_mean(anti_pool))`.
- `_recompute_composite()` — `E_composite = L2(duration_weighted_mean(pool))`, plus `verified_ms` (sum
  of pool durations) and `pool_variance` (pairwise-cosine variance). **`E_composite` is recomputed
  after each video**, so video 2's gates already use a better composite than video 1's — the designed
  order-dependence behind strict-sequential processing.
- `_accept(...)` — persist the window, append a `PoolEntry` to `pool` (or `anti_pool`), and write an
  `enrollment_vector` entry.

Then the **strictly sequential per-video loop** (canonical file order). For each video it scans frames
(`video_frame_pts_ms` + `scan_video`), writes `layer1_video_scan`, and emits a `frame_pts_mismatch`
warning if needed. On **video 1 only**, it does the clicks → `F_target` / `E_seed` / optional Track C
(two code paths: the MAR path and the audio-anchored path, §3.10). After enrollment is bootstrapped, on
**every video** it builds the production observation stream and runs Track A + Track B.

### Track A — candidate windows through the Triple Gate (MAR mode)

For each `CandidateWindow` from `_run_machine`: clamp to local PTS; discard if it overlaps the seed
span on the seed's file (`seed_overlap`); discard if shorter than `min_enroll_len_ms = 2000`
(`below_min_enroll_len`); compute VAD coverage; encode once; compute `sim_seed = cos(vector, e_seed)`
and (if anti) `sim_anti`; run `evaluate_triple_gate`. Accepted → `_accept` (Track A). Otherwise an
`enrollment_discard` with the failed gate and full detail; Gate-C failures are stashed for a possible
second pass.

### Track B — automatic anti-profile collection + the M-Trap

For each Track B candidate center (deduplicated to one per `trackb_min_spacing_ms = 2000`): window
±1000 ms (clamped; skipped if clipping leaves < half), encode, compute `sim_to_seed`. **M-Trap
(guardrail 4):** if `sim_to_seed > mtrap_sim_max = 0.60`, discard (`mtrap_high_sim_to_seed`). The trap
exists because the trigger has a systematic false positive: **bilabials/nasals (/m/, /b/, /n/) are
produced with closed lips by the target** while the target is speaking. Without the trap, those
acoustically-*target* windows poison `E_anti`, and the poisoning self-reinforces
(`sim(E_composite, E_anti)` rises, Gate C margins collapse, the anti-profile starts rejecting the
target's own windows). Survivors → `_accept` into the anti pool.

### Pool maintenance, sanity checks, quality, second pass

After Track A/B, if the anti pool is non-empty it recomputes `E_anti`, checks the anti-pool variance
increase (> `pool_var_warning` → `anti_pool_variance_increase` warning), then `_recompute_composite`.
**Guardrail 8:** with anti present, `cos(E_composite, E_anti)` → `enrollment_contamination` warning
(> 0.45) or `enrollment_contamination_critical` **blocking halt** (> 0.60). Then `assess_quality`.

**MARGINAL second pass:** if the state is MARGINAL and there are stashed Gate-C failures and an anti
profile exists, each failed candidate is re-evaluated against the *grown* anti profile (vectors and
`sim_seed` reused — each window is encoded exactly once ever; only `sim_anti` is recomputed). Accepted
recoveries re-trigger `_recompute_composite` and a re-assessment. A `layer1_quality` entry records the
state, verified ms, pool sizes, variance, and `second_pass_accepted`.

### Batch end: critical failure, variance gate, freeze

After all videos: **guardrail 9** — if `verified_ms < marginal_ms` (20 000) or the final state is
INSUFFICIENT, a `critical_enrollment_failure` blocking halt fires (operator intervention required). A
final pool variance > `variance_high` is a **warning only** (`high_pool_variance_operator_review` —
human decides; windows are flagged, never auto-discarded). Then the **`layer1_freeze`** entry records
`e_composite_sha256`, `e_anti_sha256`, the anti-pool hash, pool sizes, total verified ms, variance, and
the note "E_composite is FROZEN — never modified after this entry." That `e_composite_sha256` is the
`enrollment_ref` every Layer 2 artifact points back to. `run_layer1` returns the `EnrollmentResult`.

## 3.10 The audio-anchored (beard / unreliable-MAR) path

When the operator declares the target bearded (`target_bearded=True`), geometric lip-landmark MAR is
not a trustworthy speaking signal (2d106 lower-lip localization fails under a dense beard —
bench-confirmed 2026-06-17 on UB-clip2, where target MAR stuck ~0.20–0.25 regardless of mouth state).
The target is then enrolled from **target-solo + VAD spans**, attributed by ECAPA against a **consensus
anchor** instead of by lips. This path turned a 3.84 s crash into 85 s STRONG.

Pure helpers:
- **`target_solo_vad_spans(frames, f_target, file_audio, params)`** — contiguous spans where the
  target is the **only** person on screen (`others <= audio_solo_face_max_others`, default 0) **and**
  Silero shows speech. No MAR, no yaw — the speaker is the target *by elimination*. Runs split on a gap
  > 3 frame intervals; kept only if ≥ `audio_solo_min_ms` (2000).
- **`outlier_seed_indices(seed_sims, params)`** — seed clicks whose similarity to the consensus anchor
  falls below `audio_anchor_consistency_min` (0.65) — an unrepresentative seed (e.g. brief cross-talk).

The seed construction (video 1, audio-anchored branch): build `F_target` from every seed click's face
embedding (mean); compute solo+VAD spans; encode them; require each seed click to fall inside a solo
span (else re-click `seed_click_not_in_solo_span`); build a **provisional** anchor from the
duration-weighted mean of the seed spans, then **collect** every other span whose cosine to the
provisional ≥ `audio_anchor_collect_sim` (0.55) into the **consensus** anchor `e_seed`. A
**leave-one-out** self-consistency check scores each seed against the consensus of the others; outliers
(or a single un-checkable seed) emit a warn-only `seed_anchor_outlier` (the solo-on-screen structure
already guarantees identity, so a weak anchor is a flag to add seed clicks, not a hard block). The seed
spans are accepted into the pool and a `layer1_seed` entry (mode `audio_anchored`) is written.

Per-video collection (`_audio_collect`): for each solo+VAD span, encode, apply an **anchored Gate B**
(`sim_seed ≥ audio_anchor_accept_sim = 0.78` — bench-derived: target-vs-consensus ~0.83,
interviewer-vs-consensus peaks ~0.69) and, if anti exists, the same Gate C as the MAR path. In this
mode Track A's MAR machine and Track B are bypassed; the Track C anti-click skips the target-lips check
(MAR untrusted) but keeps the identity + VAD checks.

## 3.11 The nine guardrails (map)

1. **Speaking-click overlap** — seed-window non-target-lips-open fraction ≤ `click_overlap_max_frac`
   (0.20).
2. **Speaking-click duration** — seed window ≥ `seed_min_ms` (3000; no max).
3. **Anti-click identity** — clicked face must **not** match `F_target` (`cos ≥ face_reid_threshold`
   0.40 → re-click).
4. **M-Trap (Track B)** — `cos(candidate, E_seed) > mtrap_sim_max` (0.60) → discard.
5. **InsightFace confidence** — `det_score < 0.50` ⇒ not detected (in `vision.py`).
6. **Low detection quality** — running-mean ReID < `reid_warning_floor` (0.50) → warning.
7. **Separation margin** — Gate C `cos_seed − cos_anti ≥ margin_minimum` (0.15).
8. **Acoustic similarity** — `sim(E_composite, E_anti)`: warn > 0.45, halt > 0.60.
9. **Critical failure** — batch end still INSUFFICIENT / < 20 s verified → terminal halt.

## 3.12 `selftest.py` / `__main__.py` / `__init__.py`

`__init__.py` re-exports the public surface (`run_layer1`, `load_clicks`, `EnrollmentResult`,
`EnrollmentParams`, the click/error/pool types). `__main__.py` is the CLI (`--selftest` or `--run`,
chaining gate → Layer 0 → Layer 1). `selftest.py` (361 lines) drives every pure submodule with zero
pip installs: geometry (the corrected MAR pairs give exactly 1.0 on a synthetic mouth; EMA pre-seed;
yaw boundaries), the window machine (clean plosive expiry stops at the deadline; resume; the Early Stop
Rule; the VAD-gated start; yaw pauses the timer; absence behaves like a plosive; interviewer
closed-frame counting), gates (all A/B/C failure modes, M-Trap, contamination levels, variance,
segment lookups), quality (all six state cases including NO_ANTI escalation), encoding (l2/cosine/dwm,
chunk plans, PCM slice clamping), params+clicks (Rule-6 + canonical-JSON validity; float `pts_ms`
rejected), and the audio-anchored helpers (two solo runs survive a two-shot; no-VAD → no spans;
outlier detection). It asserts none of torch/cv2/numpy/onnxruntime/insightface/ultralytics were
imported.

---

# Part 4 — Layer 2: Calibrated Sliding-Window Tracking (`layer2_tracker.py`, 1197 lines)

The single **authoritative** target-tracking pass: scan the raw 16 kHz audio with the frozen
`E_composite` / `E_anti` via pure cosine similarity, calibrate thresholds per session, median-pool
overlapping window scores into 1-second blocks, tier them, refine HIGH-run edges (trim-only), and emit
PTS-stamped raw blocks for Layer 3 plus the SHA-256-hashed authoritative output document.

## 4.1 Score semantics: a cosine is NOT a probability

Layer 2's per-window quantities are **raw cosine similarities in [−1, 1]**, named `S_target` and
`S_interviewer` — never `P(·)`. Cosine is the **native metric** of ECAPA's embedding space (ECAPA is
trained with AAM-Softmax / ArcFace, which optimizes *angular* margins). The historical bug: early
drafts used fixed probability tiers (0.85/0.65/0.30); empirically genuine same-speaker short-window
cosines land ~0.4–0.8 and different-speaker same-channel ~0.1–0.4, so a fixed 0.85 HIGH gate would
have starved the HIGH tier on good sessions. The structural fix is **per-session calibration** + naming
discipline enforced through every payload.

## 4.2 Constants and structures

`Layer2Params` (frozen dataclass) holds the architectural windowing constants (`window_ms = 5000`,
`hop_ms = 1000`, `block_ms = 1000`, `silero_skip_floor = 0.20`, `evidence_floor = 0.20`), the edge-trim
constants (`edge_fine_window_ms = 2000`, `edge_fine_hop_ms = 250`, `edge_scan_span_ms = 2000`,
`edge_min_fine_window_ms = 1000`, `edge_max_trim_ms = 750`), the calibration constants
(`genuine_quantile = 0.10`, `impostor_safety_margin = 0.05`, `theta_clamp_low = 0.45`,
`theta_clamp_high = 0.75`, `theta_med_step = 0.15`, `theta_med_floor = 0.30`,
`min_calibration_windows = 10`, `fallback_theta_high = 0.60`, `fallback_theta_med = 0.40`,
`no_anti_theta_floor = 0.55`, `margin_minimum = 0.15`), and the diagnostics constants
(`anti_contam_warning = 0.45`, `anti_contam_halt = 0.60`, `ratio_normal = 0.25`, `ratio_low = 0.10`,
`drift_window_ms = 30000`, `drift_delta = 0.10`).

Tier constants: `HIGH`, `MEDIUM`, `SUB_THRESHOLD`, `REJECT`, `SKIPPED_NONSPEECH`. Calibration kinds:
`DERIVED`, `DERIVED_NO_ANTI`, `FALLBACK_DEFAULTS`. Operation entries: `layer2_init`, `layer2_block`,
`layer2_edge_trim`, `layer2_file_summary`.

Dataclasses: `Calibration` (`theta_high`, `theta_med`, `kind`, `record`, `calibration_ref`,
`overlap_warning`), `HighRun` (`start_local_ms`, `end_local_ms`, `blocks`), `FileTrack` (per-file tier
counts, HIGH runs, ratio/drift diagnostics, and the **full per-block tier map** added for Layer 3's
gap-dominance guard), and `Layer2Result`. The **`Scorer`** type is `Callable[[spans], List[(S_t,
S_i|None)]]` — an **injectable seam**: production binds the resident ECAPA model; the self-test injects
synthetic scorers and drives the whole flow without torch.

## 4.3 Deterministic threshold calibration

Pure, ordered arithmetic on frozen vectors — no sampling, no optimization, bit-reproducible.

- **`quantile_sorted(sorted_values, q)`** — linear interpolation between order statistics at position
  `q·(n−1)`; pinned to one definition.
- **`loo_scores(vectors, durations_ms)`** — **genuine** leave-one-out scores: each pool vector against
  the L2-normalized duration-weighted mean of the others. A pool with < 2 vectors yields `[]` (so a
  single ≥ 20 s seed-only pool legitimately routes to FALLBACK rather than crashing). LOO-against-the-
  pooled-rest mirrors the *test condition* (a short window scored against a pooled profile).
- **`derive_thresholds(genuine, impostor, anti_available, enrollment_ref, params)`** — the derivation:
  ```
  if len(genuine) < min_calibration_windows (10):
      theta_high = 0.60, theta_med = 0.40            (FALLBACK_DEFAULTS)
  else:
      q = q10(genuine)
      theta_high_raw = max(q, max(impostor) + 0.05)  if anti     (DERIVED)
      theta_high_raw = max(q, 0.55)                  if no anti   (DERIVED_NO_ANTI)
      theta_high     = clamp(theta_high_raw, 0.45, 0.75)
      theta_med      = max(theta_high − 0.15, 0.30)
  ```
  - **Impostor** scores are `cos(aⱼ, E_composite)` for every anti-pool vector. `max(impostor) + 0.05`
    places θ strictly above *every observed* impostor (the **maximum**, not a quantile — forensic
    posture: "no observed impostor may pass").
  - **`q10(genuine)`** admits ≈90% of genuine-like windows.
  - **Clamp low 0.45** stops a degenerate genuine distribution opening the gate into cross-speaker
    score zone; **clamp high 0.75** is the overlap detector — if data pushes θ past 0.75, the
    genuine/impostor distributions overlap, raising a **`CALIBRATION_OVERLAP`** warning, and the
    **margin rule becomes the primary discriminator** (a higher θ would reject genuine speech
    wholesale — flag beats fail-silent).
  - **`margin_minimum = 0.15` is NOT calibrated** — it is a relative separation requirement between two
    scores of the *same* window, identical to Layer 1's Gate C margin by design.
  - The full record (both sorted score lists, n's, every intermediate, clamp events, `enrollment_ref`)
    becomes a `calibration` manifest entry; `calibration_ref = sha256(record)` stamps every output
    block.

## 4.4 Windowing, the skip rule, and the production scorer

- **`plan_windows(duration, 5000, 1000)`** — data-relative spans `[0,5000),[1000,6000),…`; a file
  shorter than one window yields a single full-file window. **Why 5 s/1 s:** ECAPA d-vectors need
  seconds of context to stabilize; the 1 s hop makes each interior 1 s block receive **five overlapping
  estimates** — the raw material median pooling needs.
- **Silero skip rule** (in `track_file`): a window whose overlap with the speech map is `< 20%` of its
  length (`silero_skip_floor`) is **not scored**, logged as `layer2_window_skipped` /
  `SKIPPED_NONSPEECH`. A *compute* skip with an audit trail; audio untouched.
- **`_ecapa_scorer`** (the determinism-critical production scorer): groups spans by length ascending;
  batches of exactly `ECAPA_BATCH_WINDOWS = 256`. **Repeat-padding:** the final partial batch is padded
  by **repeating the last window's tensor** until full (padded rows discarded by index bookkeeping).
  Repeat- (not zero-) padding keeps pad rows in-distribution and keeps every forward pass at the
  identical `[256, T]` shape (FP reduction order is part of the numeric function). The **C1 fix** makes
  this apply to **sub-256 groups too**, so *every* file's forward pass runs at exactly 256 regardless of
  how many windows it produced. Per window: `pcm_slice` → `torch.frombuffer(int16).float()/32768.0` →
  `models.ecapa.encode_batch(stacked)` → reshape → `l2_normalize` → `S_target = cos(v, E_composite)`,
  `S_interviewer = cos(v, E_anti)` or `None`.

## 4.5 Median pooling onto the 1-second block grid

**`median_pool_blocks(duration, spans, scores, params)`** — the grid has `duration_ms // 1000` blocks;
a trailing partial second **never enters the grid** (conservative). A window votes only on blocks it
**fully covers**: `first_block = ceil(w_start/1000)`, `last_block = floor(w_end/1000)` (exclusive).
Interior blocks collect 5 votes; edge blocks taper 4…1; blocks with zero covering scored windows get
no evaluations (→ `SKIPPED_NONSPEECH` in tiering). Per block, the score is the **median** of its votes
(both tracks pooled independently via `statistics.median` — even counts average the middle pair).
**Why median, not mean:** it defends against the transient excursion (a window straddling a turn
boundary, a cough, an edge artifact). Median of *k* tolerates ⌊(k−1)/2⌋ corrupted votes (k=5 → two bad
windows leave the median on clean evidence). **Turn-boundary dilution:** a 5 s window straddling a
speaker change embeds a *mixture* → diluted score → turn-edge blocks sink to MEDIUM (desired
conservatism — turn edges are where interviewer bleed lives).

## 4.6 Tiering and margin demotion

**`tier_block(s_target, s_interviewer, calibration, anti_available, params)`**:
```
HIGH    iff S_target > theta_high  AND ( no anti  OR (S_interviewer present AND S_target − S_interviewer > 0.15) )
MEDIUM  iff S_target > theta_med            (also: a margin-failed HIGH, with margin_failed=True)
SUB     iff S_target >= 0.20                 (evidence_floor; sub-threshold log)
REJECT  otherwise
```
A block **above θ_high that fails the margin** is **demoted to MEDIUM with `margin_failed = True`** —
not rejected (target evidence is real), not HIGH (interviewer too close). **That flag is load-bearing
downstream:** Layer 3's gap-dominance guard treats `MEDIUM ∧ margin_failed` as interviewer evidence.
Every block becomes a `layer2_block` worker record (tier, both medians, evaluation count, margin flag,
`no_anti_profile`). MEDIUM is **never** auto-promoted to clean output (human-review evidence); SUB is
investigative-only; below 0.20 is rejected with the score logged.

## 4.7 Edge-Trim Boundary Refinement (trim-only)

**`refine_run_edges(...)`** runs for each **maximal run of contiguous HIGH blocks**
(`find_high_runs(tiers)`). The coarse scan localizes target speech reliably but its block edges inherit
±1 s uncertainty — exactly where interviewer bleed lives.
- **17 candidate edge positions** per boundary: `range(edge−2000, edge+2000+1, 250)` (`_fine_positions`).
- **Fine windows:** leading edge scores `[p, p+2000)`; trailing edge scores `[p−2000, p)`. Windows
  clamped at file bounds to under `edge_min_fine_window_ms = 1000` are skipped (`None` in the trace).
  Each boundary's windows go to the scorer as one batch.
- **Pass criterion = the same frozen standard as the coarse scan** (`_passes`): `S_target > θ_high` and,
  with anti, margin `> 0.15`.
- **Trim:** new start = the **smallest** passing `p ≥ run_start`; new end = the **largest** passing
  `p ≤ run_end`. Positions outside the run are scanned (their scores appear in the trace) but are
  **ineligible** → **trim-only by construction**: the run can shrink, never grow; audio outside the
  coarse HIGH region can never be promoted here.
- **Demotion:** no passing position, or a required trim beyond `edge_max_trim_ms = 750` → that
  boundary's single edge 1 s block is demoted to MEDIUM (`demoted_by_edge_trim`), and the scan **does
  not recurse** (bounded correction + audit trail beats unbounded cascade). A one-block run demotes at
  most once.
- **Audit:** `leading_trim_ms`, `trailing_trim_ms`, demotion flags, and the **complete fine-scale score
  traces** per boundary → a `layer2_edge_trim` record. Surviving runs' HIGH blocks become the per-block
  records carried to Layer 3 (with local + global PTS, both medians, edge-trim attribution,
  `no_anti_profile`).

## 4.8 Diagnostics: activity ratio and drift

- **`activity_ratio_level(ratio, params)`** — per file, `ratio = HIGH ms ÷ Silero speech ms` →
  `NORMAL > 0.25`, `LOW_ADVISORY 0.10–0.25`, `NEAR_ZERO_ALERT < 0.10` (escalates to a main-manifest
  `near_zero_activity_manual_review` warning), `NO_SPEECH` if no speech. Logged beside it:
  **`unattributed_speech_ms`** = Silero speech not claimed HIGH — the analyst's context number for
  "interviewer dominated" vs "model failure." The system refuses to hide the question.
- **`drift_notice(previous_mean, current_first_mean, params)`** — current file's mean `S_target` over
  its first `drift_window_ms = 30000` of HIGH activity vs the previous file's mean over *all* its HIGH
  blocks; a drop > `drift_delta = 0.10` logs a `drift_notice` (informational; same mic/room, drift
  expected minimal; prev mean carries forward across HIGH-less files).

## 4.9 `track_file` and `run_layer2`

**`track_file`** wires it together for one file: convert Silero segments to data-relative, plan
windows, apply the skip rule (logging skips), score, median-pool, tier (writing `layer2_block`), find
and edge-trim HIGH runs (writing `layer2_edge_trim`, applying demotions, building the per-block carry
records), compute activity ratio + `unattributed_speech_ms` (writing `layer2_file_summary`), and return
a `FileTrack` (including the full per-block map).

**`run_layer2(manifest, batch, models, enrollment, work_dir, params, authoritative=True, scorer_factory=None)`**
— the single authoritative pass (Steps 1–9):
1. `layer2_init` (params, `enrollment_ref`, `no_anti_profile`, batch constant; if not authoritative,
   marked `superseded_by: authoritative_pass`).
2. **Step 1 — sanity:** re-check `cos(E_composite, E_anti)` with the same 0.45/0.60 warn/halt constants
   as Layer 1 (re-checked at the consumer because Layer 2 may run in a different process lifetime);
   no-anti → a `sanity_check_unavailable_no_anti_profile` warning.
3. **Step 2 — calibration:** `derive_thresholds` over the LOO genuine scores and the impostor scores;
   write the `calibration` entry; emit `CALIBRATION_OVERLAP` if clamped high.
4. **Steps 3–8 — per-file tracking** through `WorkerLog`s (sequential GPU, scorer from
   `_ecapa_scorer` or an injected `scorer_factory`), including the cross-video drift check.
5. **Canonical Merge** of the worker logs.
6. **Step 9 — output:** `layer2_output.json` (schema `spovnob-layer2-output-v1`) with `enrollment_ref`,
   `calibration_ref`, `thresholds_used` (θ values, margin, floor, kind, `operator_modified:false`),
   per-file tier counts + activity diagnostics + HIGH runs with per-block records. Canonical JSON,
   SHA-256 recorded as `output_hash` — the hash every Layer 3 artifact points back to. Files at
   `NEAR_ZERO_ALERT` get a main-manifest warning.

**Single authoritative pass.** Enrollment improves monotonically through the batch and Layer 2 has no
feedback loop, so exactly one result matters — produced with `E_composite_final` + final-pool
calibration, run once after Layer 1 completes. An optional preview pass exists (init marked
`superseded_by`); **Layer 3 refuses non-authoritative input with a blocking halt**, so a preview can
never leak into output.

## 4.10 Self-test

Exercises quantiles, LOO (including the seed-only → `[]` case), every calibration branch (DERIVED,
clamp-low, clamp-high+overlap, DERIVED_NO_ANTI, FALLBACK, seed-only→FALLBACK, ref reproducibility),
window planning, all tiering boundaries (including margin-fail demotion and no-anti HIGH), three
edge-trim scenarios over a synthetic 20 s file (no trim → run `[7000,12000)`; 250 ms leading trim;
all-leading-fail → first block demoted), activity-ratio levels, drift, and a full end-to-end
`run_layer2` with injected scorers asserting a **deterministic output hash across two runs** and a
canonical worker-log merge into a verified manifest.

---

# Part 5 — Layer 3: Overlap Exclusion & Final Output (`layer3_contamination.py`, 808 lines)

The final forensic filter. Layer 2 answers *"where is the target speaking?"* but a tracker cannot
certify single-speaker-ness — a high target cosine is compatible with the interviewer talking *over*
the target. Layer 3 answers *"is that speech contaminated by simultaneous speech?"* with a model
trained for exactly that (PyAnnote OVD), applied as the absolute final filter, then bridges natural
sub-400 ms gaps and slices the **original** audio to the final WAVs.

## 5.1 The overlap detector

**`pyannote_overlap_provider(models)`** wraps PyAnnote **segmentation-3.0** in
`OverlappedSpeechDetection`, instantiated by the gate with `{"min_duration_on":0.0,"min_duration_off":0.0}`
— the model's own temporal smoothing/min-duration filters are **zeroed** because *SPOVNOB owns its
temporal policy explicitly* (a hidden 100 ms min-duration inside a third-party pipeline would be an
unaudited rule shaping evidence). OVD runs **once over each full file** (maximal context, simpler,
deterministic, strictly more information than per-block windowing). Region timestamps: float seconds
× 1000 through Python `round()` (banker's rounding) + `audio_start_pts_ms`; then `merge_regions` →
disjoint local-PTS spans, logged as `layer3_ovd_regions`. (Under the hood segmentation-3.0 is a neural
model predicting per-frame multi-speaker activity over short chunks via a powerset formulation covering
up to ~3 concurrent speakers; the OVD pipeline marks frames where ≥2 speakers are simultaneously
active.)

## 5.2 Pure interval machinery

- **`intervals_intersect(a0,a1,b0,b1)`** = `a0 < b1 and b0 < a1` — **half-open `[start, end)`**, so
  blocks that merely *touch* a region do not intersect it (adjacency ≠ contamination).
- **`merge_regions(regions)`** — sort, merge overlapping **and touching**, drop degenerates → disjoint
  spans.
- **`region_overlaps_span(regions, start, end)`** — any region intersects `[start,end)`.

## 5.3 The NaN-Only Exclusion Policy

**`classify_run_blocks(run_blocks, regions)`** — any overlap region intersecting a block voids the
**entire block**: it becomes a NaN record (designation `"NaN"` string, decision `CONTAMINATED`, span,
both Layer 2 medians, exact regions hit; → `layer3_nan_block`, permanent in the manifest). Surviving
blocks regroup into maximal **clean sub-segments** wherever adjacent (`prev.end == next.start`).
**Why whole-block voiding** rather than excising only the overlapped ms: trimming to the overlap edge
means trusting the OVD boundary to the millisecond, and boundary-trust is the guesswork the doctrine
forbids (detection models are reliable about *presence*, fuzzy about *edges*). Losing up to one second
of genuine speech is the deliberate, one-directional price of never emitting an overlapped sample.
**Forbidden alternative** (named in the code's own docstring): separation/reconstruction models
(HTDemucs, SepFormer, SpeakerBeam) are banned — they would "rescue" NaN blocks by synthesizing what
the isolated target *probably* sounded like (hallucinated, forensically indefensible).

## 5.4 Temporal smoothing — here, and only here

**`assemble_segments(subsegments, regions, block_map, params)`** bridges a gap between two clean
sub-segments **iff all three hold**:
1. `gap < merge_gap_ms = 400`;
2. the gap interval itself is **overlap-free** (no OVD region intersects it);
3. the **gap-dominance guard** does not object.

Otherwise the bridge is refused with a recorded reason (`gap_too_long`, `overlap_in_gap`, or
`interviewer_evidence_in_gap`). Bridged gap audio (< 400 ms of *original* signal between two clean
blocks) **is** included in the output segment — that is the point of smoothing; it is unmodified
original audio. Every bridge evaluation is recorded in `gap_decisions` (the audit trail).

**Why smoothing was relocated out of Layer 2 (a Rev-2 fix):** if Layer 2 bridged a 300 ms gap *before*
overlap detection, it could swallow a rapid 300 ms interviewer interjection — fusing it into a block
and hiding it from the OVD. Bridging *after* OVD, only across proven-overlap-free gaps, makes that
impossible.

**`gap_dominance_blocked(gap_start, gap_end, block_map)`** (the implementation-added safeguard, default
ON via `Layer3Params.gap_dominance_guard`): even an overlap-free gap is **not** bridged if a Layer 2
block overlapping the gap shows interviewer evidence — `tier == "MEDIUM" and margin_failed`, **or**
`s_interviewer >= s_target`. OVD only detects *simultaneous* speech; a **solo** interviewer
interjection over target silence would pass an overlap-only check, so this guard refuses the bridge
using already-computed Layer 2 scores. Trim-only philosophy: the guard can only **refuse** bridges,
never create audio. It is inert when the block map is absent (the document's base OVD-only rule).

## 5.5 WAV emission and `contaminate_file`

**`_write_segment_wav(clean_dir, file_audio, segment)`** slices the **original** audio with `pcm_slice`
and writes a 16 kHz mono PCM16 WAV plus a canonical JSON sidecar (local + global PTS, duration, bridged
gaps, block count, and the policy string `original_signal_only_no_reconstruction`); returns paths +
SHA-256s.

**`contaminate_file(...)`** per file: `merge_regions` the OVD output (logging `layer3_ovd_regions`);
`classify_run_blocks` each HIGH run into clean sub-segments and NaN records (logging each
`layer3_nan_block`); `assemble_segments` (smoothing); write each clean segment's WAV + a
`layer3_segment` record; tally `clean_ms`, `contaminated_ms`, `bridged_gap_ms`, and the gap decisions
into a `layer3_file_summary`. Returns a `FileContamination`.

## 5.6 `run_layer3`

`run_layer3(manifest, batch, models, layer2, work_dir, params, overlap_provider=None)`:
- **Refuses non-authoritative Layer 2 input** with a `layer3_requires_authoritative_layer2` blocking
  halt (the preview-leak firewall).
- Creates `<work_dir>/layer3/clean/`, writes a `layer3_init` (params, `layer2_output_sha256`, policy
  `NaN_only_exclusion_no_separation_models`).
- Per file: run the overlap provider (resident PyAnnote OVD, or an injected provider for the self-test)
  and `contaminate_file` through a `WorkerLog`.
- Canonical-merge the worker logs.
- Write `layer3_output.json` (schema `spovnob-layer3-output-v1`) with per-file clean/contaminated/
  bridged totals, overlap regions, NaN blocks, and the clean-segment listing (with WAV paths +
  SHA-256). Record an `output_hash` (+ batch clean/contaminated totals). Returns a `Layer3Result`.

`Layer3Params` is `merge_gap_ms = 400` (operator-tunable via the manifest, strongly discouraged) and
`gap_dominance_guard = True`.

## 5.7 Self-test

Exercises the interval machinery (half-open touching does not intersect; merge with touching;
degenerate dropped), NaN classification (a `(9300,9600)` overlap voids the whole `9000–10000` block and
splits the run into two clean sub-segments), all bridge cases (contaminated-block gap can never be
bridged; clean 300 ms gap bridged with `bridged_gaps == [(9000,9300)]`; exactly-400 ms not bridged;
overlap-in-gap blocks; the dominance guard refuses then a base-rule run allows; `gap_dominance_blocked`
direct cases), edge-trimmed partial blocks surviving intact, a full end-to-end `run_layer3` with
injected regions (WAVs written with `getnframes() == duration_ms * 16`, manifest chain verified, file 0
one block NaN'd → `clean_ms == 4000`, file 1 fully clean → `clean_ms == 5000`, `total_clean_ms == 9000`),
and the non-authoritative-refusal path.

---

# Part 6 — The Runner: `pipeline_runner.py` (403 lines)

The batch orchestrator and single production entrypoint. Imports every earlier module; nothing imports
it.

## 6.1 What it chains

`run_pipeline(videos, clicks_path, work_dir, model_store, manifest_path, operator=None)`:
1. `load_clicks` (re-validates the operator clicks from scratch — `click_ui.py`'s output gets no
   special trust).
2. Open the `SessionManifest`, then in order: `environment_gate.run_gate` → `preprocess_batch`
   (Layer 0) → `run_layer1` → `run_layer2` → `run_layer3` → `finalize_pipeline`.
3. Each stage is wrapped with a `time.monotonic()` clock via `_mark`; **wall time is console-only and
   never enters any payload**.
4. After the manifest closes, **re-walk the entire chain from disk** (`SessionManifest.verify_chain`)
   and record `manifest_entries_verified` — *the run does not count unless its audit trail re-verifies
   from disk.*

## 6.2 The summary document

**`build_pipeline_summary(...)`** — the **pure**, deterministic batch summary (schema
`spovnob-pipeline-output-v1`, no wall-clock data anywhere): per-file facts (source SHA-256, durations,
offsets, Silero speech ms), the enrollment block (`e_composite_sha256`, `e_anti_sha256`,
`no_anti_profile`, `total_verified_ms`, pool sizes, final quality state), the Layer 2 block (output
hash, calibration ref + kind, θ values, total HIGH ms), the Layer 3 block (output hash, total clean /
contaminated ms, segment + NaN block counts), and the full **`clean_segments`** listing (the phase's
final deliverable — file index, local+global PTS, duration, bridged-gap count, WAV path + SHA-256).

**`finalize_pipeline(...)`** writes the summary to `<work_dir>/pipeline_output.json` (canonical JSON),
runs `validate_time_fields` over it, records a **`pipeline_complete`** entry with the summary hash and
the headline totals, and returns `{summary, output_path, output_sha256}`. `pipeline_complete` is the
designated terminal operation — the anchor that makes tail-truncation detectable (§1.1.6).

## 6.3 CLI and self-test

`_main` supports `--selftest` or `--run` (with `--videos`, `--clicks`, `--work-dir`, `--model-store`,
`--manifest`, `--operator`); on success it prints the clean/excluded totals, the summary path + hash,
the verified entry count, and per-stage timings. The self-test fabricates layer results (a 2-file
batch, a STRONG enrollment, a DERIVED calibration, a Layer 3 with one bridged segment and one NaN
block) and asserts the summary is deterministic + Rule-6-valid + serializable, then exercises
`finalize_pipeline` into a real temp manifest: the `pipeline_complete` entry is recorded once, the
chain re-verifies, the written file equals `canonical_json(summary) + "\n"`, and the recorded hash
matches.

---

# Part 7 — Operator Tooling

Four operational components surround the pipeline. None of them ever writes the audit manifest; their
outputs are re-validated from scratch by the pipeline.

## 7.1 `env.sh` — the runtime bridge (11 lines, `source` before any run)

Activates the pinned `.venv` and builds `LD_LIBRARY_PATH` so ONNXRuntime's CUDA-12 provider resolves
on the CUDA-13 host. It locates the `nvidia/` site-packages dir and torch's `lib` dir at runtime and
prepends `cufft/lib`, `cuda_runtime/lib`, `cublas/lib`, and the torch lib dir, then exports
`SPOVNOB_MODEL_STORE`. Without `env.sh`, ORT's CUDA `.so` fails to load and the gate's Step-7 FLAG 2
check catches the silent CPU fallback (§1.2.5).

## 7.2 `run.sh` — the one-command batch runner (90 lines)

`./run.sh <videos_dir> [batch_name]`. In order: resolve the videos dir against the caller's CWD, `cd`
into the project, `source env.sh`; collect videos in **sorted (canonical) order** (the first
alphabetically is `file_index=0` — the one you click on); reuse `session/<batch>/clicks.json` if
present, else launch `click_ui.py` on video 0 and wait for the export; run `pipeline_runner.py --run`;
build the audit dashboard with `audit_visualizer.py`. It prints the clean-audio dir, the summary JSON,
and the dashboard path. `set -euo pipefail`; tolerates EOF on the confirm prompt for non-interactive
runs.

## 7.3 `click_ui.py` — the operator clicking UI (1872 lines, Flask)

A local, single-file Flask app for producing `clicks.json`. The operator scrubs video 0
frame-by-frame, sees exactly the face detections / MAR values / Silero VAD state the pipeline will see,
and registers the `speaking_click` (and optional `anti_click`) with **live guardrail validation** in
the browser, before processing starts.

**Parity contract (the design rule):** the UI never re-implements pipeline logic. Every validation
**imports and calls production functions** — `vision.video_frame_pts_ms` / `vision.scan_video`,
`layer0_preprocessor.extract_audio` / `silero_window_probs` / `segments_from_window_probs`, and the
enrollment underscore-internals `_face_at_click` / `_build_obs` / `_run_machine` / `_match_face` /
`_mean_embedding` / `target_solo_vad_spans`. The underscore imports are deliberate: if `enrollment.py`
changes, the tool must break loudly rather than drift silently.

Key pieces:
- **`ClickSession`** — server-side state for one video. `_validate_speaking` mirrors `run_layer1`'s
  click handling exactly (resolve the clicked face → build the anchor obs stream → run the real
  `WindowMachine` → enforce guardrail 2 `seed_too_short` and guardrail 1 `overlap_at_speaking_click` →
  refine `F_target` over the seed span). `_validate_speaking_solo` is the **beard-mode** seed path
  (no MAR; the click must land inside a `target_solo_vad_spans` span — `click_not_in_solo_speech`
  otherwise). `_validate_anti` enforces guardrail 3 (`anti_click_matches_target`), the target-lips
  check (skipped in beard mode), and the Silero-energy check, validating against `F_target` (which only
  exists after the speaking click — ordering enforced). `set_beard` toggles the per-person beard flags
  and clears clicks when the target flag changes (the validation mode changed). `export_payload` writes
  the exact `clicks.json` Layer 1 expects (including `speaking_clicks`, `target_bearded`,
  `interviewer_bearded`, and omitting `anti_click` entirely on the NO_ANTI path).
- **`reason_message` / `no_anti_warning`** — translate the production reason strings into plain-English
  operator feedback.
- **Pre-scan cache** — the startup scan (audio strip + Silero + YOLO/InsightFace over every frame +
  display JPEGs) is cached on disk, keyed by the video SHA-256, the full `EnrollmentParams` payload, the
  vision batch constants, the device, and the model store's `expected_hashes.json` digest
  (`build_cache_key`). A warm start needs no models, no torch, no GPU — startup ~2 s.
- **Flask app (`create_app`)** — routes: `/` (index), `/api/meta`, `/api/timeline`,
  `/api/frame/<index>` (display JPEG), `/api/state`, `POST /api/click`, `POST /api/clear`,
  `POST /api/export`, `POST /api/beard`. All API responses are `no-store`.
- **Chain of custody:** writes **no** manifest entries; only `clicks.json` and the cache under
  `<work_dir>/ui_cache/`. `--cpu` exists for development only; the bench runs CUDA. The self-test
  (`--selftest`) drives the real imported validation functions over synthetic frames with no
  flask/torch/cv2/ffmpeg/GPU.

## 7.4 `audit_visualizer.py` — the forensic dashboard (1367 lines, stdlib only)

A standalone, **read-only** visualizer. It reads a finished manifest (and optionally the extracted 16
kHz audio) and emits **one self-contained HTML report** with an interactive, zoomable timeline on a
single global session clock: Silero VAD, Layer 2 per-block `S_target`/`S_interviewer` scores and tiers,
PyAnnote overlap regions, and the final Layer 3 CLEAN / NaN output blocks. Hovering any block yields a
plain-English verdict reconstructed from the recorded numbers.

**Independence contract:** it imports **nothing** from the pipeline — not even `session_manifest`. Two
reasons: (1) `environment_gate` mutates `os.environ` at import time, and a read-only forensic tool must
never perturb process state; (2) the report must run on an analyst's laptop with no torch, CUDA, model
store, or pip installs. The only accepted coupling is the manifest's on-disk format — the canonical-JSON
rule and the hash-chain layout are **re-implemented here, read-only**, and the schema string
`spovnob-manifest-v1` is asserted (if `session_manifest` ever bumps its schema, this tool says so
loudly).

Key functions: `verify_chain` (re-implemented, ON by default; a broken chain paints a large red banner)
+ `sha256_of_file` audio re-hashing vs the recorded `wav_sha256`; `load_manifest` / `normalize_entry`
(un-nests the double-wrapped worker-merged records); `l2_verdict` / `nan_verdict` / `clean_verdict`
(the hover explanations); `build_session` (assembles the global timeline from the entry stream);
`wav_peak_envelope` / `build_envelopes` (a capped peak envelope per file); `render_html` / `_render_panel`
(the SVG/HTML). `--no-verify`, `--no-browser`, `--audio DIR_OR_WAV ...`, `--out` flags. The self-test
builds a synthetic hash-chained manifest (including double-nested worker records and a tampered line)
and asserts parsing, the local→global mapping, every tier verdict, chain-break detection, and the
rendered HTML.

## 7.5 `diag_windows.py` — throwaway window-machine diagnostic (138 lines)

Loads a click-UI pre-scan pickle and runs the **real** `WindowMachine` to print what speaking windows
actually exist for the target — their durations, end reasons, and guardrail-1 overlap fractions —
plus a greedy identity clustering of all faces and the longest solo+speech stretches. Read-only,
diagnostic-only; uses `face_reid_threshold`, `seed_min_ms`, `click_overlap_max_frac`, `yaw_max_degrees`,
`mar_on`. Handy when an enrollment fails and you need to see why the click landed (or didn't) in a
clickable window.

---

# Part 8 — Determinism, Time, and Hashing (cross-cutting)

The engineering target is not "numerically stable" but **bit-identical**: the run must be a pure
function from (input files, model store, parameters) to bytes. Everything below serves that.

**GPU determinism** — the four CUDA constants (§1.2.3), float32 throughout (no AMP, no TF32), fixed
batch shapes (`ECAPA_BATCH_WINDOWS = 256`, `VISUAL_BATCH_FRAMES = 32`; Layer 1 enrollment windows are
batch-of-1 because they are variable-length *by design* — batch-of-1 *is* the fixed shape), and the ~10 s
startup workload checksum that *measures* (not asserts) that the constants are in force.

**Why batch shape matters:** FP addition is non-associative, so the kernel tiling over a `[256, T]`
tensor sums differently than over a `[200, T]` tensor. The batch size is therefore part of the numeric
function, recorded in the manifest. The repeat-padding trick (§4.4) keeps every coarse-scan forward
pass at exactly `[256, T]`.

**Order-independent CPU arithmetic:**
- **`math.fsum`** for every reduction that matters — pool means, duration-weighted means, variances,
  MAR vertical averages, dot products and norms in `cosine()`. `fsum` returns the correctly rounded
  exact sum (Shewchuk's algorithm), so the result is independent of summation order. A plain `sum()`
  over floats would make pool arithmetic depend on list order.
- **`statistics.median`** for per-block pooling (even counts average the two middle order statistics —
  a fixed, documented tie rule).
- **`decimal.Decimal` with `ROUND_HALF_EVEN`** for every ffprobe decimal-seconds → ms conversion.
- **Integer milliseconds everywhere** (Rule 6). 1 ms = 16 samples exactly at 16 kHz.

**Hashing:** `payload_sha256` (the cross-run-comparable, deterministic-content-only hash),
`entry_sha256` (the run-specific chain seal), per-file `model_checksum`, per-layer `output_hash`
(over canonical JSON), per-WAV `wav_sha256`, and the final `pipeline_complete` summary hash. The chain
is verified on open, after close, and by any auditor with `verify_chain`.

**Three time frames** (§0.8) with fixed conversion sites; Rule 6 enforces integerness at every payload
boundary so a float can never leak into the timeline.

---

# Part 9 — The Five Models

| Model | Role | Device | Loader |
|---|---|---|---|
| **Silero VAD** (~1 MB TorchScript, v4) | speech/silence segment map | **CPU** | `torch.jit.load` of the commit-pinned snapshot |
| **ECAPA-TDNN** (SpeechBrain `spkrec-ecapa-voxceleb`, C=1024, 192-dim) | speaker d-vector extraction + scoring | GPU | `EncoderClassifier.from_hparams(local dir)` |
| **YOLOv8m** | person detection per frame (gate) | GPU | `YOLO(yolov8m.pt)` |
| **InsightFace `buffalo_l`** (SCRFD det + ArcFace + 2d106det) | face biometric lock + landmarks + pose | GPU (ORT CUDA) | `FaceAnalysis(...).prepare(det_size=(640,640))` |
| **PyAnnote segmentation-3.0** (OVD) | overlapping-speech detection | GPU | `Model.from_pretrained` → `OverlappedSpeechDetection` |

All five are loaded **once** by the gate and held resident for the entire batch.

**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation — TDNN). A 1D-convolutional
speaker encoder built on SE-Res2Blocks (squeeze-excitation + Res2Net multi-scale), with
multi-layer feature aggregation and **attentive statistics pooling** (an attention-weighted mean+std over
time) producing a fixed 192-dim utterance embedding regardless of input length. Trained with
**AAM-Softmax (ArcFace)**, which optimizes *angular* margins between speakers — which is precisely why
SPOVNOB compares embeddings with **cosine** similarity and treats those cosines as the native metric,
never as probabilities. SPOVNOB uses the frozen VoxCeleb-trained encoder (paper EER ~0.69% on VoxCeleb)
and keeps only the encoder, discarding the clustering back end that diarization papers bolt on (SPOVNOB
replaces clustering with the operator-witnessed visual anchor).

**InsightFace `buffalo_l`** bundles SCRFD (face detection, `det_score` gate), an ArcFace-R100 recognition
head (the L2-normalized `normed_embedding` used for `F_target` and ReID), the **2d106det** 106-point
landmark model (the source of MAR — outer lip contour 52–71, nose 72–86), and a pose estimator
(`face.pose[1]` = yaw, used for MAR suspension). It is the **sole sanctioned ONNXRuntime exception**
(FLAG 4) — no pure-PyTorch alternative provides the 106-point landmarks MAR needs.

**Silero VAD** is a tiny TorchScript model run on CPU in 512-sample (32 ms) windows; it produces a
*map*, never a *mask* — the audio is never modified.

**PyAnnote segmentation-3.0 OVD** is a neural segmentation model (SincNet-style features + recurrent
layers, powerset cross-entropy covering up to ~3 concurrent speakers) wrapped so it marks frames with
≥2 simultaneous speakers. Its internal min-duration smoothing is zeroed so SPOVNOB owns the temporal
policy.

---

# Part 10 — Complete Parameter Reference

Every operator/value is exactly as the code uses it. Operators: `>` strict, `≥` inclusive — they are
never interchangeable.

## 10.1 Layer 0 (`layer0_preprocessor.py`)

| Constant | Value | Meaning |
|---|---|---|
| `SAMPLE_RATE` | 16000 | architectural; whole pipeline is 16 kHz mono |
| `SILERO_WINDOW_SAMPLES` | 512 | 32 ms VAD windows |
| `SILERO_THRESHOLD` | 0.50 | window is speech iff `prob >= 0.50` |
| `SILERO_MIN_SPEECH_MS` | 250 | drop speech runs shorter than this (pre-padding) |
| `SILERO_MIN_SILENCE_MS` | 100 | merge speech runs across gaps `< 100 ms` |
| `SILERO_SPEECH_PAD_MS` | 30 | widen survivors by this per side |
| `EXTRACT_WORKERS` | 8 | parallel ffmpeg threads |

## 10.2 Layer 1 (`EnrollmentParams`, schema v2)

| Param | Value | Operator/use |
|---|---|---|
| `face_reid_threshold` | 0.40 | keep target lock iff `cos ≥ 0.40` |
| `reid_warning_floor` | 0.50 | guardrail 6 running-mean warning `<` |
| `mar_on` | 0.15 | hysteresis open `>` |
| `mar_off` | 0.10 | hysteresis close-arm `<` |
| `plosive_ms` | 500 | plosive buffer length |
| `vad_tol_ms` | 50 | Silero/PTS alignment tolerance |
| `min_enroll_len_ms` | 2000 | discard Track A candidates shorter than this |
| `seed_min_ms` | 3000 | E_seed minimum (no maximum) |
| `int_lips_closed_frac` | 0.80 | Gate A: interviewer-closed fraction `≥` |
| `threshold_target` | 0.70 | Gate B: `cos(window, E_seed) ≥` |
| `threshold_anti` | 0.50 | Gate C: `cos(window, E_anti) ≤` |
| `margin_minimum` | 0.15 | Gate C: `cos_seed − cos_anti ≥` |
| `mtrap_sim_max` | 0.60 | M-Trap: discard Track B if `cos(·,E_seed) >` |
| `anti_contam_warning` / `anti_contam_halt` | 0.45 / 0.60 | guardrail 8 `sim(E_composite,E_anti)` |
| `pool_var_warning` | 0.05 | anti-pool variance-increase warning |
| `yaw_max_degrees` | 35.0 | suspend MAR beyond `|yaw| >` |
| `ema_span` | 5 | causal EMA span (α = 1/3, pre-seeded) |
| `upper_inner_lip` / `lower_inner_lip` / `mouth_width_pair` | (71,63,68) / (62,54,57) / (52,61) | corrected 2d106det MAR indices |
| `insightface_min_det_score` | 0.50 | guardrail 5: below ⇒ not detected |
| `yolo_min_conf` | 0.30 | YOLO person gate |
| `silence_stride` | 1 | optional visual-scan stride (1 = off) |
| `encode_max_ms` / `encode_overlap_ms` | 60000 / 2000 | single-pass cap / chunk overlap |
| `gate_a_vad_min_coverage` | 0.50 | Gate A VAD coverage `≥` |
| `trackb_window_ms` / `trackb_min_spacing_ms` | 2000 / 2000 | Track B window / dedupe spacing |
| `click_overlap_max_frac` | 0.20 | guardrail 1 visual overlap proxy `>` fails |
| `strong_ms` / `strong_ms_no_anti` | 45000 / 60000 | STRONG threshold (NO_ANTI escalation) |
| `marginal_ms` | 20000 | MARGINAL floor |
| `variance_high` | 0.05 | "high variance" for quality / final warning |
| `audio_anchor_accept_sim` | 0.78 | anchored Gate B (beard path) `≥` |
| `audio_anchor_collect_sim` | 0.55 | provisional-anchor collection `≥` |
| `audio_anchor_consistency_min` | 0.65 | seed outlier floor `<` |
| `audio_solo_min_ms` | 2000 | min target-solo+VAD span |
| `audio_solo_face_max_others` | 0 | "solo" = this many other faces |
| `CLICK_MATCH_MAX_GAP_MS` (module const) | 200 | nearest analyzed frame to a click |

## 10.3 Layer 2 (`Layer2Params`)

| Param | Value | Use |
|---|---|---|
| `window_ms` / `hop_ms` / `block_ms` | 5000 / 1000 / 1000 | sliding window, hop, block grid |
| `silero_skip_floor` | 0.20 | skip windows with `< 20%` speech overlap |
| `evidence_floor` | 0.20 | SUB tier floor `≥` |
| `edge_fine_window_ms` / `edge_fine_hop_ms` | 2000 / 250 | edge-trim fine window / hop |
| `edge_scan_span_ms` | 2000 | fine scan reaches ±this |
| `edge_min_fine_window_ms` | 1000 | skip clamped fine windows below this |
| `edge_max_trim_ms` | 750 | trim beyond this → demote the edge block |
| `genuine_quantile` | 0.10 | q10 of genuine LOO scores |
| `impostor_safety_margin` | 0.05 | `max(impostor) + 0.05` |
| `theta_clamp_low` / `theta_clamp_high` | 0.45 / 0.75 | θ_high clamp (high = overlap detector) |
| `theta_med_step` / `theta_med_floor` | 0.15 / 0.30 | θ_med = max(θ_high − 0.15, 0.30) |
| `min_calibration_windows` | 10 | below ⇒ FALLBACK_DEFAULTS |
| `fallback_theta_high` / `fallback_theta_med` | 0.60 / 0.40 | FALLBACK values |
| `no_anti_theta_floor` | 0.55 | DERIVED_NO_ANTI floor |
| `margin_minimum` | 0.15 | dual-target margin (NOT calibrated) |
| `anti_contam_warning` / `anti_contam_halt` | 0.45 / 0.60 | Step-1 sanity re-check |
| `ratio_normal` / `ratio_low` | 0.25 / 0.10 | activity-ratio tiers |
| `drift_window_ms` / `drift_delta` | 30000 / 0.10 | cross-video drift |

## 10.4 Layer 3 (`Layer3Params`)

| Param | Value | Use |
|---|---|---|
| `merge_gap_ms` | 400 | bridge clean gaps `< 400 ms` only |
| `gap_dominance_guard` | True | refuse bridges over interviewer-evidence blocks |

## 10.5 Environment gate constants

`ECAPA_BATCH_WINDOWS = 256`, `VISUAL_BATCH_FRAMES = 32`, `TORCH_NUM_THREADS = 8`,
`GLOBAL_SEED = 20260611`, `INSIGHTFACE_DET_SIZE = (640, 640)`,
`PYANNOTE_OVD_HYPERPARAMS = {min_duration_on: 0.0, min_duration_off: 0.0}`,
`EXPECTED_CUDA_VERSION = "12.1"`, `EXPECTED_PYTHON_PREFIX = "3.10."`.

---

# Part 11 — Manifest Operation Vocabulary

Cross-layer (`session_manifest.Operation`): `batch_init`, `model_checksum`, `determinism_check`,
`parameter_modified`, `enrollment_vector`, `enrollment_discard`, `calibration`, `video_gap`,
`drift_notice`, `warning`, `blocking_halt`, `destructive_op`, `output_hash`, `worker_log_merged`.

Layer 0: `layer0_file`, `video_gap`, `batch_init`.
Layer 1: `layer1_init`, `layer1_seed`, `layer1_video_scan`, `layer1_quality`, `layer1_freeze`
(+ `enrollment_vector` / `enrollment_discard`). Warnings include `reclick_required`,
`frame_pts_mismatch`, `pose_unavailable`, `low_detection_quality`, `anti_pool_variance_increase`,
`enrollment_contamination`, `seed_anchor_outlier`, `high_pool_variance_operator_review`. Halts include
`speaking_click_not_on_first_video`, `anti_click_not_on_first_video`,
`enrollment_contamination_critical`, `critical_enrollment_failure`.
Layer 2: `layer2_init`, `layer2_window_skipped`, `layer2_block`, `layer2_edge_trim`,
`layer2_file_summary`, `calibration`, `output_hash`; warnings `CALIBRATION_OVERLAP`,
`layer2_enrollment_contamination`, `sanity_check_unavailable_no_anti_profile`,
`near_zero_activity_manual_review`; halt `layer2_enrollment_contamination_critical`.
Layer 3: `layer3_init`, `layer3_ovd_regions`, `layer3_nan_block`, `layer3_segment`,
`layer3_file_summary`, `output_hash`; halt `layer3_requires_authoritative_layer2`.
Runner: `pipeline_complete` (the designated terminal entry).

Gate halt reasons: `wrong_platform`, `wrong_python`, `environment_variable_drift`, `ffmpeg_missing`,
`version_pin_mismatch`, `forbidden_module_imported`, `hash_registry_missing`, `hash_registry_bad_schema`,
`model_dirs_missing`, `model_checksum_failure`, `wrong_cuda_version`, `cuda_unavailable`,
`onnxruntime_cuda_provider_missing`, `onnxruntime_cuda_probe_missing`,
`onnxruntime_cuda_provider_inactive`, `gpu_determinism_failure`.

---

# Part 12 — Setup & Deployment

Recommended install root `/opt/spovnob/` with `code/`, `.venv/`, `wheelhouse/`, `model_store/`,
`session/`. Full procedure (consolidated from the Ubuntu guide):

1. **System prerequisites:** `build-essential git curl unzip python3.10 python3.10-venv python3.10-dev
   ffmpeg libsndfile1 libglib2.0-0`. The box runs the **580-series driver (CUDA 13.x host)**; do **not**
   apt-install `nvidia-cuda-toolkit` (the `+cu121` wheels bundle CUDA 12.1 + cuDNN 8.9 as pip packages).
2. **Python env:** `python3.10 -m venv .venv`; `pip install --upgrade "pip==24.0"`.
3. **Packages:** the single source of truth is `requirements.txt` (23 pins + two extra wheel indexes —
   PyTorch cu121 and the ONNXRuntime CUDA-12 feed). Staging: `pip download -r requirements.txt -d wheelhouse`;
   air-gapped: `pip install --no-index --find-links wheelhouse -r requirements.txt`. After install,
   `pip freeze > requirements.lock` and record its SHA-256.
4. **Model vendoring (five models):** Silero VAD git snapshot pinned to **commit**
   `915dd3d639b8333a52e001af095f87c5b7f1e0ac` (the `v4.0` tag was moved upstream); SpeechBrain ECAPA;
   YOLOv8m; InsightFace `buffalo_l` (into `insightface/models/buffalo_l/`); PyAnnote segmentation-3.0
   (gated — accept conditions first). Then `environment_gate.py --freeze-hashes` once on the staging box
   (writes `expected_hashes.json`, chmod 444).
5. **Runtime exports:** `source env.sh` (or the equivalent `LD_LIBRARY_PATH` with `cufft/lib`,
   `cuda_runtime/lib`, `cublas/lib`, torch lib). The determinism env vars are set by `environment_gate`
   at import time — they do **not** need exporting.
6. **Verify:** `environment_gate.py --run` (the go/no-go). Success: *"environment gate PASSED — all
   checks recorded in manifest."*

**The seven setup gotchas (consolidated from `Ubuntu_Setup_Gotchas.md`):**
1. **PyTorch CUDA libs are hidden inside the `torch` pip package** → export their dir to
   `LD_LIBRARY_PATH` (this is what `env.sh` does).
2. **ONNXRuntime silently installs the CUDA-11.8 wheel** because PyPI has the same version number →
   force the Microsoft CUDA-12 feed with `--index-url`. (The bench actually had to run an explicit
   `pip install nvidia-cufft-cu12==11.0.2.54 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1`
   — the torch wheel did **not** auto-vendor cufft/cudart as expected.)
3. **SpeechBrain crashes under `HF_HUB_OFFLINE=1`** if `label_encoder.txt` is missing → make the
   vendored ECAPA dir *complete* (preferred for air-gap) rather than relying on the global HF cache,
   which is outside the hash registry.
4. **`expected_hashes.json` permission denied on re-freeze** → it is chmod 444 on purpose; `rm -f` it
   first.
5. (Node.js for Claude Code — environment-only, not pipeline.)
6. **The `ManifestTimeError` Layer 0 crash — already fixed** in commit `7dc3daa` (nested
   `silero_segments` shape); just `git pull`, do not apply the old manual rename.
7. **The 2d106det landmark indices — already fixed** in `params.py` (the corrected
   `(71,63,68)/(62,54,57)/(52,61)` MAR pairs; +3 windows, +7 250 ms clean audio).

---

# Part 13 — The Self-Test Architecture

Every module from 0a onward ships a **stdlib-only self-test** runnable on a bare Python 3.10 with zero
pip installs, no torch, no cv2, no numpy, no GPU, no ffmpeg:

```bash
python3 session_manifest.py
python3 environment_gate.py --selftest
python3 layer0_preprocessor.py --selftest
python3 -m layer1_enrollment --selftest
python3 layer2_tracker.py --selftest
python3 layer3_contamination.py --selftest
python3 pipeline_runner.py --selftest
python3 click_ui.py --selftest
python3 audit_visualizer.py --selftest
```

The seams that make this possible are **architectural, not scaffolding**: pure functions split from
GPU/IO; injectable `Scorer` (Layer 2) and `OverlapProvider` (Layer 3) seams so the full flow runs
against synthetic scores; injectable `version_of` / `probe` in the gate; lazy `import torch` inside the
functions that touch the model. Each self-test asserts `"torch" not in sys.modules` at start and end,
so a stray heavy import is caught immediately. The self-tests cover the parts the air-gapped
zero-dependency policy cannot otherwise reach (e.g. Layer 0's Rule-6 payload shape, which a real run
needs ffmpeg+Silero to exercise).

---

# Part 14 — Bench-Validation Register & Known Limitations

**Validated on the bench:**
- **MAR landmark correction** (2026-06-12, NT-clip27 @ 42000 ms): the doc indices were wrong; the
  corrected outer-lip-inner-edge pairs give range ~0.10–0.25 (vs ~0.44–0.57 constant) →
  +3 enrollment windows, +7 250 ms clean audio. Hysteresis recalibrated to `mar_on=0.15`/`mar_off=0.10`;
  the Silero VAD gate added to the window machine's start/resume/close transitions.
- **First full pipeline run** (all 4 NT clips): STRONG quality, 8 segments, ~225.75 s clean.
- **Audio-anchored / beard path** (2026-06-17/18, UB-clip2): geometric MAR stuck ~0.20–0.25 under a
  dense beard; the solo+VAD + ECAPA-consensus path turned a 3.84 s crash into 85 s STRONG. Anchored Gate
  B `0.78` derived from target-vs-consensus ~0.83 / interviewer-vs-consensus ~0.69.
- **FLAG 2 hardening** (2026-06-15/16): `check_onnxruntime_cuda` now builds a real `InferenceSession`
  on `buffalo_l/det_10g.onnx` and asserts the active provider is CUDA — catching the
  `libcublasLt.so.11`/`libcufft.so.11` silent-CPU-fallback that `get_available_providers()` alone misses.

**Known limitations / open items:**
- **Tail-truncation** of the manifest is not detectable from the chain alone (mitigated by the terminal
  `pipeline_complete` entry, the post-close re-verify + entry count, and the recommendation to retain
  the final `entry_sha256` off-box).
- **SpeechBrain auxiliary-file fallback** to the global HF cache (outside the hash registry) if the
  vendored ECAPA dir is incomplete — fix by making the vendored dir complete (Gotcha #3).
- **`face.pose` availability** depends on the InsightFace build; if absent, yaw suspension is inactive
  and a per-video `pose_unavailable` warning is logged (visible degradation, never silent).
- **Cross-architecture determinism** is checked (via the recorded `gpu_workload_checksum`), not assumed:
  deterministic-per-machine is the guarantee.
- **Track B uses raw (not smoothed) MAR** for candidate triggering — a review-flagged simplification.
- **Behavioral / paralinguistic analysis** is deferred to a separate design phase; the clean WAVs +
  sidecars are its input contract.

---

# Appendix A — File-by-File Line / Function Map

| File | Lines | Key functions / classes |
|---|---|---|
| `session_manifest.py` | 492 | `canonical_json`, `sha256_*`, `validate_time_fields`, `Operation`, `SessionManifest` (`append`/`guard_destructive`/`verify_chain`), `WorkerLog`, `merge_worker_logs` |
| `environment_gate.py` | 713 | `DETERMINISM_ENV`, `PINNED_VERSIONS`, `ResidentModels`, `check_*` (platform/ffmpeg/versions/forbidden/onnxruntime_cuda), `freeze_model_hashes`, `verify_model_store`, `enforce_torch_determinism`, `gpu_determinism_selftest`, `load_resident_models`, `run_gate` |
| `layer0_preprocessor.py` | 648 | `ms_from_samples`, `decimal_seconds_to_ms`, `parse_probe`, `segments_from_window_probs`, `FileAudio`/`BatchAudio`, `layer0_file_payload`, `extract_audio`, `silero_window_probs`, `preprocess_batch` |
| `layer1_enrollment/params.py` | 111 | `EnrollmentParams` (schema v2), `PARAM_SCHEMA_VERSION` |
| `layer1_enrollment/errors.py` | 25 | `Layer1Error`, `Layer1ReclickError` |
| `layer1_enrollment/geometry.py` | 78 | `compute_mar`, `yaw_suspends_mar`, `CausalEMA` |
| `layer1_enrollment/window_machine.py` | 253 | `FrameObs`, `CandidateWindow`, `WindowMachine` (`step`/`finalize`) |
| `layer1_enrollment/gates.py` | 148 | `evaluate_triple_gate`, `mtrap_discard`, `contamination_level`, `pairwise_cosine_variance`, `segment_overlap_ms`, `vad_near` |
| `layer1_enrollment/quality.py` | 48 | `assess_quality` |
| `layer1_enrollment/encoding.py` | 158 | `l2_normalize`, `cosine`, `duration_weighted_mean`, `plan_chunks`, `pcm_slice`, `ecapa_encode_pcm`, `encode_window` |
| `layer1_enrollment/vision.py` | 194 | `FaceObs`, `FrameFaces`, `video_frame_pts_ms`, `scan_video` |
| `layer1_enrollment/enrollment.py` | 1129 | clicks, `_face_at_click`/`_build_obs`/`_run_machine`, `target_solo_vad_spans`/`outlier_seed_indices`, `run_layer1` |
| `layer1_enrollment/selftest.py` | 361 | stdlib self-test of every pure submodule |
| `layer1_enrollment/__main__.py` | 75 | CLI |
| `layer2_tracker.py` | 1197 | `Layer2Params`, `quantile_sorted`, `loo_scores`, `derive_thresholds`, `plan_windows`, `tier_block`, `median_pool_blocks`, `find_high_runs`, `refine_run_edges`, `_ecapa_scorer`, `track_file`, `run_layer2` |
| `layer3_contamination.py` | 808 | `intervals_intersect`, `merge_regions`, `classify_run_blocks`, `gap_dominance_blocked`, `assemble_segments`, `pyannote_overlap_provider`, `_write_segment_wav`, `contaminate_file`, `run_layer3` |
| `pipeline_runner.py` | 403 | `build_pipeline_summary`, `finalize_pipeline`, `run_pipeline` |
| `click_ui.py` | 1872 | `ClickSession` (validation), `prepare_session`, `create_app` (Flask routes) |
| `audit_visualizer.py` | 1367 | `verify_chain`, `build_session`, `*_verdict`, `render_html` |
| `diag_windows.py` | 138 | window-machine diagnostic (read-only) |

# Appendix B — Glossary

- **E_seed** — the operator-anchored seed d-vector (Gate B reference + M-Trap reference).
- **E_composite** — the duration-weighted-mean target profile, recomputed per video, **frozen** at
  Layer 1 end. Its SHA-256 is the `enrollment_ref`.
- **E_anti** — the interviewer anti-profile (Track B auto + Track C click), powers Gate C and Layer 2's
  dual-target margin.
- **F_target / F_interviewer** — ArcFace face embeddings used for visual identity matching (not audio).
- **S_target / S_interviewer** — raw per-window/per-block cosine scores (NOT probabilities).
- **θ_high / θ_med** — per-session calibrated tiering thresholds.
- **HIGH / MEDIUM / SUB / REJECT / SKIPPED_NONSPEECH** — Layer 2 block tiers; only HIGH reaches Layer 3.
- **NaN block** — a HIGH block voided by overlap; the *string* `"NaN"`, permanently excluded.
- **Triple Gate (A/B/C)** — Layer 1's per-window acceptance: VAD/visual (A), seed similarity (B), anti
  rejection + margin (C).
- **M-Trap** — the guard that stops the target's own closed-lip phonemes from poisoning E_anti.
- **Canonical Merge Rule** — the deterministic four-tuple sort that makes parallel worker logs
  byte-identical regardless of scheduling.
- **Audio-anchored path** — the beard / unreliable-MAR enrollment route (solo+VAD spans + ECAPA
  consensus, no lip-reading).
- **PTS** — container Presentation Timestamps; the only time source (never frame indices).

---

*SPOVNOB Master Reference — generated from the code at repository head. This document replaces
`Audio_Diarization.md`, `SPOVNOB_COMPLETE_CONTEXT.md`, `SPOVNOB_TECHNICAL_DEEP_DIVE.md`, and
`SPOVNOB_PIPELINE_SUMMARY.md`. Where this document and the code ever disagree, the code wins.*
