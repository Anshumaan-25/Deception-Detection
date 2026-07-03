#!/usr/bin/env python3
"""
SPOVNOB Batch Queue Processing Daemon
======================================
Defense-Grade Autonomous Session Factory.

Monitors SPOVNOB_intake/ for session_profile.json manifests,
orchestrates GPU-throttled parallel processing via asyncio + subprocess spawn,
and maintains an idempotent persistent ledger for crash-safe resumption.

Target Hardware: 44-Core CPU / 512GB ECC RAM / RTX 6000 Ada (48GB VRAM)
"""

import os
import sys
import json
import time
import fcntl
import signal
import bisect
import asyncio
import resource
import logging
import tempfile
import traceback
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from datetime import datetime, timezone

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ═══════════════════════════════════════════════════════════════════
# 0. KERNEL RESOURCE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def maximize_file_descriptors():
    """
    Programmatically maximize the soft limit of open file descriptors
    to match the operating system's absolute hard-limit ceiling.
    Prevents FD exhaustion crashes during high-concurrency batch runs.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        logging.info(f"RLIMIT_NOFILE raised: {soft} → {hard}")
    else:
        logging.info(f"RLIMIT_NOFILE already at hard limit: {hard}")

maximize_file_descriptors()

# ═══════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTAKE_DIR = PROJECT_ROOT / "SPOVNOB_intake"
OUTPUT_DIR = PROJECT_ROOT / "pipeline_system_outputs"
LEDGER_PATH = OUTPUT_DIR / "batch_ledger.json"

INTAKE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CPU_WORKERS = 12
GPU_SEMAPHORE_LIMIT = 2

VALID_STATES = [
    "QUEUED", "AUDIO_PROCESSING", "TENSORRT_ACTIVE",
    "MATH_NORMALIZATION", "COMPLETED", "FAILED", "INTERRUPTED"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BatchDaemon")

# ═══════════════════════════════════════════════════════════════════
# 2. PERSISTENT IDEMPOTENT LEDGER ENGINE
# ═══════════════════════════════════════════════════════════════════

class BatchLedger:
    """
    Disk-persistent state tracker for all session processing.
    Uses atomic file replacement (os.replace) to prevent corruption
    during mid-write OS shutdown events.
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.entries = {}
        self._load()

    def _load(self):
        if self.ledger_path.exists():
            with open(self.ledger_path, "r") as f:
                self.entries = json.load(f)
            logger.info(f"Ledger loaded: {len(self.entries)} entries from disk.")
        else:
            self.entries = {}
            logger.info("No existing ledger found. Starting fresh.")

    def flush(self):
        """Atomic write: write to temp file, then os.replace to prevent corruption."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.ledger_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(self.entries, f, indent=2)
            os.replace(tmp_path, str(self.ledger_path))
        except Exception:
            # If replace fails, clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def get_state(self, session_id: str) -> str:
        entry = self.entries.get(session_id, {})
        return entry.get("state", None)

    def set_state(self, session_id: str, state: str, extra: dict = None):
        if session_id not in self.entries:
            self.entries[session_id] = {
                "session_id": session_id,
                "queued_at": datetime.now(timezone.utc).isoformat(),
            }
        self.entries[session_id]["state"] = state
        self.entries[session_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        if extra:
            self.entries[session_id].update(extra)
        self.flush()

    def is_terminal(self, session_id: str) -> bool:
        state = self.get_state(session_id)
        return state in ("COMPLETED", "FAILED")

    def get_active_sessions(self) -> list:
        return [
            sid for sid, entry in self.entries.items()
            if entry.get("state") not in ("COMPLETED", "FAILED")
        ]

    def mark_interrupted(self):
        """Flag all non-terminal sessions as INTERRUPTED for graceful shutdown."""
        for sid, entry in self.entries.items():
            if entry.get("state") not in ("COMPLETED", "FAILED", "INTERRUPTED"):
                entry["state"] = "INTERRUPTED"
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.flush()

# ═══════════════════════════════════════════════════════════════════
# 3. ELAN ANNOTATION BINARY SEARCH INTERVAL TREE
# ═══════════════════════════════════════════════════════════════════

class ELANAnnotationMapper:
    """
    Chronological binary search interval tree utilizing bisect.bisect_right()
    for mapping feature matrix window timestamps to ELAN behavioral annotations.
    """

    def __init__(self, elan_file_path: str = None):
        self.intervals = []
        self.starts = []

        if elan_file_path is None:
            return

        path = Path(elan_file_path)
        if not path.exists():
            logger.warning(f"ELAN file not found at {path}. Defaulting to 'unlabeled'.")
            return

        try:
            # ELAN exports are typically CSV/TSV with columns:
            # start_ms, end_ms, annotation_value (or similar)
            df = pd.read_csv(path, sep=None, engine="python")

            # Normalize column names to lowercase
            df.columns = [c.strip().lower() for c in df.columns]

            # Attempt to find time and annotation columns
            start_col = None
            end_col = None
            label_col = None

            for c in df.columns:
                if "start" in c and ("ms" in c or "time" in c or "begin" in c):
                    start_col = c
                elif "end" in c and ("ms" in c or "time" in c):
                    end_col = c
                elif "annot" in c or "label" in c or "value" in c or "tier" in c:
                    label_col = c

            # Fallback: use first three columns positionally
            if start_col is None and len(df.columns) >= 3:
                start_col = df.columns[0]
                end_col = df.columns[1]
                label_col = df.columns[2]

            if start_col is None or end_col is None or label_col is None:
                logger.error("ELAN CSV structure unrecognized. Need start/end/label columns.")
                return

            for _, row in df.iterrows():
                try:
                    start_ms = float(row[start_col])
                    end_ms = float(row[end_col])
                    label = str(row[label_col]).strip().lower()
                    self.intervals.append((start_ms, end_ms, label))
                except (ValueError, TypeError):
                    continue

            self.intervals.sort(key=lambda x: x[0])
            self.starts = [x[0] for x in self.intervals]
            logger.info(f"ELAN mapper loaded: {len(self.intervals)} annotation intervals.")
        except Exception as e:
            logger.error(f"Failed to parse ELAN file: {e}")
            self.intervals = []
            self.starts = []

    def lookup(self, timestamp_ms: float) -> str:
        """
        O(log N) interval search using bisect.bisect_right().
        Returns the annotation label string, or 'unlabeled' if no match.
        """
        if not self.starts:
            return "unlabeled"

        idx = bisect.bisect_right(self.starts, timestamp_ms) - 1

        if idx >= 0:
            start_ms, end_ms, label = self.intervals[idx]
            if timestamp_ms < end_ms:
                return label

        return "unlabeled"


def inject_elan_labels(csv_path: str, elan_mapper: ELANAnnotationMapper):
    """
    Reads the final calibrated/windowed CSV, computes each window's middle
    timestamp, maps it through the ELAN interval tree, and injects the
    `target_ground_truth` column. Overwrites the CSV in place.
    """
    df = pd.read_csv(csv_path)

    if "start_time_ms" in df.columns and "end_time_ms" in df.columns:
        midpoints = (df["start_time_ms"] + df["end_time_ms"]) / 2.0
        df["target_ground_truth"] = midpoints.apply(elan_mapper.lookup)
    else:
        df["target_ground_truth"] = "unlabeled"

    df.to_csv(csv_path, index=False)
    logger.info(f"ELAN labels injected into {csv_path}")

# ═══════════════════════════════════════════════════════════════════
# 4. FILE LOCK VERIFICATION (fcntl Non-Blocking)
# ═══════════════════════════════════════════════════════════════════

def is_file_transfer_complete(file_path: Path) -> bool:
    """
    Attempts a non-blocking exclusive lock on the target file.
    If the lock succeeds, the file is fully written and not held by
    another process (e.g., rsync, scp, NFS copy). Immediately releases.
    """
    try:
        with open(file_path, "rb") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return True
    except (IOError, OSError):
        return False

# ═══════════════════════════════════════════════════════════════════
# 5. GPU-ISOLATED SUBPROCESS WORKER (multiprocessing spawn context)
# ═══════════════════════════════════════════════════════════════════

def _gpu_worker_entrypoint(
    session_id: str,
    canonical_mp4: str,
    canonical_wav: str,
    output_root: str,
    yolo_path: str,
    session_manifest_path: str
):
    """
    Spawned in a completely isolated subprocess via multiprocessing.get_context('spawn').
    This guarantees a fresh CUDA context that is fully wiped when the subprocess terminates,
    preventing GPU memory leaks across sequential batch sessions.
    """
    # All GPU-touching imports happen INSIDE the subprocess to guarantee fresh CUDA init
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from main_pipeline import MultimodalProductionOrchestrator

    orchestrator = MultimodalProductionOrchestrator(
        output_root=output_root,
        yolo_path=yolo_path
    )
    orchestrator.process_video_session(
        canonical_mp4_path=canonical_mp4,
        canonical_wav_path=canonical_wav,
        session_id=session_id,
        session_manifest_path=session_manifest_path,
    )


def _gpu_worker_batch_entrypoint(
    recording_id: str,
    clips: list,
    output_root: str,
    yolo_path: str,
    diarization_output_json: str,
    offset_ms: int = 0,
    session_manifest_path: str = None,
    baseline_file_index: int = 0,
):
    """
    Batch counterpart of _gpu_worker_entrypoint (audio-diarization merge).
    Spawned in an isolated subprocess (fresh CUDA context). Runs the whole
    recording — N clips that share one audio-diarization enrollment — feeding
    each clip its verified target-speech segments via DiarizationBridge, then
    calibrates every clip against the dedicated baseline clip
    (baseline_file_index) and assembles the recording-level CSV.

    clips: list of (canonical_mp4, canonical_wav) pairs ordered by file_index.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from main_pipeline import MultimodalProductionOrchestrator

    orchestrator = MultimodalProductionOrchestrator(
        output_root=output_root,
        yolo_path=yolo_path,
    )
    orchestrator.process_recording_session(
        clips=clips,
        recording_id=recording_id,
        diarization_output_json=diarization_output_json,
        offset_ms=offset_ms,
        session_manifest_path=session_manifest_path,
        baseline_file_index=baseline_file_index,
    )

# ═══════════════════════════════════════════════════════════════════
# 6. ASYNC ORCHESTRATOR (Producer-Consumer Matrix)
# ═══════════════════════════════════════════════════════════════════

class BatchOrchestrator:
    """
    Decoupled asyncio-driven orchestrator.
    - CPU phases: up to MAX_CPU_WORKERS (12) concurrent paths.
    - GPU phases: throttled by asyncio.Semaphore(GPU_SEMAPHORE_LIMIT=2).
    Uses multiprocessing.get_context('spawn') for GPU isolation.
    """

    def __init__(self, ledger: BatchLedger):
        self.ledger = ledger
        self.gpu_semaphore = asyncio.Semaphore(GPU_SEMAPHORE_LIMIT)
        self.spawn_ctx = multiprocessing.get_context("spawn")
        self.active_processes = {}
        self.yolo_path = str(PROJECT_ROOT / "weights" / "yolov8n.pt")
        self._shutdown_event = asyncio.Event()

    def request_shutdown(self):
        self._shutdown_event.set()

    async def process_session(self, session_id: str, profile: dict, intake_session_dir: Path):
        """
        Full lifecycle handler for a single session.
        Acquires the GPU semaphore before launching the spawn-isolated pipeline.
        """
        if self._shutdown_event.is_set():
            return

        # ── BATCH MODE routing (audio-diarization merge) ─────────────
        # A recording bucket carries a shared pipeline_output.json (the audio-
        # diarization output for its N clips, produced out-of-band with the
        # operator click). Route it to the batch handler. Plain single-clip
        # buckets (no such file) fall through to the legacy path unchanged.
        if (intake_session_dir / "pipeline_output.json").exists():
            return await self.process_recording(session_id, profile, intake_session_dir)

        # Locate canonical files
        video_file = None
        wav_file = None
        for f in intake_session_dir.iterdir():
            if f.suffix == ".mp4":
                video_file = f
            elif f.suffix == ".wav":
                wav_file = f

        if video_file is None:
            self.ledger.set_state(session_id, "FAILED", {"error": "No .mp4 found in intake bucket"})
            logger.error(f"[{session_id}] No .mp4 found in {intake_session_dir}")
            return

        if wav_file is None:
            self.ledger.set_state(session_id, "FAILED", {"error": "No .wav found in intake bucket"})
            logger.error(f"[{session_id}] No .wav found in {intake_session_dir}")
            return

        # Verify file transfer completion via fcntl
        if not is_file_transfer_complete(video_file):
            logger.warning(f"[{session_id}] Video file locked by another process. Deferring.")
            self.ledger.set_state(session_id, "QUEUED", {"deferred_reason": "file_locked"})
            return

        if not is_file_transfer_complete(wav_file):
            logger.warning(f"[{session_id}] WAV file locked by another process. Deferring.")
            self.ledger.set_state(session_id, "QUEUED", {"deferred_reason": "file_locked"})
            return

        # Extract optional session manifest path
        session_manifest_path = profile.get("session_manifest_path", None)
        if session_manifest_path:
            session_manifest_path = str(intake_session_dir / session_manifest_path)

        # Acquire GPU Semaphore (blocks if 2 GPU workers are already active)
        logger.info(f"[{session_id}] Awaiting GPU semaphore ({GPU_SEMAPHORE_LIMIT} slots)...")
        async with self.gpu_semaphore:
            if self._shutdown_event.is_set():
                return

            self.ledger.set_state(session_id, "TENSORRT_ACTIVE", {
                "video_file": str(video_file),
                "wav_file": str(wav_file),
            })
            logger.info(f"[{session_id}] GPU semaphore acquired. Spawning isolated subprocess.")

            loop = asyncio.get_event_loop()
            try:
                process = self.spawn_ctx.Process(
                    target=_gpu_worker_entrypoint,
                    args=(
                        session_id,
                        str(video_file),
                        str(wav_file),
                        str(OUTPUT_DIR),
                        self.yolo_path,
                        session_manifest_path,
                    ),
                    daemon=False,
                )
                process.start()
                self.active_processes[session_id] = process

                # Wait for subprocess to complete without blocking the event loop
                await loop.run_in_executor(None, process.join)

                exit_code = process.exitcode
                del self.active_processes[session_id]

                if exit_code == 0:
                    # Post-processing: ELAN label injection
                    self.ledger.set_state(session_id, "MATH_NORMALIZATION")
                    elan_file = profile.get("elan_annotation_file", None)
                    elan_mapper = ELANAnnotationMapper(
                        str(intake_session_dir / elan_file) if elan_file else None
                    )

                    # Inject into calibrated CSV (or windowed if calibrated doesn't exist)
                    session_output = OUTPUT_DIR / session_id
                    calibrated_csv = session_output / f"{session_id}_calibrated_features.csv"
                    windowed_csv = session_output / f"{session_id}_windowed_features.csv"

                    target_csv = calibrated_csv if calibrated_csv.exists() else windowed_csv
                    if target_csv.exists():
                        await loop.run_in_executor(
                            None, inject_elan_labels, str(target_csv), elan_mapper
                        )

                    self.ledger.set_state(session_id, "COMPLETED", {
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "exit_code": exit_code,
                    })
                    logger.info(f"[{session_id}] ✅ Pipeline COMPLETED successfully.")
                else:
                    self.ledger.set_state(session_id, "FAILED", {
                        "exit_code": exit_code,
                        "error": f"Subprocess exited with code {exit_code}",
                    })
                    logger.error(f"[{session_id}] ❌ Pipeline FAILED (exit code {exit_code}).")

            except Exception as e:
                self.ledger.set_state(session_id, "FAILED", {"error": str(e)})
                logger.error(f"[{session_id}] ❌ Exception: {e}")
                traceback.print_exc()

    async def process_recording(self, recording_id: str, profile: dict, intake_session_dir: Path):
        """
        Batch lifecycle handler (audio-diarization merge).

        A recording bucket holds N clips (.mp4 + matching .wav) plus a shared
        pipeline_output.json produced out-of-band by the audio-diarization
        pipeline (operator click + run). Each clip is processed through the
        deception cascade with its verified target-speech segments, in ONE
        spawn-isolated subprocess (fresh CUDA context for the whole recording).
        The clip at profile["baseline_file_index"] (default 0) is the dedicated
        baseline video: every clip is z-scored against ITS stats, and the
        per-clip results are assembled into
        OUTPUT_DIR/<recording_id>/<recording_id>_recording_calibrated.csv.

        NOTE (parked): ELAN ground-truth injection stays single-clip-only — it
        is a training-corpus concern (production has no ground truth, ever) and
        no training is planned right now.
        """
        if self._shutdown_event.is_set():
            return

        diar_json = intake_session_dir / "pipeline_output.json"
        if not is_file_transfer_complete(diar_json):
            logger.warning(f"[{recording_id}] diarization output still locked. Deferring.")
            self.ledger.set_state(recording_id, "QUEUED", {"deferred_reason": "diar_json_locked"})
            return

        # Pair clips ↔ wavs ↔ diarization file_index (suffix-tolerant, ordered).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from recording_intake import collect_recording_clips
        try:
            clips = collect_recording_clips(intake_session_dir, diar_json)
        except Exception as e:
            self.ledger.set_state(recording_id, "FAILED", {"error": f"clip collection failed: {e}"})
            logger.error(f"[{recording_id}] clip collection failed: {e}")
            return

        if not clips:
            self.ledger.set_state(recording_id, "FAILED", {"error": "no clips resolved from pipeline_output.json"})
            logger.error(f"[{recording_id}] no clips resolved from pipeline_output.json")
            return

        # Every clip's files must be fully transferred before we spawn.
        for mp4, wav in clips:
            if not (is_file_transfer_complete(Path(mp4)) and is_file_transfer_complete(Path(wav))):
                logger.warning(f"[{recording_id}] clip files still locked. Deferring.")
                self.ledger.set_state(recording_id, "QUEUED", {"deferred_reason": "clip_locked"})
                return

        offset_ms = int(profile.get("offset_ms", 0))
        baseline_file_index = int(profile.get("baseline_file_index", 0))
        # Optional investigative session manifest (ContextMapper), resolved
        # relative to the bucket — mirrors the single-clip path above.
        session_manifest_path = profile.get("session_manifest_path", None)
        if session_manifest_path:
            session_manifest_path = str(intake_session_dir / session_manifest_path)
        logger.info(f"[{recording_id}] batch mode: {len(clips)} clip(s), "
                    f"baseline_file_index={baseline_file_index}. Awaiting GPU semaphore...")
        async with self.gpu_semaphore:
            if self._shutdown_event.is_set():
                return

            self.ledger.set_state(recording_id, "TENSORRT_ACTIVE", {
                "mode": "batch",
                "clip_count": len(clips),
                "diarization_output": str(diar_json),
            })
            logger.info(f"[{recording_id}] GPU semaphore acquired. Spawning batch subprocess.")

            loop = asyncio.get_event_loop()
            try:
                process = self.spawn_ctx.Process(
                    target=_gpu_worker_batch_entrypoint,
                    args=(
                        recording_id,
                        clips,
                        str(OUTPUT_DIR),
                        self.yolo_path,
                        str(diar_json),
                        offset_ms,
                        session_manifest_path,
                        baseline_file_index,
                    ),
                    daemon=False,
                )
                process.start()
                self.active_processes[recording_id] = process

                await loop.run_in_executor(None, process.join)

                exit_code = process.exitcode
                del self.active_processes[recording_id]

                if exit_code == 0:
                    # Baseline calibration + recording assembly now happen
                    # inside the subprocess (process_recording_session).
                    # Per-clip outputs: OUTPUT_DIR/<recording_id>_<file_index>;
                    # recording-level: OUTPUT_DIR/<recording_id>/. ELAN
                    # injection stays single-clip-only (see docstring).
                    self.ledger.set_state(recording_id, "COMPLETED", {
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "exit_code": exit_code,
                        "mode": "batch",
                        "clip_count": len(clips),
                    })
                    logger.info(f"[{recording_id}] ✅ Batch ({len(clips)} clips) COMPLETED.")
                else:
                    self.ledger.set_state(recording_id, "FAILED", {
                        "exit_code": exit_code,
                        "error": f"Subprocess exited with code {exit_code}",
                    })
                    logger.error(f"[{recording_id}] ❌ Batch FAILED (exit code {exit_code}).")

            except Exception as e:
                self.ledger.set_state(recording_id, "FAILED", {"error": str(e)})
                logger.error(f"[{recording_id}] ❌ Exception: {e}")
                traceback.print_exc()

    def terminate_active(self):
        """Force-terminate any running subprocess workers during shutdown."""
        for sid, proc in self.active_processes.items():
            if proc.is_alive():
                logger.warning(f"[{sid}] Terminating active subprocess (PID {proc.pid}).")
                proc.terminate()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()

# ═══════════════════════════════════════════════════════════════════
# 7. WATCHDOG FILESYSTEM MONITOR
# ═══════════════════════════════════════════════════════════════════

class IntakeManifestHandler(FileSystemEventHandler):
    """
    Monitors SPOVNOB_intake/ for newly created session_profile.json files.
    When detected, enqueues the session into the async processing queue.
    """

    def __init__(self, session_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.session_queue = session_queue
        self.loop = loop

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.name == "session_profile.json":
            logger.info(f"📥 Detected new manifest: {path}")
            asyncio.run_coroutine_threadsafe(
                self.session_queue.put(path), self.loop
            )

# ═══════════════════════════════════════════════════════════════════
# 8. UNIX SIGNAL TRAPPING
# ═══════════════════════════════════════════════════════════════════

class GracefulShutdownManager:
    """
    Traps SIGINT and SIGTERM. On signal:
    1. Flags all active sessions as INTERRUPTED.
    2. Terminates running subprocesses.
    3. Atomically flushes the ledger to disk.
    4. Exits cleanly.
    """

    def __init__(self, ledger: BatchLedger, orchestrator: BatchOrchestrator):
        self.ledger = ledger
        self.orchestrator = orchestrator
        self._triggered = False

    def install(self):
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if self._triggered:
            return
        self._triggered = True

        sig_name = signal.Signals(signum).name
        logger.warning(f"🛑 Received {sig_name}. Initiating graceful shutdown sequence...")

        self.orchestrator.request_shutdown()
        self.orchestrator.terminate_active()
        self.ledger.mark_interrupted()

        logger.info("📋 Ledger flushed. All handles released. Exiting.")
        sys.exit(0)

# ═══════════════════════════════════════════════════════════════════
# 9. MAIN EVENT LOOP
# ═══════════════════════════════════════════════════════════════════

async def main_loop():
    ledger = BatchLedger(LEDGER_PATH)
    session_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    orchestrator = BatchOrchestrator(ledger)
    shutdown_manager = GracefulShutdownManager(ledger, orchestrator)
    shutdown_manager.install()

    # Idempotent Boot: scan for previously QUEUED or INTERRUPTED sessions
    for session_dir in INTAKE_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / "session_profile.json"
        if not manifest_path.exists():
            continue

        sid_candidate = session_dir.name
        if ledger.is_terminal(sid_candidate):
            logger.info(f"[{sid_candidate}] Skipping (already {ledger.get_state(sid_candidate)}).")
            continue

        # Re-enqueue for processing
        await session_queue.put(manifest_path)
        logger.info(f"[{sid_candidate}] Re-enqueued from previous boot cycle.")

    # Start watchdog observer
    event_handler = IntakeManifestHandler(session_queue, loop)
    observer = Observer()
    observer.schedule(event_handler, str(INTAKE_DIR), recursive=True)
    observer.start()
    logger.info(f"👁️  Watchdog monitoring: {INTAKE_DIR}")

    # Consumer loop
    active_tasks = set()

    try:
        while not orchestrator._shutdown_event.is_set():
            try:
                manifest_path = await asyncio.wait_for(
                    session_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                # Clean up completed tasks
                done = {t for t in active_tasks if t.done()}
                for t in done:
                    exc = t.exception() if not t.cancelled() else None
                    if exc:
                        logger.error(f"Task exception: {exc}")
                active_tasks -= done
                continue

            # Parse the manifest
            try:
                with open(manifest_path, "r") as f:
                    profile = json.load(f)
            except Exception as e:
                logger.error(f"Failed to parse {manifest_path}: {e}")
                continue

            session_id = profile.get("session_id", manifest_path.parent.name)
            intake_session_dir = manifest_path.parent

            # Idempotency: skip if already completed or failed
            if ledger.is_terminal(session_id):
                logger.info(f"[{session_id}] Already terminal ({ledger.get_state(session_id)}). Skipping.")
                continue

            ledger.set_state(session_id, "QUEUED", {
                "intake_path": str(intake_session_dir),
                "profile": profile,
            })

            # Spawn async task (throttled by GPU semaphore internally)
            task = asyncio.create_task(
                orchestrator.process_session(session_id, profile, intake_session_dir)
            )
            active_tasks.add(task)

            # Enforce CPU worker ceiling
            if len(active_tasks) >= MAX_CPU_WORKERS:
                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    exc = t.exception() if not t.cancelled() else None
                    if exc:
                        logger.error(f"Task exception: {exc}")

    except asyncio.CancelledError:
        pass
    finally:
        observer.stop()
        observer.join()
        # Wait for remaining tasks
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        logger.info("🏁 Batch daemon shutdown complete.")


if __name__ == "__main__":
    print("═" * 60)
    print("  SPOVNOB BATCH QUEUE DAEMON — PRODUCTION ENGINE")
    print(f"  Intake Directory : {INTAKE_DIR}")
    print(f"  Output Directory : {OUTPUT_DIR}")
    print(f"  GPU Concurrency  : {GPU_SEMAPHORE_LIMIT}")
    print(f"  CPU Concurrency  : {MAX_CPU_WORKERS}")
    print("═" * 60)
    asyncio.run(main_loop())
