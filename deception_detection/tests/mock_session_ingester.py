#!/usr/bin/env python3
"""
SPOVNOB Mock Session Ingester — End-to-End Sandbox Validation Harness
=====================================================================
Defense-Grade Synthetic Payload Factory.

Generates a complete synthetic interview session intake package inside
SPOVNOB_intake/SESSION_TEST_MOCK/ to validate the batch_daemon watchdog,
fcntl lock verification, ledger state machine, ELAN annotation injection,
and SSE progress broadcast chain.

This script has ZERO pipeline dependencies — it uses only stdlib + numpy.
It constructs valid binary containers (MP4/WAV) from raw bytes without
requiring ffmpeg, opencv, or any multimedia library.

Target Hardware: Ubuntu 44-Core / 512GB ECC / RTX 6000 Ada (48GB VRAM)

Usage:
    python tests/mock_session_ingester.py              # Generate payload
    python tests/mock_session_ingester.py --verify      # Generate + verify daemon pickup
    python tests/mock_session_ingester.py --clean       # Remove test artifacts
"""

import os
import sys
import json
import struct
import time
import tempfile
import argparse
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════
# 0. PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTAKE_DIR = PROJECT_ROOT / "SPOVNOB_intake"
SESSION_ID = "SESSION_TEST_MOCK"
SESSION_DIR = INTAKE_DIR / SESSION_ID
OUTPUT_DIR = PROJECT_ROOT / "pipeline_system_outputs"
LEDGER_PATH = OUTPUT_DIR / "batch_ledger.json"

# Session timing constants (milliseconds)
SESSION_DURATION_MS = 180_000  # 3 minutes
SAMPLE_RATE = 16000            # 16kHz audio
AUDIO_CHANNELS = 1             # Mono
AUDIO_BIT_DEPTH = 16           # 16-bit PCM
VIDEO_FPS = 30                 # 30 frames per second
VIDEO_TIMESCALE = 90000        # Standard MP4 timescale

# ═══════════════════════════════════════════════════════════════════
# 1. ISO BASE MEDIA FILE FORMAT (MP4) CONTAINER BUILDER
# ═══════════════════════════════════════════════════════════════════

def _pack_box(box_type: bytes, payload: bytes) -> bytes:
    """
    Constructs a single ISO BMFF box (atom).
    Box layout: [4 bytes size][4 bytes type][N bytes payload]
    Size includes the 8-byte header itself.
    """
    size = 8 + len(payload)
    return struct.pack(">I", size) + box_type + payload


def _build_ftyp_box() -> bytes:
    """
    File Type Box — declares this as an ISO Base Media File (isom).
    Required as the very first box in any valid MP4 container.
    """
    major_brand = b"isom"
    minor_version = struct.pack(">I", 512)
    compatible_brands = b"isomiso2mp41"
    return _pack_box(b"ftyp", major_brand + minor_version + compatible_brands)


def _build_mvhd_box(duration_ms: int) -> bytes:
    """
    Movie Header Box — contains global timing metadata.
    Encodes the total duration so ffprobe/mediainfo correctly
    reports the container length without scanning mdat.

    Uses version 0 (32-bit fields) for maximum compatibility.
    """
    timescale = VIDEO_TIMESCALE
    duration_ticks = int((duration_ms / 1000.0) * timescale)

    # Version 0 mvhd: 108 bytes total payload
    creation_time = 0
    modification_time = 0
    next_track_id = 2  # One video track

    payload = struct.pack(">I", 0)                # version + flags
    payload += struct.pack(">I", creation_time)
    payload += struct.pack(">I", modification_time)
    payload += struct.pack(">I", timescale)
    payload += struct.pack(">I", duration_ticks)
    payload += struct.pack(">I", 0x00010000)       # preferred rate (1.0 fixed-point)
    payload += struct.pack(">H", 0x0100)           # preferred volume (1.0 fixed-point)
    payload += b"\x00" * 10                        # reserved
    # Unity matrix (3x3 fixed-point, 36 bytes)
    payload += struct.pack(">9I",
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000
    )
    payload += b"\x00" * 24                        # pre-defined (6 x uint32)
    payload += struct.pack(">I", next_track_id)

    return _pack_box(b"mvhd", payload)


def _build_tkhd_box(duration_ms: int, width: int = 640, height: int = 480) -> bytes:
    """
    Track Header Box — defines the video track geometry and duration.
    """
    timescale = VIDEO_TIMESCALE
    duration_ticks = int((duration_ms / 1000.0) * timescale)

    payload = struct.pack(">I", 0x00000003)        # version=0, flags=track_enabled|in_movie
    payload += struct.pack(">I", 0)                # creation_time
    payload += struct.pack(">I", 0)                # modification_time
    payload += struct.pack(">I", 1)                # track_id
    payload += struct.pack(">I", 0)                # reserved
    payload += struct.pack(">I", duration_ticks)
    payload += b"\x00" * 8                         # reserved
    payload += struct.pack(">H", 0)                # layer
    payload += struct.pack(">H", 0)                # alternate_group
    payload += struct.pack(">H", 0)                # volume (0 for video)
    payload += struct.pack(">H", 0)                # reserved
    # Unity matrix
    payload += struct.pack(">9I",
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000
    )
    # Width and height in 16.16 fixed-point
    payload += struct.pack(">I", width << 16)
    payload += struct.pack(">I", height << 16)

    return _pack_box(b"tkhd", payload)


def _build_mdhd_box(duration_ms: int) -> bytes:
    """
    Media Header Box — track-level timing (uses same timescale as mvhd).
    """
    timescale = VIDEO_TIMESCALE
    duration_ticks = int((duration_ms / 1000.0) * timescale)

    payload = struct.pack(">I", 0)                 # version + flags
    payload += struct.pack(">I", 0)                # creation_time
    payload += struct.pack(">I", 0)                # modification_time
    payload += struct.pack(">I", timescale)
    payload += struct.pack(">I", duration_ticks)
    payload += struct.pack(">H", 0x55C4)           # language: 'und' (undetermined)
    payload += struct.pack(">H", 0)                # pre-defined

    return _pack_box(b"mdhd", payload)


def _build_hdlr_box() -> bytes:
    """
    Handler Reference Box — declares this track as a video handler.
    """
    payload = struct.pack(">I", 0)                 # version + flags
    payload += struct.pack(">I", 0)                # pre-defined
    payload += b"vide"                             # handler_type
    payload += b"\x00" * 12                        # reserved
    payload += b"VideoHandler\x00"                 # name (null-terminated)

    return _pack_box(b"hdlr", payload)


def _build_stbl_box() -> bytes:
    """
    Sample Table Box — contains minimal empty sample description tables.
    Required for structural validity even though we have no real samples.
    """
    # stsd: Sample Description Box (empty, 1 entry count but no actual entries for brevity)
    stsd_payload = struct.pack(">I", 0)            # version + flags
    stsd_payload += struct.pack(">I", 0)           # entry_count = 0
    stsd = _pack_box(b"stsd", stsd_payload)

    # stts: Decoding Time to Sample Box (empty)
    stts_payload = struct.pack(">I", 0) + struct.pack(">I", 0)
    stts = _pack_box(b"stts", stts_payload)

    # stsc: Sample to Chunk Box (empty)
    stsc_payload = struct.pack(">I", 0) + struct.pack(">I", 0)
    stsc = _pack_box(b"stsc", stsc_payload)

    # stsz: Sample Size Box (empty)
    stsz_payload = struct.pack(">I", 0)            # version + flags
    stsz_payload += struct.pack(">I", 0)           # sample_size (0 = variable)
    stsz_payload += struct.pack(">I", 0)           # sample_count
    stsz = _pack_box(b"stsz", stsz_payload)

    # stco: Chunk Offset Box (empty)
    stco_payload = struct.pack(">I", 0) + struct.pack(">I", 0)
    stco = _pack_box(b"stco", stco_payload)

    return _pack_box(b"stbl", stsd + stts + stsc + stsz + stco)


def _build_minf_box() -> bytes:
    """
    Media Information Box — wraps the video media handler hint and sample table.
    """
    # vmhd: Video Media Header Box
    vmhd_payload = struct.pack(">I", 0x00000001)   # version=0, flags=1
    vmhd_payload += struct.pack(">H", 0)           # graphicsmode
    vmhd_payload += struct.pack(">3H", 0, 0, 0)   # opcolor
    vmhd = _pack_box(b"vmhd", vmhd_payload)

    # dinf + dref: Data Information + Data Reference (self-contained)
    dref_payload = struct.pack(">I", 0)            # version + flags
    dref_payload += struct.pack(">I", 1)           # entry_count
    # url entry (self-contained flag)
    url_payload = struct.pack(">I", 0x00000001)    # version=0, flags=self_contained
    url_entry = _pack_box(b"url ", url_payload)
    dref_payload += url_entry
    dref = _pack_box(b"dref", dref_payload)
    dinf = _pack_box(b"dinf", dref)

    stbl = _build_stbl_box()

    return _pack_box(b"minf", vmhd + dinf + stbl)


def _build_mdia_box(duration_ms: int) -> bytes:
    """Media Box — wraps mdhd + hdlr + minf."""
    mdhd = _build_mdhd_box(duration_ms)
    hdlr = _build_hdlr_box()
    minf = _build_minf_box()
    return _pack_box(b"mdia", mdhd + hdlr + minf)


def _build_trak_box(duration_ms: int) -> bytes:
    """Track Box — wraps tkhd + mdia."""
    tkhd = _build_tkhd_box(duration_ms)
    mdia = _build_mdia_box(duration_ms)
    return _pack_box(b"trak", tkhd + mdia)


def _build_moov_box(duration_ms: int) -> bytes:
    """Movie Box — wraps mvhd + trak."""
    mvhd = _build_mvhd_box(duration_ms)
    trak = _build_trak_box(duration_ms)
    return _pack_box(b"moov", mvhd + trak)


def build_mock_mp4(output_path: Path, duration_ms: int, target_size_bytes: int = 2_500_000) -> dict:
    """
    Constructs a structurally valid MP4 container from raw bytes.
    The mdat box is filled with pseudorandom data to hit the target file size.

    Returns metadata dict with file size and SHA-256 hash.
    """
    ftyp = _build_ftyp_box()
    moov = _build_moov_box(duration_ms)

    # Calculate mdat payload size to reach target file size
    header_size = len(ftyp) + len(moov) + 8  # +8 for mdat box header
    mdat_payload_size = max(0, target_size_bytes - header_size)

    # Generate pseudorandom mdat payload (simulates encoded video data)
    # Using os.urandom for high-throughput entropy without numpy dependency
    mdat_data = os.urandom(mdat_payload_size)
    mdat = _pack_box(b"mdat", mdat_data)

    # Write the complete container
    with open(output_path, "wb") as f:
        f.write(ftyp)
        f.write(moov)
        f.write(mdat)

    file_size = output_path.stat().st_size
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    return {
        "path": str(output_path),
        "size_bytes": file_size,
        "sha256": sha256,
        "duration_ms": duration_ms,
        "container": "ISO BMFF (isom/mp41)",
    }


# ═══════════════════════════════════════════════════════════════════
# 2. RIFF/WAVE PCM AUDIO FILE BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_mock_wav(output_path: Path, duration_ms: int) -> dict:
    """
    Constructs a valid PCM WAV file (16kHz mono 16-bit) containing
    low-amplitude Gaussian white noise.

    Returns metadata dict with file size, sample count, and SHA-256 hash.
    """
    duration_sec = duration_ms / 1000.0
    num_samples = int(SAMPLE_RATE * duration_sec)
    bytes_per_sample = AUDIO_BIT_DEPTH // 8
    data_chunk_size = num_samples * AUDIO_CHANNELS * bytes_per_sample

    # RIFF header
    riff_size = 36 + data_chunk_size  # 36 = fmt chunk overhead + data header

    # Generate low-amplitude Gaussian noise (±1000 range in int16)
    # Bulk generation: build entire sample array then single struct.pack
    # This avoids 5.76M individual pack_into calls (60s → ~3s)
    rng = random.Random(0xCAFEBABE)
    samples_list = [
        max(-32000, min(32000, int(rng.gauss(0, 300))))
        for _ in range(num_samples)
    ]
    samples_raw = struct.pack(f"<{num_samples}h", *samples_list)

    with open(output_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")

        # fmt sub-chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))             # Sub-chunk size (PCM = 16)
        f.write(struct.pack("<H", 1))              # Audio format (1 = PCM)
        f.write(struct.pack("<H", AUDIO_CHANNELS))
        f.write(struct.pack("<I", SAMPLE_RATE))
        byte_rate = SAMPLE_RATE * AUDIO_CHANNELS * bytes_per_sample
        f.write(struct.pack("<I", byte_rate))
        block_align = AUDIO_CHANNELS * bytes_per_sample
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", AUDIO_BIT_DEPTH))

        # data sub-chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_chunk_size))
        f.write(bytes(samples_raw))

    file_size = output_path.stat().st_size
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    return {
        "path": str(output_path),
        "size_bytes": file_size,
        "sha256": sha256,
        "duration_ms": duration_ms,
        "sample_rate": SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "bit_depth": AUDIO_BIT_DEPTH,
        "num_samples": num_samples,
        "format": "RIFF/WAVE PCM",
    }


# ═══════════════════════════════════════════════════════════════════
# 3. SESSION MANIFEST GENERATOR (14 Timeline Phases)
# ═══════════════════════════════════════════════════════════════════

def build_session_manifest(output_path: Path) -> dict:
    """
    Generates a 14-phase interview timeline manifest matching the
    ContextMapper schema: [{start_ms, end_ms, phase_label, question_id}, ...]

    Phase structure (3-minute session):
      Phase 0:  Baseline neutral (0-30s)
      Phase 1:  Briefing/Instruction (30-38s)
      Phase 2:  Question 1 (38-48s)
      Phase 3:  Response 1 (48-62s)
      Phase 4:  Question 2 (62-70s)
      Phase 5:  Response 2 (70-88s)
      Phase 6:  Question 3 (88-96s)
      Phase 7:  Response 3 (96-112s)
      Phase 8:  Question 4 (112-120s)
      Phase 9:  Response 4 (120-135s)
      Phase 10: Question 5 (135-142s)
      Phase 11: Response 5 (142-157s)
      Phase 12: Question 6 (157-164s)
      Phase 13: Response 6 / Debrief (164-180s)
    """
    phases = [
        {"start_ms": 0,      "end_ms": 30000,  "phase_label": "baseline_neutral",    "question_id": 0},
        {"start_ms": 30000,  "end_ms": 38000,  "phase_label": "briefing_instruction", "question_id": 0},
        {"start_ms": 38000,  "end_ms": 48000,  "phase_label": "question_delivery",   "question_id": 1},
        {"start_ms": 48000,  "end_ms": 62000,  "phase_label": "subject_response",    "question_id": 1},
        {"start_ms": 62000,  "end_ms": 70000,  "phase_label": "question_delivery",   "question_id": 2},
        {"start_ms": 70000,  "end_ms": 88000,  "phase_label": "subject_response",    "question_id": 2},
        {"start_ms": 88000,  "end_ms": 96000,  "phase_label": "question_delivery",   "question_id": 3},
        {"start_ms": 96000,  "end_ms": 112000, "phase_label": "subject_response",    "question_id": 3},
        {"start_ms": 112000, "end_ms": 120000, "phase_label": "question_delivery",   "question_id": 4},
        {"start_ms": 120000, "end_ms": 135000, "phase_label": "subject_response",    "question_id": 4},
        {"start_ms": 135000, "end_ms": 142000, "phase_label": "question_delivery",   "question_id": 5},
        {"start_ms": 142000, "end_ms": 157000, "phase_label": "subject_response",    "question_id": 5},
        {"start_ms": 157000, "end_ms": 164000, "phase_label": "question_delivery",   "question_id": 6},
        {"start_ms": 164000, "end_ms": 180000, "phase_label": "subject_response",    "question_id": 6},
    ]

    with open(output_path, "w") as f:
        json.dump(phases, f, indent=2)

    return {
        "path": str(output_path),
        "phase_count": len(phases),
        "total_duration_ms": phases[-1]["end_ms"],
        "phases": [p["phase_label"] for p in phases],
    }


# ═══════════════════════════════════════════════════════════════════
# 4. ELAN ANNOTATION EXPORT CSV GENERATOR
# ═══════════════════════════════════════════════════════════════════

def build_elan_export(output_path: Path) -> dict:
    """
    Generates a multi-row ELAN annotation CSV with columns:
      start_time_ms, end_time_ms, annotation_value

    These map to the ELANAnnotationMapper's positional fallback parser
    (columns[0]=start, columns[1]=end, columns[2]=label) as defined in
    batch_daemon.py lines 199-202.

    Annotation layout (8 intervals across the 3-minute session):
      truth  →  48000-62000   (Q1 response)
      lying  →  70000-82000   (Q2 response first half)
      truth  →  82000-88000   (Q2 response second half)
      lying  →  96000-108000  (Q3 response first half)
      truth  → 108000-112000  (Q3 response tail)
      lying  → 120000-130000  (Q4 response first half)
      truth  → 130000-135000  (Q4 response tail)
      lying  → 142000-157000  (Q5 full response)
    """
    annotations = [
        {"start_time_ms": 48000,  "end_time_ms": 62000,  "annotation_value": "truth"},
        {"start_time_ms": 70000,  "end_time_ms": 82000,  "annotation_value": "lying"},
        {"start_time_ms": 82000,  "end_time_ms": 88000,  "annotation_value": "truth"},
        {"start_time_ms": 96000,  "end_time_ms": 108000, "annotation_value": "lying"},
        {"start_time_ms": 108000, "end_time_ms": 112000, "annotation_value": "truth"},
        {"start_time_ms": 120000, "end_time_ms": 130000, "annotation_value": "lying"},
        {"start_time_ms": 130000, "end_time_ms": 135000, "annotation_value": "truth"},
        {"start_time_ms": 142000, "end_time_ms": 157000, "annotation_value": "lying"},
    ]

    # Write CSV manually to avoid pandas dependency
    with open(output_path, "w") as f:
        f.write("start_time_ms,end_time_ms,annotation_value\n")
        for ann in annotations:
            f.write(f"{ann['start_time_ms']},{ann['end_time_ms']},{ann['annotation_value']}\n")

    return {
        "path": str(output_path),
        "annotation_count": len(annotations),
        "truth_intervals": sum(1 for a in annotations if a["annotation_value"] == "truth"),
        "lying_intervals": sum(1 for a in annotations if a["annotation_value"] == "lying"),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. SESSION PROFILE TRIGGER (Atomic Write — Written LAST)
# ═══════════════════════════════════════════════════════════════════

def build_session_profile(output_path: Path) -> dict:
    """
    Constructs the watchdog trigger file: session_profile.json

    This is the file that batch_daemon.py's IntakeManifestHandler monitors.
    When created, the watchdog enqueues the session for processing.

    CRITICAL: This file is written LAST via atomic os.replace() to guarantee
    that all other session assets (video, audio, manifest, ELAN) are fully
    flushed to disk before the watchdog fires.

    Schema matches batch_daemon.py lines 372-374 and 583:
      - session_id: identifies the session in the ledger
      - session_manifest_path: relative path to the context manifest
      - elan_annotation_file: relative path to the ELAN CSV
    """
    profile = {
        "session_id": SESSION_ID,
        "session_manifest_path": "session_manifest.json",
        "elan_annotation_file": "elan_export.csv",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mock_payload": True,
        "expected_duration_ms": SESSION_DURATION_MS,
        "notes": "Synthetic validation payload generated by mock_session_ingester.py",
    }

    # Atomic write: temp file → os.replace()
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(output_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(profile, f, indent=2)
        os.replace(tmp_path, str(output_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return {
        "path": str(output_path),
        "session_id": SESSION_ID,
        "trigger_time": profile["created_at"],
    }


# ═══════════════════════════════════════════════════════════════════
# 6. SELF-VALIDATION CHECKS
# ═══════════════════════════════════════════════════════════════════

def validate_payload(session_dir: Path) -> bool:
    """
    Post-generation integrity checks to verify the payload is structurally
    sound before the watchdog picks it up.
    """
    print("\n" + "═" * 60)
    print("  PAYLOAD SELF-VALIDATION")
    print("═" * 60)

    checks_passed = 0
    checks_total = 0

    def _check(description: str, condition: bool):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ FAIL: {description}")

    # 1. File existence
    expected_files = [
        "interview_video.mp4",
        "interview_audio.wav",
        "session_manifest.json",
        "elan_export.csv",
        "session_profile.json",
    ]
    for fname in expected_files:
        fpath = session_dir / fname
        _check(f"{fname} exists", fpath.exists())

    # 2. MP4 structural integrity
    mp4_path = session_dir / "interview_video.mp4"
    if mp4_path.exists():
        with open(mp4_path, "rb") as f:
            header = f.read(12)
        # Check ftyp box signature at offset 4
        _check("MP4 starts with ftyp box", header[4:8] == b"ftyp")
        _check("MP4 major brand is isom", header[8:12] == b"isom")
        _check(f"MP4 file size > 2MB ({mp4_path.stat().st_size:,} bytes)",
               mp4_path.stat().st_size > 2_000_000)

    # 3. WAV structural integrity
    wav_path = session_dir / "interview_audio.wav"
    if wav_path.exists():
        with open(wav_path, "rb") as f:
            header = f.read(44)
        _check("WAV starts with RIFF header", header[0:4] == b"RIFF")
        _check("WAV contains WAVE format", header[8:12] == b"WAVE")
        _check("WAV contains fmt chunk", header[12:16] == b"fmt ")
        # Parse sample rate from fmt chunk (offset 24, little-endian uint32)
        parsed_sr = struct.unpack("<I", header[24:28])[0]
        _check(f"WAV sample rate is 16kHz (parsed: {parsed_sr})", parsed_sr == 16000)
        _check(f"WAV file size > 5MB ({wav_path.stat().st_size:,} bytes)",
               wav_path.stat().st_size > 5_000_000)

    # 4. Session manifest JSON integrity
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        _check("Session manifest is a list", isinstance(manifest, list))
        _check(f"Session manifest has 14 phases ({len(manifest)})", len(manifest) == 14)
        # Verify chronological ordering
        starts = [p["start_ms"] for p in manifest]
        _check("Manifest phases are chronologically ordered",
               all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1)))
        # Verify full coverage
        _check("Manifest starts at 0ms", manifest[0]["start_ms"] == 0)
        _check(f"Manifest ends at {SESSION_DURATION_MS}ms",
               manifest[-1]["end_ms"] == SESSION_DURATION_MS)
        # Verify required fields
        required_keys = {"start_ms", "end_ms", "phase_label", "question_id"}
        _check("All phases have required keys",
               all(required_keys.issubset(p.keys()) for p in manifest))

    # 5. ELAN CSV integrity
    elan_path = session_dir / "elan_export.csv"
    if elan_path.exists():
        with open(elan_path, "r") as f:
            lines = f.readlines()
        header = lines[0].strip()
        _check("ELAN CSV header matches expected schema",
               header == "start_time_ms,end_time_ms,annotation_value")
        data_rows = [l.strip() for l in lines[1:] if l.strip()]
        _check(f"ELAN CSV has 8 data rows ({len(data_rows)})", len(data_rows) == 8)
        # Verify all annotations are truth or lying
        labels = set()
        for row in data_rows:
            parts = row.split(",")
            if len(parts) >= 3:
                labels.add(parts[2])
        _check("ELAN labels are {truth, lying}",
               labels == {"truth", "lying"})

    # 6. Session profile JSON integrity
    profile_path = session_dir / "session_profile.json"
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile = json.load(f)
        _check("Profile contains session_id",
               profile.get("session_id") == SESSION_ID)
        _check("Profile contains session_manifest_path",
               "session_manifest_path" in profile)
        _check("Profile contains elan_annotation_file",
               "elan_annotation_file" in profile)
        # Verify referenced files exist
        manifest_ref = profile.get("session_manifest_path", "")
        _check(f"Referenced manifest exists: {manifest_ref}",
               (session_dir / manifest_ref).exists() if manifest_ref else False)
        elan_ref = profile.get("elan_annotation_file", "")
        _check(f"Referenced ELAN file exists: {elan_ref}",
               (session_dir / elan_ref).exists() if elan_ref else False)

    # Summary
    print(f"\n   {'─' * 40}")
    print(f"   Results: {checks_passed}/{checks_total} passed")
    if checks_passed == checks_total:
        print("   🏆 ALL SELF-VALIDATION CHECKS PASSED")
    else:
        print(f"   ⚠️  {checks_total - checks_passed} FAILURES DETECTED")

    return checks_passed == checks_total


# ═══════════════════════════════════════════════════════════════════
# 7. DAEMON PICKUP VERIFICATION (--verify mode)
# ═══════════════════════════════════════════════════════════════════

def verify_daemon_pickup(timeout_sec: int = 30) -> bool:
    """
    Monitors the batch_ledger.json for SESSION_TEST_MOCK state transitions.
    Expects the daemon to be running in a separate terminal.

    Polling interval: 2 seconds (matches the SSE push interval in server.py).
    """
    print("\n" + "═" * 60)
    print("  DAEMON PICKUP VERIFICATION (Live)")
    print(f"  Monitoring ledger for {timeout_sec}s...")
    print("═" * 60)

    if not LEDGER_PATH.parent.exists():
        print(f"   ❌ Output directory does not exist: {LEDGER_PATH.parent}")
        return False

    start_time = time.monotonic()
    last_state = None
    state_transitions = []

    while time.monotonic() - start_time < timeout_sec:
        if LEDGER_PATH.exists():
            try:
                with open(LEDGER_PATH, "r") as f:
                    ledger = json.load(f)
                entry = ledger.get(SESSION_ID)
                if entry:
                    state = entry.get("state", "UNKNOWN")
                    if state != last_state:
                        elapsed = time.monotonic() - start_time
                        state_transitions.append((elapsed, state))
                        print(f"   📡 [{elapsed:6.1f}s] State transition: {last_state} → {state}")
                        last_state = state

                    # Terminal states: stop monitoring
                    if state in ("COMPLETED", "FAILED"):
                        print(f"\n   ✅ Session reached terminal state: {state}")
                        if "error" in entry:
                            print(f"   📝 Error field: {entry['error'][:200]}")
                        break
            except (json.JSONDecodeError, IOError):
                pass  # Ledger mid-write, retry next cycle

        time.sleep(2.0)

    if not state_transitions:
        print(f"   ❌ No state transitions observed for {SESSION_ID} within {timeout_sec}s.")
        print("   💡 Is batch_daemon.py running? Start it with:")
        print(f"      python -u app/batch_daemon.py")
        return False

    print(f"\n   📊 State transition history ({len(state_transitions)} transitions):")
    for elapsed, state in state_transitions:
        print(f"      [{elapsed:6.1f}s] {state}")

    return True


# ═══════════════════════════════════════════════════════════════════
# 8. SSE STREAM VERIFICATION
# ═══════════════════════════════════════════════════════════════════

def verify_sse_stream(timeout_sec: int = 15) -> bool:
    """
    Connects to the SSE endpoint and verifies that at least one event
    contains SESSION_TEST_MOCK. Uses raw socket/http to avoid external
    dependencies (no httpx/requests needed).

    Requires server.py to be running on port 8000.
    """
    print("\n" + "═" * 60)
    print("  SSE STREAM VERIFICATION (Live)")
    print("═" * 60)

    import http.client

    try:
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=timeout_sec)
        conn.request("GET", "/api/factory/stream", headers={"Accept": "text/event-stream"})
        response = conn.getresponse()

        if response.status != 200:
            print(f"   ❌ SSE endpoint returned HTTP {response.status}")
            return False

        print(f"   ✅ Connected to SSE stream (HTTP {response.status})")
        print(f"   ⏳ Listening for {timeout_sec}s...")

        buffer = b""
        start_time = time.monotonic()
        events_received = 0
        session_seen = False

        while time.monotonic() - start_time < timeout_sec:
            try:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk

                # Parse SSE events (delimited by double newline)
                while b"\n\n" in buffer:
                    event_data, buffer = buffer.split(b"\n\n", 1)
                    event_str = event_data.decode("utf-8", errors="replace")

                    if event_str.startswith("data: "):
                        events_received += 1
                        if SESSION_ID in event_str:
                            session_seen = True
                            print(f"   📡 Event #{events_received}: Contains {SESSION_ID} ✅")
                        else:
                            print(f"   📡 Event #{events_received}: Ledger update (no mock session)")
            except Exception:
                break

        conn.close()

        print(f"\n   📊 Total SSE events received: {events_received}")
        if session_seen:
            print(f"   ✅ {SESSION_ID} observed in SSE stream")
        else:
            print(f"   ⚠️  {SESSION_ID} not observed (daemon may not have processed it yet)")

        return events_received > 0

    except ConnectionRefusedError:
        print("   ❌ Connection refused on port 8000.")
        print("   💡 Is server.py running? Start it with:")
        print(f"      python -u app/server.py")
        return False
    except Exception as e:
        print(f"   ❌ SSE verification error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# 9. FACTORY STATUS API CHECK
# ═══════════════════════════════════════════════════════════════════

def verify_factory_status() -> bool:
    """
    Hits GET /api/factory/status and validates the response contains
    a sessions dict. Uses stdlib http.client.
    """
    print("\n" + "═" * 60)
    print("  FACTORY STATUS API VERIFICATION")
    print("═" * 60)

    import http.client

    try:
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)
        conn.request("GET", "/api/factory/status")
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()

        if response.status != 200:
            print(f"   ❌ Factory status returned HTTP {response.status}")
            return False

        data = json.loads(body)
        print(f"   ✅ Factory status endpoint responded (HTTP 200)")
        print(f"   📊 Summary: {json.dumps(data.get('summary', {}), indent=6)}")

        sessions = data.get("sessions", {})
        if SESSION_ID in sessions:
            state = sessions[SESSION_ID].get("state", "UNKNOWN")
            print(f"   ✅ {SESSION_ID} found in factory status: state={state}")
        else:
            print(f"   ⚠️  {SESSION_ID} not yet in factory status")

        return True

    except ConnectionRefusedError:
        print("   ❌ Connection refused on port 8000. Is server.py running?")
        return False
    except Exception as e:
        print(f"   ❌ Factory status check error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# 10. CLEANUP
# ═══════════════════════════════════════════════════════════════════

def clean_mock_session():
    """Removes all test artifacts generated by this script."""
    import shutil

    targets = [
        SESSION_DIR,
        OUTPUT_DIR / SESSION_ID,
    ]

    print("\n" + "═" * 60)
    print("  CLEANUP — Removing Mock Session Artifacts")
    print("═" * 60)

    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"   🗑️  Removed: {target}")
        else:
            print(f"   ⏭️  Not found (already clean): {target}")

    # Remove session from ledger if present
    if LEDGER_PATH.exists():
        try:
            with open(LEDGER_PATH, "r") as f:
                ledger = json.load(f)
            if SESSION_ID in ledger:
                del ledger[SESSION_ID]
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(LEDGER_PATH.parent), suffix=".tmp"
                )
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(ledger, f, indent=2)
                os.replace(tmp_path, str(LEDGER_PATH))
                print(f"   🗑️  Removed {SESSION_ID} from batch_ledger.json")
            else:
                print(f"   ⏭️  {SESSION_ID} not in ledger (already clean)")
        except Exception as e:
            print(f"   ⚠️  Could not clean ledger: {e}")

    print("   ✅ Cleanup complete.")


# ═══════════════════════════════════════════════════════════════════
# 11. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SPOVNOB Mock Session Ingester — Sandbox Validation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/mock_session_ingester.py              # Generate mock payload only
  python tests/mock_session_ingester.py --verify      # Generate + verify daemon/server pickup
  python tests/mock_session_ingester.py --clean       # Remove all test artifacts
  python tests/mock_session_ingester.py --verify-only # Skip generation, verify existing payload
        """
    )
    parser.add_argument("--verify", action="store_true",
                        help="After generation, monitor daemon pickup and SSE stream")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip payload generation; only run live verification checks")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all mock session artifacts and exit")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Daemon pickup verification timeout in seconds (default: 30)")

    args = parser.parse_args()

    print("═" * 60)
    print("  SPOVNOB MOCK SESSION INGESTER")
    print(f"  Session ID : {SESSION_ID}")
    print(f"  Target Dir : {SESSION_DIR}")
    print(f"  Duration   : {SESSION_DURATION_MS / 1000:.0f} seconds")
    print(f"  Timestamp  : {datetime.now(timezone.utc).isoformat()}")
    print("═" * 60)

    if args.clean:
        clean_mock_session()
        return

    if args.verify_only:
        verify_daemon_pickup(timeout_sec=args.timeout)
        verify_sse_stream(timeout_sec=15)
        verify_factory_status()
        return

    # ── Create session directory ──────────────────────────────────
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Session directory created: {SESSION_DIR}")

    # ── Phase 1: Build synthetic MP4 container ────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 1: Building Mock MP4 Container")
    print("─" * 60)
    mp4_meta = build_mock_mp4(
        SESSION_DIR / "interview_video.mp4",
        duration_ms=SESSION_DURATION_MS,
        target_size_bytes=2_500_000,
    )
    print(f"   ✅ MP4 built: {mp4_meta['size_bytes']:,} bytes")
    print(f"   📦 Container: {mp4_meta['container']}")
    print(f"   ⏱️  Duration:  {mp4_meta['duration_ms'] / 1000:.0f}s")
    print(f"   🔑 SHA-256:   {mp4_meta['sha256'][:32]}...")

    # ── Phase 2: Build synthetic WAV audio ────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 2: Building Mock WAV Audio")
    print("─" * 60)
    wav_meta = build_mock_wav(
        SESSION_DIR / "interview_audio.wav",
        duration_ms=SESSION_DURATION_MS,
    )
    print(f"   ✅ WAV built: {wav_meta['size_bytes']:,} bytes")
    print(f"   📦 Format:    {wav_meta['format']}")
    print(f"   🎵 Samples:   {wav_meta['num_samples']:,} ({wav_meta['sample_rate']}Hz × {wav_meta['duration_ms'] / 1000:.0f}s)")
    print(f"   🔑 SHA-256:   {wav_meta['sha256'][:32]}...")

    # ── Phase 3: Build session manifest ───────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 3: Building Session Manifest (14 Timeline Phases)")
    print("─" * 60)
    manifest_meta = build_session_manifest(
        SESSION_DIR / "session_manifest.json",
    )
    print(f"   ✅ Manifest built: {manifest_meta['phase_count']} phases")
    print(f"   ⏱️  Coverage: 0ms → {manifest_meta['total_duration_ms']}ms")
    for i, phase in enumerate(manifest_meta["phases"]):
        print(f"      Phase {i:2d}: {phase}")

    # ── Phase 4: Build ELAN annotation export ─────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 4: Building ELAN Annotation Export")
    print("─" * 60)
    elan_meta = build_elan_export(
        SESSION_DIR / "elan_export.csv",
    )
    print(f"   ✅ ELAN CSV built: {elan_meta['annotation_count']} intervals")
    print(f"   📊 Truth intervals: {elan_meta['truth_intervals']}")
    print(f"   📊 Lying intervals: {elan_meta['lying_intervals']}")

    # ── Phase 5: Write session profile trigger (LAST) ─────────────
    print("\n" + "─" * 60)
    print("  PHASE 5: Writing Session Profile Trigger (ATOMIC)")
    print("─" * 60)
    profile_meta = build_session_profile(
        SESSION_DIR / "session_profile.json",
    )
    print(f"   ✅ Profile trigger written via os.replace()")
    print(f"   🎯 Session ID: {profile_meta['session_id']}")
    print(f"   🕐 Trigger:    {profile_meta['trigger_time']}")

    # ── Self-Validation ───────────────────────────────────────────
    all_ok = validate_payload(SESSION_DIR)

    # ── Write generation manifest (for audit trail) ───────────────
    generation_log = {
        "session_id": SESSION_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "mock_session_ingester.py",
        "assets": {
            "mp4": mp4_meta,
            "wav": wav_meta,
            "manifest": manifest_meta,
            "elan": elan_meta,
            "profile": profile_meta,
        },
        "validation_passed": all_ok,
    }
    log_path = SESSION_DIR / "_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(generation_log, f, indent=2, default=str)
    print(f"\n📋 Generation log written: {log_path}")

    # ── Summary ───────────────────────────────────────────────────
    total_size = sum(
        (SESSION_DIR / f).stat().st_size
        for f in SESSION_DIR.iterdir()
        if f.is_file()
    )
    print("\n" + "═" * 60)
    print("  PAYLOAD GENERATION COMPLETE")
    print(f"  Total disk footprint: {total_size / (1024 * 1024):.2f} MB")
    print(f"  Files generated:      {sum(1 for _ in SESSION_DIR.iterdir())}")
    print("═" * 60)

    if not all_ok:
        print("\n❌ Self-validation FAILED. Aborting.")
        sys.exit(1)

    # ── Live Verification (optional) ──────────────────────────────
    if args.verify:
        print("\n" + "═" * 60)
        print("  LIVE SYSTEM VERIFICATION MODE")
        print("  Expecting batch_daemon.py and server.py to be running.")
        print("═" * 60)

        verify_daemon_pickup(timeout_sec=args.timeout)
        verify_sse_stream(timeout_sec=15)
        verify_factory_status()

    print("\n🏁 Mock session ingester complete.")


if __name__ == "__main__":
    main()
