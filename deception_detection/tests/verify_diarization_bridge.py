"""
Audio-Diarization Bridge Verification — Merge Phase 1
=====================================================
Validates the diarization adapter that wires the audio-diarization pipeline's
pipeline_output.json into the deception detection seam (replacement for
mock_pyannote_segments at main_pipeline.py:384).

Checks:
  1. Schema guard — a wrong/absent schema is rejected.
  2. Parse — file indices + source mapping recovered from the summary.
  3. segments_for(local) — correct ("TARGET", start, end) tuples, sorted, grouped.
  4. segments_for(global) — uses the batch-timeline boundaries.
  5. offset_ms — subtracted from every boundary, start clamped at 0 (§7 hook).
  6. Empty file — a file with no clean segments yields [].
  7. Clip mapping — original + _canonical/_hubert names map to the same index;
     an unknown clip raises.
  8. WAV resolution + file_offset_ms — relative wav_path resolves under the
     inferred diarization root; per-file global offset recovered.
  9. Validation — invalid clock raises; parsing is deterministic.
 10. Real data (optional) — if session/batch01/pipeline_output.json exists, assert
     its known facts (8 segments, {0:2,1:3,2:1,3:2}, 233000 ms clean).

This test uses ONLY the standard library + a synthetic fixture — no numpy, no
torch, no GPU, no real audio.

Run:
    python tests/verify_diarization_bridge.py
"""

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Ensure project root is on path for imports.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio_isolation.core.diarization_bridge import (  # noqa: E402
    DIARIZATION_OUTPUT_SCHEMA,
    TARGET_SPEAKER_ID,
    CleanSegment,
    DiarizationBridge,
    DiarizationBridgeError,
    _normalize_stem,
)


def _seg(file_index, s_local, e_local, *, offset=0, n=0):
    """Build one clean_segments[] dict with global = local + offset."""
    return {
        "file_index": file_index,
        "start_local_ms": s_local,
        "end_local_ms": e_local,
        "start_global_ms": s_local + offset,
        "end_global_ms": e_local + offset,
        "duration_ms": e_local - s_local,
        "bridged_gap_count": 0,
        "wav_path": f"session/synth/layer3/clean/clean_{file_index:03d}_{s_local}_{e_local}.wav",
        "wav_sha256": f"{n:064d}",
    }


def _synthetic_summary():
    """A 3-file batch: file0 has 2 segments (out of order), file1 has none
    (empty-target case), file2 has 1 segment with a global offset of 133328."""
    files = [
        {"file_index": 0, "source_file": "/diar/in/NT-clip27.mp4",
         "source_sha256": "0" * 64, "duration_ms": 83328,
         "file_offset_ms": 0, "silero_speech_ms": 59416},
        {"file_index": 1, "source_file": "/diar/in/NT-clip28.mov",
         "source_sha256": "1" * 64, "duration_ms": 50000,
         "file_offset_ms": 83328, "silero_speech_ms": 0},
        {"file_index": 2, "source_file": "/diar/in/NT-clip29.mp4",
         "source_sha256": "2" * 64, "duration_ms": 60000,
         "file_offset_ms": 133328, "silero_speech_ms": 30000},
    ]
    clean = [
        _seg(0, 38000, 81000, n=2),   # deliberately listed before the earlier one
        _seg(0, 13000, 32000, n=1),
        _seg(2, 5000, 9000, offset=133328, n=3),
    ]
    return {
        "schema": DIARIZATION_OUTPUT_SCHEMA,
        "manifest_path": "session/synth.manifest.jsonl",
        "files": files,
        "enrollment": {"final_quality_state": "STRONG"},
        "layer2": {"theta_high": 0.58},
        "layer3": {"total_clean_ms": 66000, "total_contaminated_ms": 0,
                   "segment_count": 3, "nan_block_count": 0},
        "clean_segments": clean,
    }


def _write_fixture(root: Path):
    """Write a synthetic pipeline_output.json + touch its referenced wav files."""
    summary = _synthetic_summary()
    work = root / "session" / "synth"
    (work / "layer3" / "clean").mkdir(parents=True, exist_ok=True)
    for seg in summary["clean_segments"]:
        (root / seg["wav_path"]).touch()
    json_path = work / "pipeline_output.json"
    json_path.write_text(json.dumps(summary), encoding="utf-8")
    return json_path, summary


def main() -> int:
    # --- 1. Schema guard -------------------------------------------------
    try:
        DiarizationBridge({"schema": "something-else", "clean_segments": []})
    except DiarizationBridgeError:
        pass
    else:
        raise AssertionError("expected DiarizationBridgeError on bad schema")
    print("✅ 1. schema guard rejects unexpected schema")

    # --- stem normalization (helper) ------------------------------------
    assert _normalize_stem("NT-clip27.mp4") == "nt-clip27"
    assert _normalize_stem("NT-clip27_canonical.mp4") == "nt-clip27"
    assert _normalize_stem("/a/b/NT-clip27_hubert.wav") == "nt-clip27"
    assert _normalize_stem("NT-clip27_canonical_hubert.wav") == "nt-clip27"
    print("✅    stem normalization strips _canonical/_hubert suffixes")

    with tempfile.TemporaryDirectory() as tmp:
        # resolve(): on macOS the tempdir lives under /var, a symlink to
        # /private/var; _infer_root resolves paths, so the fixture root must
        # be resolved too or the equality checks below fail on Darwin.
        root = Path(tmp).resolve()
        json_path, summary = _write_fixture(root)

        # --- 2. Parse ----------------------------------------------------
        bridge = DiarizationBridge.from_output_json(json_path)
        assert bridge.schema == DIARIZATION_OUTPUT_SCHEMA
        assert bridge.file_indices() == [0, 1, 2]
        assert bridge.source_file(0).endswith("NT-clip27.mp4")
        assert bridge.diarization_root == root, (bridge.diarization_root, root)
        print("✅ 2. parse recovers file indices, sources, and infers root")

        # --- 3. segments_for(local) — sorted, labeled, grouped -----------
        s0 = bridge.segments_for(0)
        assert s0 == [
            (TARGET_SPEAKER_ID, 13000, 32000),
            (TARGET_SPEAKER_ID, 38000, 81000),
        ], s0
        assert all(seg[0] == TARGET_SPEAKER_ID for seg in s0)
        assert all(isinstance(seg[1], int) and isinstance(seg[2], int) for seg in s0)
        print("✅ 3. segments_for(local) → sorted ('TARGET', start, end) tuples")

        # --- 4. segments_for(global) -------------------------------------
        s2_local = bridge.segments_for(2, clock="local")
        s2_global = bridge.segments_for(2, clock="global")
        assert s2_local == [(TARGET_SPEAKER_ID, 5000, 9000)], s2_local
        assert s2_global == [(TARGET_SPEAKER_ID, 138328, 142328)], s2_global
        print("✅ 4. segments_for(global) uses batch-timeline boundaries")

        # --- 5. offset_ms — subtract + clamp -----------------------------
        off = bridge.segments_for(0, offset_ms=20000)
        assert off == [
            (TARGET_SPEAKER_ID, 0, 12000),     # 13000-20000 clamped to 0; 32000-20000
            (TARGET_SPEAKER_ID, 18000, 61000),
        ], off
        print("✅ 5. offset_ms subtracts from boundaries and clamps start at 0")

        # --- 6. Empty-target file ----------------------------------------
        assert bridge.segments_for(1) == []
        assert bridge.clean_segments(1) == []
        print("✅ 6. file with no clean segments yields []")

        # --- 7. Clip mapping ---------------------------------------------
        assert bridge.index_for_clip("NT-clip27.mp4") == 0
        assert bridge.index_for_clip("/x/y/NT-clip27_canonical.mp4") == 0
        assert bridge.index_for_clip("NT-clip27_hubert.wav") == 0
        assert bridge.index_for_clip("NT-clip29_canonical.mp4") == 2
        assert bridge.segments_for_clip("NT-clip29_hubert.wav") == s2_local
        try:
            bridge.index_for_clip("UNKNOWN-clip.mp4")
        except DiarizationBridgeError:
            pass
        else:
            raise AssertionError("expected DiarizationBridgeError for unknown clip")
        print("✅ 7. clip mapping tolerates _canonical/_hubert; unknown clip raises")

        # --- 8. WAV resolution + global offset ---------------------------
        wavs = bridge.wav_paths_for(0)
        assert len(wavs) == 2
        assert all(p.exists() for p in wavs), wavs
        assert bridge.file_offset_ms(0) == 0
        assert bridge.file_offset_ms(2) == 133328
        print("✅ 8. wav_path resolves to real files; file_offset_ms recovered")

        # --- 9. Validation + determinism ---------------------------------
        try:
            bridge.segments_for(0, clock="frame")
        except DiarizationBridgeError:
            pass
        else:
            raise AssertionError("expected DiarizationBridgeError for bad clock")
        bridge_b = DiarizationBridge.from_output_json(json_path)
        assert bridge.segments_for(0) == bridge_b.segments_for(0)
        assert bridge.total_clean_ms() == 66000
        assert bridge.segment_count() == 3
        assert bridge.enrollment_quality() == "STRONG"
        print("✅ 9. invalid clock raises; parsing is deterministic; roll-ups OK")

    # --- 10. Real data (optional) ----------------------------------------
    real = Path(PROJECT_ROOT).parent / "session" / "batch01" / "pipeline_output.json"
    if real.exists():
        rb = DiarizationBridge.from_output_json(real)
        dist = Counter()
        total = 0
        for fi in rb.file_indices():
            segs = rb.segments_for(fi)
            dist[fi] = len(segs)
            for _, s, e in segs:
                total += e - s
        assert rb.segment_count() == 8, rb.segment_count()
        assert dict(dist) == {0: 2, 1: 3, 2: 1, 3: 2}, dict(dist)
        assert rb.total_clean_ms() == 233000, rb.total_clean_ms()
        # the adapted local-segment span sums to the reported clean total
        assert total == rb.total_clean_ms(), (total, rb.total_clean_ms())
        print(f"✅ 10. real batch01: 8 segments {dict(dist)}, {total} ms clean")
    else:
        print("➖ 10. real batch01 output not present — skipped (synthetic OK)")

    print("\ndiarization_bridge verification OK — adapter ready for Phase 2 wiring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
