"""
Recording Intake Verification — Phase 3 (batch front-door)
==========================================================
Validates clip ↔ wav ↔ file_index pairing for a recording bucket — the helper
the batch daemon uses to turn N clips + a shared pipeline_output.json into the
ordered (mp4, wav) list fed to process_recording_session. No GPU / torch.

Run:
    python tests/verify_recording_intake.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.recording_intake import collect_recording_clips  # noqa: E402
from audio_isolation.core.diarization_bridge import DIARIZATION_OUTPUT_SCHEMA  # noqa: E402


def _summary():
    return {
        "schema": DIARIZATION_OUTPUT_SCHEMA,
        "files": [
            {"file_index": 0, "source_file": "/diar/NT-clip27.mp4", "duration_ms": 80000,
             "file_offset_ms": 0, "silero_speech_ms": 40000, "source_sha256": "0" * 64},
            {"file_index": 1, "source_file": "/diar/NT-clip28.mov", "duration_ms": 60000,
             "file_offset_ms": 80000, "silero_speech_ms": 30000, "source_sha256": "1" * 64},
        ],
        "clean_segments": [
            {"file_index": 0, "start_local_ms": 1000, "end_local_ms": 3000, "start_global_ms": 1000,
             "end_global_ms": 3000, "duration_ms": 2000, "bridged_gap_count": 0,
             "wav_path": "a.wav", "wav_sha256": "0" * 64},
        ],
        "layer3": {"total_clean_ms": 2000, "segment_count": 1},
        "enrollment": {"final_quality_state": "STRONG"},
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        bucket = Path(tmp)
        (bucket / "pipeline_output.json").write_text(json.dumps(_summary()))
        # canonical clips + hubert wavs → suffix-tolerant pairing to source stems
        (bucket / "NT-clip27_canonical.mp4").touch()
        (bucket / "NT-clip27_hubert.wav").touch()
        (bucket / "NT-clip28_canonical.mp4").touch()
        (bucket / "NT-clip28_hubert.wav").touch()

        # --- 1. pairing + ordering ---------------------------------------
        clips = collect_recording_clips(bucket, bucket / "pipeline_output.json")
        assert len(clips) == 2, clips
        assert clips[0][0].endswith("NT-clip27_canonical.mp4"), clips[0]
        assert clips[0][1].endswith("NT-clip27_hubert.wav"), clips[0]
        assert clips[1][0].endswith("NT-clip28_canonical.mp4"), clips[1]
        assert clips[1][1].endswith("NT-clip28_hubert.wav"), clips[1]
        print("✅ 1. clips paired + ordered by file_index (suffix-tolerant)")

        # --- 2. coverage check: missing wav → raise ----------------------
        (bucket / "NT-clip28_hubert.wav").unlink()
        try:
            collect_recording_clips(bucket, bucket / "pipeline_output.json")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError when a wav is missing")
        print("✅ 2. missing clip file → FileNotFoundError (coverage enforced)")

    print("\nrecording_intake verification OK — batch front-door pairing ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
