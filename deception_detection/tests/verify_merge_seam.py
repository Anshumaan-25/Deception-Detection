"""
Merge Seam Verification — Phase 2 (audio diarization → deception detection)
==========================================================================
Validates that real audio-diarization segments (via DiarizationBridge) flow
correctly through the deception pipeline's audio-isolation seam — the path that
replaces mock_pyannote_segments at main_pipeline.py:384.

Exercises the ACTUAL deception consumer (MediaPipeAudioDiarizer) with bridge
output + synthetic audio/lip data. No GPU, no torch, no HuBERT — the
anchor+isolation path is pure numpy/scipy, so the whole audio side of the merge
is testable here. (The HuBERT/visual stages need a GPU end-to-end run; that's
Phase 4.)

Checks:
  1. Bridge → seam contract: segments_for / segments_for_clip yield
     [("TARGET", start_ms, end_ms), ...] tuples the diarizer accepts.
  2. anchor_target_identity resolves "TARGET" from the diarization segments.
  3. Isolation masks by the REAL spans: samples inside target segments are
     preserved; samples outside are attenuated by the diarizer's factor.
  4. Empty-target degradation: [] segments → TARGET_SILENT → whole track
     attenuated (no crash) — the per-clip "target silent here" case.

Run:
    python tests/verify_merge_seam.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio_isolation.core.diarization_bridge import (  # noqa: E402
    DIARIZATION_OUTPUT_SCHEMA,
    TARGET_SPEAKER_ID,
    DiarizationBridge,
)
from audio_isolation.core.diarizer_engine import MediaPipeAudioDiarizer  # noqa: E402

SR = 16000
ATT = 0.05
SIG = 8000  # constant int16 amplitude (clear inside/outside distinction)


def _summary():
    return {
        "schema": DIARIZATION_OUTPUT_SCHEMA,
        "files": [{"file_index": 0, "source_file": "/x/NT-clip27.mp4",
                   "duration_ms": 10000, "file_offset_ms": 0,
                   "silero_speech_ms": 4000, "source_sha256": "0" * 64}],
        "clean_segments": [
            {"file_index": 0, "start_local_ms": 1000, "end_local_ms": 3000,
             "start_global_ms": 1000, "end_global_ms": 3000, "duration_ms": 2000,
             "bridged_gap_count": 0, "wav_path": "a.wav", "wav_sha256": "0" * 64},
            {"file_index": 0, "start_local_ms": 5000, "end_local_ms": 7000,
             "start_global_ms": 5000, "end_global_ms": 7000, "duration_ms": 2000,
             "bridged_gap_count": 0, "wav_path": "b.wav", "wav_sha256": "1" * 64},
        ],
        "layer3": {"total_clean_ms": 4000, "segment_count": 2},
        "enrollment": {"final_quality_state": "STRONG"},
    }


def _lip_logs():
    # one log every 100 ms across 10 s; lips "moving" throughout, so the single
    # TARGET speaker is selected by the cross-modal anchor.
    return [{"timestamp_ms": t, "is_moving": 1.0, "mor_velocity": 0.5}
            for t in range(0, 10000, 100)]


def _const_wav(path, value=SIG, dur_s=10):
    wavfile.write(path, SR, np.full(SR * dur_s, value, dtype=np.int16))


def main() -> int:
    # --- 1. Bridge → seam contract ---------------------------------------
    bridge = DiarizationBridge(_summary())
    segs = bridge.segments_for(0)
    assert segs == [(TARGET_SPEAKER_ID, 1000, 3000),
                    (TARGET_SPEAKER_ID, 5000, 7000)], segs
    assert bridge.segments_for_clip("NT-clip27_canonical.mp4") == segs
    print("✅ 1. bridge → seam contract: [('TARGET', start, end)] segments")

    # --- 2. anchor resolves TARGET ---------------------------------------
    diar = MediaPipeAudioDiarizer(attenuation_factor=ATT)
    logs = _lip_logs()
    target_id, score = diar.anchor_target_identity(segs, logs)
    assert target_id == TARGET_SPEAKER_ID, target_id
    assert score > 0.0, score
    print(f"✅ 2. anchor_target_identity resolves '{target_id}' (score {score:.3f})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_wav = tmp / "canonical.wav"
        _const_wav(in_wav)
        out_dir = tmp / "audio_isolation"

        # --- 3. isolation masks by the REAL spans ------------------------
        diar.execute_isolation_pipeline(str(in_wav), logs, segs, str(out_dir))
        _, iso = wavfile.read(out_dir / "isolated_target_audio.wav")
        att_val = int(SIG * ATT)  # 400
        # inside target segments [1-3s] and [5-7s] → preserved
        assert iso[int(1.5 * SR)] == SIG and iso[int(6.0 * SR)] == SIG, "inside not preserved"
        # outside (between/after segments) → attenuated
        assert iso[int(3.5 * SR)] == att_val and iso[int(8.0 * SR)] == att_val, "outside not attenuated"
        print(f"✅ 3. isolation masks by REAL segments (inside={SIG}, outside={att_val})")

        # --- 4. empty-target degradation ---------------------------------
        diar.execute_isolation_pipeline(str(in_wav), logs, [], str(out_dir))
        _, iso2 = wavfile.read(out_dir / "isolated_target_audio.wav")
        assert (iso2 == att_val).all(), "empty target should attenuate whole track"
        print("✅ 4. empty-target ([]) → TARGET_SILENT → whole track attenuated")

    print("\nmerge seam verification OK — real diarization segments flow through "
          "anchor + isolation (GPU/HuBERT not required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
