"""
verify_acoustic_gating.py — window-level acoustic block gated by is_audio_active
=================================================================================
Regression test for a major data-integrity bug found + fixed 2026-07-07
(adversarial review of the single-pass WavLM rewrite): the 20-column
window-level acoustic block (ACOUSTIC_COLUMN_NAMES, via
WavLMAcousticExtractor.extract_window_features) was only ever nulled by the
isolated WAV's own RMS floor — a check that is blind to WHO is speaking.
Since audio isolation ATTENUATES (does not zero) non-target audio, a loud
interviewer segment can sit above that floor, and its acoustic profile would
be written into the window record as if it were the target's own behavior —
directly undermining the personal-behavioral-fingerprint premise the whole
pipeline is built on.

The fix (analytics/dynamic_window_engine.py + analytics/temporal_window_generator.py):
gate the acoustic block on the window's confidence-weighted mean of
is_audio_active (the ground-truth diarizer signal, independent of RMS),
requiring the target to be verifiably speaking for at least half the window.

Pure pandas/numpy on a synthetic 30 fps raw CSV with a mock acoustic
extractor (no GPU, no real WavLM) — mirrors the house verify_*.py style.
Run from deception_detection/:  python tests/verify_acoustic_gating.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.dynamic_window_engine import DynamicWindowEngine  # noqa: E402
from audio_isolation.core.acoustic_extractor import ACOUSTIC_COLUMN_NAMES  # noqa: E402

PASS = 0
FAIL = 0


def check(description: str, condition: bool):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"✅ {description}")
    else:
        FAIL += 1
        print(f"❌ {description}")


class _MockAcousticExtractor:
    """Always returns a fixed, obviously-nonzero acoustic block — real WavLM
    output would vary, but the gate must null it regardless of what the
    (mocked) extractor returns, purely from is_audio_active."""

    def __init__(self):
        self.call_count = 0

    def extract_window_features(self, start_ms, end_ms):
        self.call_count += 1
        return {col: 42.0 for col in ACOUSTIC_COLUMN_NAMES}


AU_COLS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]


def make_raw_csv(path: Path, n_frames: int = 90, active_until_ms: float = 1500.0):
    """90 frames @ 30fps = 3.0s: is_audio_active=1 for t < active_until_ms,
    else 0. Window [0,2000) is then 75% active (speaking); window
    [1000,3000) is 25% active (not speaking) — a clean gate test on both
    sides of the 0.5 threshold."""
    ts = np.arange(n_frames) * (1000.0 / 30.0)
    df = pd.DataFrame({
        "timestamp": ts,
        "left_wrist_velocity": 0.1, "right_wrist_velocity": 0.1,
        "macro_motion_energy": 1.0,
        "left_hand_face_distance": 100.0, "right_hand_face_distance": 100.0,
        "emotion_confidence": 0.9, "emotion_label": "Neutral",
        "gaze_x": 0.0, "gaze_y": 0.0, "gaze_z": 1.0,
        "head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0,
        "is_audio_active": (ts < active_until_ms).astype(float),
    }, index=range(n_frames))
    for au in AU_COLS:
        df[au] = 0.1
    df.to_csv(path, index=False)
    return path


def main():
    engine = DynamicWindowEngine(window_size_ms=2000.0, stride_ms=1000.0,
                                 min_fill_rate=0.25, min_confidence_threshold=0.35)

    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = make_raw_csv(Path(tmp) / "raw.csv")
        out_csv = Path(tmp) / "windowed.csv"
        extractor = _MockAcousticExtractor()

        engine.compile_sliding_windows(str(raw_csv), str(out_csv), acoustic_extractor=extractor)
        windowed = pd.read_csv(out_csv)

        w0 = windowed[windowed["start_time_ms"] == 0.0].iloc[0]     # 75% active → speaking
        w1 = windowed[windowed["start_time_ms"] == 1000.0].iloc[0]  # 25% active → not speaking

        check("1. window with is_audio_active mostly 1 (75%) gets the real "
              "acoustic values (extractor's mock output passes through)",
              all(np.isclose(w0[col], 42.0) for col in ACOUSTIC_COLUMN_NAMES))
        check("2. window with is_audio_active mostly 0 (25%, mostly the "
              "interviewer) has ALL 20 acoustic columns nulled to NaN — "
              "the leak the review found is closed",
              all(np.isnan(w1[col]) for col in ACOUSTIC_COLUMN_NAMES))

        # The gate short-circuits BEFORE calling extract_window_features when
        # the target isn't speaking (no point computing a value that's about
        # to be discarded) — so across the two windows in this fixture
        # (one speaking, one not), the extractor should be invoked exactly
        # once, not twice.
        check("3. the extractor is only invoked for the speaking window — "
              "the gate short-circuits rather than computing-then-discarding",
              extractor.call_count == 1)

        # No is_audio_active column at all (legacy single-clip raw CSVs
        # predating this column) must not crash and must NOT gate — the
        # original ungated behavior is the correct fallback there.
        raw_csv_legacy = make_raw_csv(Path(tmp) / "raw_legacy.csv")
        df_legacy = pd.read_csv(raw_csv_legacy).drop(columns=["is_audio_active"])
        df_legacy.to_csv(raw_csv_legacy, index=False)
        out_legacy = Path(tmp) / "windowed_legacy.csv"
        engine.compile_sliding_windows(str(raw_csv_legacy), str(out_legacy),
                                       acoustic_extractor=_MockAcousticExtractor())
        legacy = pd.read_csv(out_legacy)
        w0_legacy = legacy[legacy["start_time_ms"] == 0.0].iloc[0]
        check("4. legacy raw CSV with no is_audio_active column: acoustic "
              "block passes through ungated (backward compatible)",
              all(np.isclose(w0_legacy[col], 42.0) for col in ACOUSTIC_COLUMN_NAMES))

    print(f"\n{'=' * 70}")
    print(f"VERIFICATION RESULTS: {PASS}/{PASS + FAIL} checks passed")
    if FAIL == 0:
        print("🏆 SUCCESS — window-level acoustic is_audio_active gating verified.")
        return 0
    print("❌ FAILURES PRESENT.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
