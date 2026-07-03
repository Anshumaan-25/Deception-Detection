"""
Recording Calibration Verification — Phase A
=============================================
Validates the baseline-clip calibration model (BaselineCalibrator.fit/apply)
and the recording-level assembly (analytics/recording_assembler.py) against
synthetic per-clip windowed CSVs with hand-computable statistics.

Production model under test (RECORDING_TIMELINE_AND_ACOUSTIC_UPGRADE_PLAN.md
§2-§3): the recording's first clip is a DEDICATED BASELINE VIDEO (target giving
generic info); baseline stats are fitted on ALL of its windows (no 30 s cap)
and applied to every clip — never each clip against its own opening seconds.

No GPU, no torch, no real footage — pure pandas/numpy, same style as
verify_diarization_bridge.py.

Run (from deception_detection/):  python tests/verify_recording_calibration.py
"""

import os
import sys
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.baseline_calibrator import (  # noqa: E402
    BaselineCalibrator,
    BaselineCalibrationError,
    BaselineStats,
)
from analytics.recording_assembler import (  # noqa: E402
    assemble_recording,
    RecordingAssemblyError,
)


def make_windowed_csv(path, start_times, feat_a, feat_b=None, feat_const=None,
                      question_ids=None):
    """Build a synthetic windowed-features CSV with the metadata columns the
    real DynamicWindowEngine emits, plus 2-3 synthetic feature columns."""
    n = len(start_times)
    df = pd.DataFrame({
        "window_id": range(n),
        "start_time_ms": start_times,
        "end_time_ms": [t + 2000.0 for t in start_times],
        "frame_count": [60] * n,
        "cumulative_confidence": [55.0] * n,
        "emotion_label_mode": ["neutral"] * n,
        "context_phase": [np.nan] * n,
        "question_id": question_ids if question_ids is not None else [-1] * n,
        "phase_elapsed_ms": [np.nan] * n,
        "feat_a": feat_a,
    })
    if feat_b is not None:
        df["feat_b"] = feat_b
    if feat_const is not None:
        df["feat_const"] = feat_const
    df.to_csv(path, index=False)
    return path


def main():
    calibrator = BaselineCalibrator()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── 1. fit uses the WHOLE baseline clip, not a 30 s cap ────────
        # Windows at 0/30/60/90 s: a 30 s-capped fit would see only the first.
        baseline_csv = make_windowed_csv(
            tmp / "baseline_windowed.csv",
            start_times=[0.0, 30000.0, 60000.0, 90000.0],
            feat_a=[1.0, 2.0, 3.0, 4.0],
            feat_b=[10.0, 10.0, 14.0, 14.0],
            feat_const=[5.0, 5.0, 5.0, 5.0],
        )
        stats = calibrator.fit(str(baseline_csv))
        assert stats.baseline_window_count == 4, stats.baseline_window_count
        assert math.isclose(stats.feature_means["feat_a"], 2.5), stats.feature_means
        expected_std = math.sqrt(5.0 / 3.0)  # pandas ddof=1 over [1,2,3,4]
        assert math.isclose(stats.feature_stds["feat_a"], expected_std,
                            rel_tol=1e-9), stats.feature_stds
        print("✅ 1. fit pools the WHOLE baseline clip (4 windows spanning 90 s, "
              "no 30 s cap); mean/std match hand computation")

        # ── 2. zero-std feature → NaN std in stats ─────────────────────
        assert np.isnan(stats.feature_stds["feat_const"]), stats.feature_stds
        # metadata never becomes a feature
        for col in ("question_id", "phase_elapsed_ms", "start_time_ms",
                    "frame_count", "cumulative_confidence"):
            assert col not in stats.feature_means, f"{col} leaked into features"
        print("✅ 2. zero-std → NaN; metadata/context columns never treated as features")

        # ── 3. stats JSON round-trip (incl. NaN std) ───────────────────
        stats_json = tmp / "baseline_stats.json"
        stats.to_json(str(stats_json))
        loaded = BaselineStats.from_json(str(stats_json))
        assert loaded.baseline_window_count == 4
        assert math.isclose(loaded.feature_means["feat_a"], 2.5)
        assert np.isnan(loaded.feature_stds["feat_const"])
        assert loaded.source_csv == stats.source_csv
        print("✅ 3. BaselineStats JSON round-trip preserves values and NaN stds")

        # ── 4. apply z-scores an interview clip against the BASELINE ──
        # feat_a constant at 10 → z = (10-2.5)/std everywhere. If the clip were
        # (wrongly) calibrated against its own stats, z would be 0/NaN.
        interview_csv = make_windowed_csv(
            tmp / "interview_windowed.csv",
            start_times=[0.0, 2000.0],
            feat_a=[10.0, 10.0],
            feat_b=[12.0, 16.0],
            feat_const=[5.0, 9.0],
            question_ids=[3, 7],
        )
        interview_out = tmp / "interview_calibrated.csv"
        calibrator.apply(str(interview_csv), stats, str(interview_out))
        cal = pd.read_csv(interview_out)
        expected_z = (10.0 - 2.5) / expected_std
        assert np.allclose(cal["feat_a"], expected_z), cal["feat_a"].tolist()
        assert not np.allclose(cal["feat_a"], 0.0), \
            "clip was z-scored against its own mean, not the baseline's"
        print(f"✅ 4. interview clip z-scored against BASELINE stats "
              f"(z={expected_z:.4f}, not 0)")

        # ── 5. zero-std feature → NaN after apply; metadata left raw ──
        assert cal["feat_const"].isna().all(), cal["feat_const"].tolist()
        assert cal["question_id"].tolist() == [3, 7], cal["question_id"].tolist()
        assert cal["start_time_ms"].tolist() == [0.0, 2000.0]
        print("✅ 5. uncalibrateable feature → NaN; question_id/start_time_ms untouched")

        # ── 6. deviation_magnitude present; percentile deliberately NOT ─
        assert "deviation_magnitude" in cal.columns
        assert "deviation_percentile" not in cal.columns, \
            "percentile must be ranked over the whole recording, not per clip"
        print("✅ 6. apply adds deviation_magnitude; percentile deferred to assembly")

        # ── 7. baseline applied to itself → z-means ≈ 0 ────────────────
        baseline_out = tmp / "baseline_calibrated.csv"
        calibrator.apply(str(baseline_csv), stats, str(baseline_out))
        base_cal = pd.read_csv(baseline_out)
        assert abs(base_cal["feat_a"].mean()) < 1e-9, base_cal["feat_a"].mean()
        assert abs(base_cal["feat_b"].mean()) < 1e-9, base_cal["feat_b"].mean()
        print("✅ 7. baseline clip vs its own stats → mean z ≈ 0 (sanity invariant)")

        # ── 8. fit hard-fails on an unusable baseline ──────────────────
        one_window = make_windowed_csv(tmp / "one.csv", [0.0], [1.0])
        try:
            calibrator.fit(str(one_window))
            raise AssertionError("fit accepted a 1-window baseline")
        except BaselineCalibrationError:
            pass
        all_nan = make_windowed_csv(tmp / "nan.csv", [0.0, 2000.0],
                                    [np.nan, np.nan])
        try:
            calibrator.fit(str(all_nan))
            raise AssertionError("fit accepted an all-NaN baseline")
        except BaselineCalibrationError:
            pass
        try:
            calibrator.fit(str(tmp / "does_not_exist.csv"))
            raise AssertionError("fit accepted a missing baseline CSV")
        except BaselineCalibrationError:
            pass
        print("✅ 8. fit hard-fails (BaselineCalibrationError) on <2 windows, "
              "all-NaN features, and missing CSV")

        # ── 9. assembly: rebase, order, renumber, provenance ───────────
        clip1 = tmp / "c1_cal.csv"
        clip0_df = pd.read_csv(baseline_out)
        calibrator.apply(str(interview_csv), stats, str(clip1))
        assembled_csv = tmp / "recording_calibrated.csv"
        assemble_recording(
            [
                # deliberately out of order — assembler must sort by file_index
                {"file_index": 1, "csv_path": str(clip1), "offset_ms": 92000},
                {"file_index": 0, "csv_path": str(baseline_out), "offset_ms": 0},
            ],
            str(assembled_csv),
        )
        rec = pd.read_csv(assembled_csv)
        assert len(rec) == len(clip0_df) + 2
        assert rec["window_id"].tolist() == list(range(len(rec)))
        assert rec["file_index"].tolist() == [0, 0, 0, 0, 1, 1]
        assert rec["clip_window_id"].tolist() == [0, 1, 2, 3, 0, 1]
        # clip 1's windows started at 0 / 2000 locally → 92000 / 94000 globally
        assert rec.loc[rec["file_index"] == 1, "start_time_ms"].tolist() == \
            [92000.0, 94000.0]
        assert rec.loc[rec["file_index"] == 0, "start_time_ms"].tolist() == \
            [0.0, 30000.0, 60000.0, 90000.0]
        print("✅ 9. assembly rebases times by file offsets, orders by file_index, "
              "renumbers window_id, keeps clip provenance")

        # ── 10. percentile ranked over the WHOLE recording ─────────────
        assert "deviation_percentile" in rec.columns
        max_idx = rec["deviation_magnitude"].idxmax()
        assert rec.loc[max_idx, "deviation_percentile"] == 1.0
        # interview windows (far from baseline) must outrank baseline windows
        interview_min = rec.loc[rec["file_index"] == 1, "deviation_percentile"].min()
        baseline_max = rec.loc[rec["file_index"] == 0, "deviation_percentile"].max()
        assert interview_min > baseline_max, (interview_min, baseline_max)
        print("✅ 10. deviation_percentile ranked over the combined recording; "
              "interview deviations outrank baseline's")

        # ── 11. assembly failure modes ─────────────────────────────────
        try:
            assemble_recording([], str(tmp / "x.csv"))
            raise AssertionError("assembler accepted an empty clip list")
        except RecordingAssemblyError:
            pass
        try:
            assemble_recording(
                [{"file_index": 0, "csv_path": str(tmp / "missing.csv"),
                  "offset_ms": 0}],
                str(tmp / "y.csv"),
            )
            raise AssertionError("assembler accepted a missing per-clip CSV")
        except RecordingAssemblyError:
            pass
        print("✅ 11. assembler raises on empty input and missing per-clip CSVs")

    print("\nrecording calibration verification OK — baseline-clip fit/apply + "
          "recording assembly ready (no GPU, no real footage).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
