"""
End-to-End Pipeline Unification — Integration Test Suite
==========================================================
Validates the full data cascade: Raw 30fps CSV → DynamicWindowEngine
(confidence-weighted fusion + HuBERT acoustic NaN injection + FFT
behavioral periodicity) → BaselineCalibrator (Z-score normalization)
→ Final calibrated feature matrix.

This test operates entirely on synthetic data and requires NO GPU,
no CUDA, no real video, and no real audio.

Hardware Target: Runs on macOS (developer machine) for offline validation.
Production Target: Ubuntu workstation (44 cores, RTX 6000 Ada).
"""

import sys
import os
import json
import shutil
import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.confidence_math import FFT_COLUMN_NAMES

# The acoustic extractor imports heavy GPU dependencies (transformers, torch).
# On macOS dev machines without these packages, we define the column names
# inline to allow the integration test to run without CUDA.
try:
    from audio_isolation.core.acoustic_extractor import ACOUSTIC_COLUMN_NAMES
except ImportError:
    # Mirror the canonical 20-column schema from acoustic_extractor.py
    ACOUSTIC_COLUMN_NAMES = (
        ["acoustic_volatility", "prosodic_velocity"]
        + [f"hubert_latent_{i}" for i in range(16)]
        + ["vocal_entropy", "acoustic_energy_rms"]
    )

# The DynamicWindowEngine also imports ACOUSTIC_COLUMN_NAMES from the extractor.
# We need to patch it before importing the engine.
# Strategy: Mock the acoustic_extractor module in sys.modules if it can't import.
import types
try:
    from analytics.dynamic_window_engine import DynamicWindowEngine
except ImportError:
    # Create a minimal stub for the acoustic_extractor module
    stub_module = types.ModuleType("audio_isolation.core.acoustic_extractor")
    stub_module.ACOUSTIC_COLUMN_NAMES = ACOUSTIC_COLUMN_NAMES
    stub_module.HuBERTAcousticExtractor = None

    # Ensure the package hierarchy exists in sys.modules
    if "audio_isolation" not in sys.modules:
        sys.modules["audio_isolation"] = types.ModuleType("audio_isolation")
    if "audio_isolation.core" not in sys.modules:
        sys.modules["audio_isolation.core"] = types.ModuleType("audio_isolation.core")
    sys.modules["audio_isolation.core.acoustic_extractor"] = stub_module

    # Now import should succeed
    from analytics.dynamic_window_engine import DynamicWindowEngine

from analytics.baseline_calibrator import BaselineCalibrator


# ═══════════════════════════════════════════════════════════════════
# Test Framework
# ═══════════════════════════════════════════════════════════════════
_pass_count = 0
_fail_count = 0
_assertion_count = 0


def check(description: str, condition: bool):
    """Register a single assertion result."""
    global _pass_count, _fail_count, _assertion_count
    _assertion_count += 1
    if condition:
        _pass_count += 1
        print(f"   ✅ {description}")
    else:
        _fail_count += 1
        print(f"   ❌ FAIL: {description}")


# ═══════════════════════════════════════════════════════════════════
# Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════════
def generate_synthetic_raw_csv(output_path: str, num_frames: int = 1200, fps: float = 30.0):
    """
    Generates a synthetic raw 30fps feature CSV that mirrors the exact
    column schema output by main_pipeline.py's compile_raw_features().

    This generates 1200 frames (40 seconds at 30fps), which provides:
      - 30+ seconds for baseline calibration (15+ windows at 2s/1s stride)
      - 10+ seconds of test period for deviation detection
    """
    np.random.seed(42)  # Deterministic for reproducibility

    timestamps_ms = np.arange(0, num_frames) * (1000.0 / fps)

    au_columns = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]

    records = []
    for i in range(num_frames):
        t = timestamps_ms[i]

        # Simulate a behavioral shift at 32 seconds (frame 960)
        # Baseline (0-30s): calm, low variance
        # Test period (30-40s): elevated stress indicators
        is_stress = t > 32000.0

        base_motion = 0.5 if not is_stress else 2.5
        base_au = 0.3 if not is_stress else 0.7

        record = {
            "timestamp": t,
            # Kinematics
            "left_wrist_x": 100.0 + np.random.normal(0, 2),
            "left_wrist_y": 200.0 + np.random.normal(0, 2),
            "right_wrist_x": 300.0 + np.random.normal(0, 2),
            "right_wrist_y": 200.0 + np.random.normal(0, 2),
            "nose_x": 200.0 + np.random.normal(0, 1),
            "nose_y": 150.0 + np.random.normal(0, 1),
            "left_wrist_velocity": abs(np.random.normal(base_motion, 0.5)),
            "right_wrist_velocity": abs(np.random.normal(base_motion, 0.5)),
            "left_hand_face_distance": abs(np.random.normal(120, 20)),
            "right_hand_face_distance": abs(np.random.normal(130, 20)),
            # Head Pose
            "head_yaw": np.random.normal(0, 5 if not is_stress else 15),
            "head_pitch": np.random.normal(0, 3 if not is_stress else 10),
            "head_roll": np.random.normal(0, 2),
            # Facial / Emotion
            "emotion_label": np.random.choice(["neutral", "happy", "surprise"]),
            "emotion_confidence": np.random.uniform(0.6, 0.95),
            "face_confidence": np.random.uniform(0.8, 1.0),
            # Gaze
            "gaze_x": np.random.normal(0, 0.1),
            "gaze_y": np.random.normal(0, 0.1),
            "gaze_z": np.random.normal(-1.0, 0.05),
            "gaze_velocity": abs(np.random.normal(0.02, 0.01)),
            # Cross-modal
            "mismatch_incongruence": np.random.choice([0.0, 1.0], p=[0.9, 0.1]),
            "silent_incongruence": np.random.choice([0.0, 1.0], p=[0.85, 0.15]),
            "is_audio_active": np.random.choice([0.0, 1.0], p=[0.4, 0.6]),
            # Macro body
            "macro_motion_energy": abs(np.random.normal(base_motion, 0.3)),
            "postural_stillness": float(np.exp(-0.5 * abs(np.random.normal(base_motion, 0.3)))),
            # Blink / EAR
            "ear": np.random.uniform(0.25, 0.35),
            "is_blinking": 0,
            # Confidence vector components
            "yolo_conf": np.random.uniform(0.85, 0.98),
            "facelock_conf": np.random.uniform(0.80, 0.95),
            "diarizer_conf": np.random.choice([0.9, 1.0]),
            "joint_confidence": np.random.uniform(0.5, 0.9),
        }

        # AUs with onset velocities
        for au in au_columns:
            record[au] = abs(np.random.normal(base_au, 0.15))
            record[f"{au}_velocity"] = abs(np.random.normal(0.05 if not is_stress else 0.15, 0.03))

        # Inject occasional blinks
        if i % 90 == 0:
            record["ear"] = np.random.uniform(0.12, 0.19)
            record["is_blinking"] = 1

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Synthetic Raw Data Integrity
# ═══════════════════════════════════════════════════════════════════
def test_synthetic_data_integrity(raw_csv_path: str):
    """Validate the synthetic raw CSV has the correct structure."""
    print("\n🧪 Test 1: Synthetic Raw Data Integrity")

    df = pd.read_csv(raw_csv_path)

    check("Raw CSV has 1200 frames", len(df) == 1200)
    check("Timestamps are monotonically increasing", df["timestamp"].is_monotonic_increasing)
    check("Timestamp span covers ~40 seconds", df["timestamp"].max() > 39000.0)

    # Core columns that DynamicWindowEngine requires
    required_columns = [
        "timestamp", "left_wrist_velocity", "right_wrist_velocity",
        "left_hand_face_distance", "right_hand_face_distance",
        "emotion_label", "emotion_confidence",
        "gaze_x", "gaze_y", "gaze_z",
        "head_yaw", "head_pitch", "head_roll",
        "joint_confidence", "ear", "is_blinking",
        "mismatch_incongruence", "silent_incongruence",
        "macro_motion_energy", "postural_stillness",
    ]
    for col in required_columns:
        check(f"Column '{col}' present in raw CSV", col in df.columns)

    # AU columns
    for au in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]:
        check(f"AU column '{au}' present", au in df.columns)
        check(f"AU velocity '{au}_velocity' present", f"{au}_velocity" in df.columns)


# ═══════════════════════════════════════════════════════════════════
# TEST 2: DynamicWindowEngine Chain
# ═══════════════════════════════════════════════════════════════════
def test_window_engine_chain(raw_csv_path: str, windowed_csv_path: str):
    """
    Fire DynamicWindowEngine without an acoustic extractor (acoustic
    columns will be NaN-filled per the schema contract). Validate
    output structure, window count, and column completeness.
    """
    print("\n🧪 Test 2: DynamicWindowEngine Chain Verification")

    engine = DynamicWindowEngine(
        window_size_ms=2000.0,
        stride_ms=1000.0,
        min_fill_rate=0.25,
        assumed_fps=30.0,
        min_confidence_threshold=0.35,
    )

    result = engine.compile_sliding_windows(
        raw_csv_path=raw_csv_path,
        output_csv_path=windowed_csv_path,
        acoustic_extractor=None,  # No HuBERT — acoustic columns → NaN
    )

    check("Window engine returned a result path", result is not None)
    check("Windowed CSV file was created", os.path.exists(windowed_csv_path))

    df_w = pd.read_csv(windowed_csv_path)

    # 40 seconds of data / 1s stride = ~39 windows (minus partial trailing)
    check("Window count is reasonable (30-42 range)", 30 <= len(df_w) <= 42)

    # Metadata columns
    check("window_id column present", "window_id" in df_w.columns)
    check("start_time_ms column present", "start_time_ms" in df_w.columns)
    check("end_time_ms column present", "end_time_ms" in df_w.columns)
    check("frame_count column present", "frame_count" in df_w.columns)
    check("cumulative_confidence column present", "cumulative_confidence" in df_w.columns)

    # Window IDs are sequential
    check("window_id starts at 0", df_w["window_id"].iloc[0] == 0)
    check("window_id is sequential", list(df_w["window_id"]) == list(range(len(df_w))))

    # start_time_ms is monotonically increasing
    check("start_time_ms is monotonically increasing", df_w["start_time_ms"].is_monotonic_increasing)

    # cumulative_confidence is populated for all windows
    check("cumulative_confidence has no NaN", df_w["cumulative_confidence"].notna().all())

    # Kinematic feature columns
    kinematic_cols = [
        "left_wrist_velocity_mean", "left_wrist_velocity_max",
        "right_wrist_velocity_mean", "right_wrist_velocity_max",
        "macro_motion_energy_mean", "macro_motion_energy_var",
        "left_hand_face_distance_min", "right_hand_face_distance_min",
    ]
    for col in kinematic_cols:
        check(f"Kinematic column '{col}' present", col in df_w.columns)

    # Gaze columns
    gaze_cols = ["gaze_x_mean", "gaze_x_var", "gaze_y_mean", "gaze_y_var",
                 "gaze_z_mean", "gaze_z_var", "gaze_entropy"]
    for col in gaze_cols:
        check(f"Gaze column '{col}' present", col in df_w.columns)

    # Head pose columns
    head_cols = ["head_yaw_mean", "head_yaw_var", "head_pitch_mean",
                 "head_pitch_var", "head_roll_mean", "head_roll_var"]
    for col in head_cols:
        check(f"Head pose column '{col}' present", col in df_w.columns)

    # AU aggregation columns (8 AUs × 5 metrics = 40 columns)
    au_names = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
    for au in au_names:
        for suffix in ["_mean", "_max", "_var", "_velocity_max", "_velocity_mean"]:
            col = f"{au}{suffix}"
            check(f"AU column '{col}' present", col in df_w.columns)

    # Co-occurrence indices
    cooccurrence_cols = ["duchenne_index", "cognitive_load_index",
                         "speech_hesitation_index", "disgust_leak",
                         "postural_freeze_index"]
    for col in cooccurrence_cols:
        check(f"Co-occurrence column '{col}' present", col in df_w.columns)

    # Blink columns
    check("blink_count column present", "blink_count" in df_w.columns)
    check("blink_rate column present", "blink_rate" in df_w.columns)
    check("ear_mean column present", "ear_mean" in df_w.columns)
    check("ear_var column present", "ear_var" in df_w.columns)

    # Cross-modal columns
    check("mismatch_ratio column present", "mismatch_ratio" in df_w.columns)
    check("silent_speech_duration_ms column present", "silent_speech_duration_ms" in df_w.columns)

    # Acoustic columns (should be NaN since no extractor provided)
    for col in ACOUSTIC_COLUMN_NAMES:
        check(f"Acoustic column '{col}' present (NaN-filled)", col in df_w.columns)

    # Verify acoustic columns are all NaN (no extractor = placeholder)
    acoustic_all_nan = all(df_w[col].isna().all() for col in ACOUSTIC_COLUMN_NAMES if col in df_w.columns)
    check("All acoustic columns are NaN (no extractor provided)", acoustic_all_nan)

    # FFT periodicity columns (36 columns)
    for col in FFT_COLUMN_NAMES:
        check(f"FFT column '{col}' present", col in df_w.columns)

    # Verify FFT columns have SOME non-NaN values (synthetic data is clean)
    fft_has_values = any(
        df_w[col].notna().any()
        for col in FFT_COLUMN_NAMES
        if col in df_w.columns
    )
    check("FFT columns contain computed values (not all NaN)", fft_has_values)

    return df_w


# ═══════════════════════════════════════════════════════════════════
# TEST 3: BaselineCalibrator Chain
# ═══════════════════════════════════════════════════════════════════
def test_baseline_calibrator_chain(windowed_csv_path: str, calibrated_csv_path: str):
    """
    Feed the windowed CSV into BaselineCalibrator and validate
    Z-score normalization, deviation columns, and schema preservation.
    """
    print("\n🧪 Test 3: BaselineCalibrator Chain Verification")

    calibrator = BaselineCalibrator(calibration_duration_ms=30000.0)

    result = calibrator.calibrate(
        windowed_csv_path=windowed_csv_path,
        output_csv_path=calibrated_csv_path,
    )

    check("Calibrator returned a result path", result is not None)
    check("Calibrated CSV file was created", os.path.exists(calibrated_csv_path))

    df_c = pd.read_csv(calibrated_csv_path)
    df_w = pd.read_csv(windowed_csv_path)

    # Row count preserved
    check("Row count preserved after calibration", len(df_c) == len(df_w))

    # All original columns are preserved
    for col in df_w.columns:
        check(f"Original column '{col}' preserved in calibrated output", col in df_c.columns)

    # Calibration-specific columns added
    check("deviation_magnitude column added", "deviation_magnitude" in df_c.columns)
    check("deviation_percentile column added", "deviation_percentile" in df_c.columns)

    # deviation_magnitude is populated (not all NaN)
    check("deviation_magnitude has computed values", df_c["deviation_magnitude"].notna().any())

    # deviation_percentile is bounded [0, 1]
    valid_pctile = df_c["deviation_percentile"].dropna()
    if len(valid_pctile) > 0:
        check("deviation_percentile bounded [0, 1]",
              valid_pctile.min() >= 0.0 and valid_pctile.max() <= 1.0)
    else:
        check("deviation_percentile bounded [0, 1]", False)

    # Baseline period windows (first 30s): Z-scores should be near zero
    baseline_mask = df_c["start_time_ms"] < 30000.0
    baseline_windows = df_c[baseline_mask]
    check("Baseline period has windows", len(baseline_windows) > 0)

    if len(baseline_windows) > 2:
        baseline_dev = baseline_windows["deviation_magnitude"].mean()
        # Test period deviation should be higher than baseline
        test_mask = df_c["start_time_ms"] >= 32000.0
        test_windows = df_c[test_mask]
        if len(test_windows) > 0:
            test_dev = test_windows["deviation_magnitude"].mean()
            check(
                f"Test period deviation ({test_dev:.2f}) > baseline deviation ({baseline_dev:.2f})",
                test_dev > baseline_dev
            )

    # Metadata columns are NOT Z-score normalized (they should remain unchanged)
    check("window_id values preserved", list(df_c["window_id"]) == list(df_w["window_id"]))
    check("start_time_ms values preserved",
          np.allclose(df_c["start_time_ms"].values, df_w["start_time_ms"].values))

    return df_c


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Schema Completeness Guard
# ═══════════════════════════════════════════════════════════════════
def test_schema_completeness(windowed_df, calibrated_df):
    """
    Verify that calibration does not drop any columns. The calibrated
    CSV must contain every column from the windowed CSV plus the
    calibration-added columns.
    """
    print("\n🧪 Test 4: Schema Completeness Guard")

    windowed_cols = set(windowed_df.columns)
    calibrated_cols = set(calibrated_df.columns)

    # Every windowed column must appear in calibrated
    missing = windowed_cols - calibrated_cols
    check(f"No columns dropped during calibration (missing: {missing})", len(missing) == 0)

    # Calibrated has extra columns (deviation_magnitude, deviation_percentile)
    extra = calibrated_cols - windowed_cols
    check(f"Calibration added columns: {extra}", len(extra) >= 2)
    check("deviation_magnitude in extra columns", "deviation_magnitude" in extra)
    check("deviation_percentile in extra columns", "deviation_percentile" in extra)

    # Count total feature columns (rough sanity check)
    # Expected: ~5 metadata + ~27 kinematic/facial + 40 AU + 5 co-occurrence
    #           + 4 blink + 2 cross-modal + 20 acoustic + 36 FFT + 2 calibration
    #           = ~141+ columns
    total_cols = len(calibrated_df.columns)
    check(f"Total calibrated columns ({total_cols}) exceeds 100", total_cols > 100)


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Manifest Contract Structure
# ═══════════════════════════════════════════════════════════════════
def test_manifest_structure(manifest_dir: str):
    """
    Generate a mock metadata.json matching the production contract
    and validate its JSON schema keys.
    """
    print("\n🧪 Test 5: Manifest Contract Structure")

    manifest = {
        "session_id": "TEST_SESSION",
        "status": "success",
        "stages": {
            "visual_extraction": {
                "status": "success",
                "pose_records": 1180,
                "lip_logs": 1200,
            },
            "audio_isolation": {
                "status": "success",
                "target_speaker": "SPEAKER_00",
                "correlation_score": 0.847,
            },
            "acoustic_extraction": {
                "status": "success",
                "model": "facebook/hubert-base-ls960",
                "layer": 7,
                "audio_duration_ms": 40000.0,
            },
            "raw_compilation": {
                "status": "success",
                "total_fused_frames": 1180,
            },
            "window_aggregation": {
                "status": "success",
                "total_windows": 39,
                "feature_columns": 135,
            },
            "baseline_calibration": {
                "status": "success",
                "baseline_windows": 15,
                "calibrated_features": 137,
            },
        },
        "outputs": {
            "raw_features_30fps": "/absolute/path/TEST_SESSION_raw_features_30fps.csv",
            "isolated_target_audio": "/absolute/path/audio_isolation/isolated_target_audio.wav",
            "windowed_features": "/absolute/path/TEST_SESSION_windowed_features.csv",
            "calibrated_features": "/absolute/path/TEST_SESSION_calibrated_features.csv",
        },
    }

    manifest_path = os.path.join(manifest_dir, "metadata.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    # Read back and validate structure
    with open(manifest_path, "r") as f:
        loaded = json.load(f)

    check("Manifest contains session_id", "session_id" in loaded)
    check("Manifest contains status", "status" in loaded)
    check("Manifest status is 'success'", loaded["status"] == "success")
    check("Manifest contains stages dict", isinstance(loaded.get("stages"), dict))
    check("Manifest contains outputs dict", isinstance(loaded.get("outputs"), dict))

    # Required stages
    required_stages = [
        "visual_extraction", "audio_isolation", "acoustic_extraction",
        "raw_compilation", "window_aggregation", "baseline_calibration"
    ]
    for stage in required_stages:
        check(f"Stage '{stage}' present in manifest", stage in loaded["stages"])
        check(f"Stage '{stage}' has status field", "status" in loaded["stages"].get(stage, {}))

    # Required outputs
    required_outputs = [
        "raw_features_30fps", "isolated_target_audio",
        "windowed_features", "calibrated_features"
    ]
    for output_key in required_outputs:
        check(f"Output '{output_key}' present in manifest", output_key in loaded["outputs"])

    # Stage-specific fields
    check("visual_extraction has pose_records",
          "pose_records" in loaded["stages"]["visual_extraction"])
    check("audio_isolation has target_speaker",
          "target_speaker" in loaded["stages"]["audio_isolation"])
    check("audio_isolation has correlation_score",
          "correlation_score" in loaded["stages"]["audio_isolation"])
    check("acoustic_extraction has model name",
          loaded["stages"]["acoustic_extraction"].get("model") == "facebook/hubert-base-ls960")
    check("acoustic_extraction has layer index",
          loaded["stages"]["acoustic_extraction"].get("layer") == 7)
    check("window_aggregation has total_windows",
          "total_windows" in loaded["stages"]["window_aggregation"])
    check("baseline_calibration has baseline_windows",
          "baseline_windows" in loaded["stages"]["baseline_calibration"])


# ═══════════════════════════════════════════════════════════════════
# TEST 6: End-to-End Data Integrity
# ═══════════════════════════════════════════════════════════════════
def test_end_to_end_data_integrity(raw_csv_path: str, windowed_csv_path: str, calibrated_csv_path: str):
    """
    Cross-validate the data cascade: ensure temporal coverage
    is consistent, window boundaries don't overlap incorrectly,
    and feature values are numerically sane.
    """
    print("\n🧪 Test 6: End-to-End Data Integrity")

    df_r = pd.read_csv(raw_csv_path)
    df_w = pd.read_csv(windowed_csv_path)
    df_c = pd.read_csv(calibrated_csv_path)

    # Temporal coverage: windows should span the raw data range
    raw_max_ms = df_r["timestamp"].max()
    window_max_end = df_w["end_time_ms"].max()
    check(f"Window coverage approaches raw data span ({window_max_end:.0f} vs {raw_max_ms:.0f})",
          window_max_end >= raw_max_ms * 0.9)

    # Window stride validation: consecutive windows should be ~1000ms apart
    if len(df_w) > 1:
        strides = df_w["start_time_ms"].diff().dropna().values
        check("Window stride is consistently 1000ms",
              np.allclose(strides, 1000.0, atol=1.0))

    # Window size validation: end_time - start_time should be 2000ms
    window_durations = (df_w["end_time_ms"] - df_w["start_time_ms"]).values
    check("All windows are 2000ms wide",
          np.allclose(window_durations, 2000.0, atol=1.0))

    # Frame count per window should be ~60 (2s × 30fps)
    median_frame_count = df_w["frame_count"].median()
    check(f"Median frame count per window ({median_frame_count:.0f}) is near 60",
          50 <= median_frame_count <= 70)

    # No infinite values in calibrated output
    numeric_cols = df_c.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df_c[numeric_cols].values).sum()
    check(f"No infinite values in calibrated output (found {inf_count})", inf_count == 0)

    # Joint confidence should have been used (cumulative_confidence > 0 for valid windows)
    valid_windows = df_w[df_w["frame_count"] > 0]
    check("All valid windows have positive cumulative_confidence",
          (valid_windows["cumulative_confidence"] > 0).all())


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  END-TO-END PIPELINE UNIFICATION — INTEGRATION TEST SUITE")
    print("=" * 60)

    # Create isolated test workspace inside the project directory
    test_workspace = os.path.join(PROJECT_ROOT, ".test_workspace_e2e")
    if os.path.exists(test_workspace):
        shutil.rmtree(test_workspace)
    os.makedirs(test_workspace)
    print(f"\n📁 Test workspace: {test_workspace}")

    try:
        raw_csv_path = os.path.join(test_workspace, "raw_features_30fps.csv")
        windowed_csv_path = os.path.join(test_workspace, "windowed_features.csv")
        calibrated_csv_path = os.path.join(test_workspace, "calibrated_features.csv")

        # Generate synthetic data
        print("\n🔧 Generating synthetic raw feature CSV (1200 frames / 40 seconds)...")
        generate_synthetic_raw_csv(raw_csv_path, num_frames=1200, fps=30.0)

        # Execute test sequence
        test_synthetic_data_integrity(raw_csv_path)

        windowed_df = test_window_engine_chain(raw_csv_path, windowed_csv_path)

        calibrated_df = test_baseline_calibrator_chain(windowed_csv_path, calibrated_csv_path)

        test_schema_completeness(windowed_df, calibrated_df)

        test_manifest_structure(test_workspace)

        test_end_to_end_data_integrity(raw_csv_path, windowed_csv_path, calibrated_csv_path)

    finally:
        # Clean up temporary workspace
        shutil.rmtree(test_workspace, ignore_errors=True)
        print(f"\n🧹 Test workspace cleaned up.")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS: {_pass_count}/{_assertion_count} passed, {_fail_count} failed")
    if _fail_count == 0:
        print("  🏆 ALL TESTS PASSED — End-to-End Pipeline Unification Verified")
    else:
        print(f"  ⚠️  {_fail_count} FAILURES DETECTED — Review test output above")
    print(f"{'='*60}")

    sys.exit(0 if _fail_count == 0 else 1)
