#!/usr/bin/env python3
"""
TARGET #15: CONFIDENCE-WEIGHTED ML FUSION — VERIFICATION SUITE
===============================================================
Tests joint confidence math, piecewise Z-regularization, occupancy
drop rules, and column-schema parity between both window engines.

Updated to use the shared analytics.confidence_math module and
the standardized _var column naming convention.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path so we can import our packages
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# The acoustic extractor imports heavy GPU dependencies (transformers, torch).
# On macOS dev machines without these packages, we define the column names
# inline and mock the module to allow the verification tests to run.
try:
    from audio_isolation.core.acoustic_extractor import ACOUSTIC_COLUMN_NAMES
except ImportError:
    ACOUSTIC_COLUMN_NAMES = (
        ["acoustic_volatility", "prosodic_velocity"]
        + [f"wavlm_latent_{i}" for i in range(16)]
        + ["vocal_entropy", "acoustic_energy_rms"]
    )

import types
import sys

# Stub the acoustic_extractor module so the window engines can be imported
if "audio_isolation" not in sys.modules:
    sys.modules["audio_isolation"] = types.ModuleType("audio_isolation")
if "audio_isolation.core" not in sys.modules:
    sys.modules["audio_isolation.core"] = types.ModuleType("audio_isolation.core")
if "audio_isolation.core.acoustic_extractor" not in sys.modules:
    stub_module = types.ModuleType("audio_isolation.core.acoustic_extractor")
    stub_module.ACOUSTIC_COLUMN_NAMES = ACOUSTIC_COLUMN_NAMES
    stub_module.WavLMAcousticExtractor = None
    sys.modules["audio_isolation.core.acoustic_extractor"] = stub_module

from analytics.dynamic_window_engine import DynamicWindowEngine as DynamicWindowEngineMs
from analytics.temporal_window_generator import TemporalWindowEngine as DynamicWindowEngineSec
from analytics.confidence_math import (
    regularize_value,
    confidence_weighted_mean,
    confidence_weighted_var,
    confidence_weighted_std,
)

PASS = "✅"
FAIL = "❌"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status} {name}")
    if not condition and detail:
        print(f"     └── {detail}")

print("\n" + "=" * 70)
print("TARGET #15: CONFIDENCE-WEIGHTED ML FUSION — VERIFICATION")
print("=" * 70)

# ============================================================
# CHECK 0: Shared Math Module Unit Tests
# ============================================================
print("\n🔬 Check 0: Shared Math Module (analytics.confidence_math):")

# regularize_value: above threshold → passthrough
check("regularize_value passthrough (w >= threshold)", regularize_value(5.0, 0.9, 0.35) == 5.0)
# regularize_value: below threshold → linear suppression
check("regularize_value suppression (w < threshold)", np.allclose(regularize_value(5.0, 0.1, 0.35), 0.5))
# regularize_value: NaN propagation
check("regularize_value NaN propagation", np.isnan(regularize_value(np.nan, 0.9, 0.35)))

# confidence_weighted_var: negative variance guard (should clamp to 0, not negative)
check("Variance is always >= 0", confidence_weighted_var(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])) >= 0.0)
# confidence_weighted_var: single-sample guard
check("Variance returns NaN for single sample", np.isnan(confidence_weighted_var(np.array([5.0]), np.array([1.0]))))
# confidence_weighted_std: safe sqrt
check("confidence_weighted_std safe on NaN var", np.isnan(confidence_weighted_std(np.array([5.0]), np.array([1.0]))))

# confidence_weighted_mean: all-NaN input
check("Weighted mean returns NaN for all-NaN", np.isnan(confidence_weighted_mean(np.array([np.nan, np.nan]), np.array([1.0, 1.0]))))
# confidence_weighted_mean: all-zero weights
check("Weighted mean returns NaN for zero weights", np.isnan(confidence_weighted_mean(np.array([5.0, 3.0]), np.array([0.0, 0.0]))))


# ============================================================
# CREATE SYNTHETIC TEST DATA
# ============================================================
os.makedirs("pipeline_system_outputs/test_session", exist_ok=True)
raw_csv_path = "pipeline_system_outputs/test_session/raw_test_features.csv"

# 1. We construct 130 frames (4.33 seconds at 30 FPS) with custom confidence/tracking scenarios:
# - Frame 0 to 21: High-confidence visual tracking + Target speaking (22 frames)
# - Frame 22 to 39: Momentary visual occlusion / tracking drop (low confidence) + High AU noise spike (18 frames)
# - Frame 40 to 129: Severe tracking loss (YOLO tracking completely lost, conf = 0.0) (90 frames)
timestamps = [i * (1000.0 / 30.0) for i in range(130)] # 30 fps
pose_yolo_conf = []
face_lock_conf = []
face_extractor_conf = []
diarizer_speaker_conf = []
macro_motion_energy = []
au1_intensities = []

for idx in range(130):
    if idx < 22:
        # High confidence, target speaking
        pose_yolo_conf.append(0.95)
        face_lock_conf.append(0.92)
        face_extractor_conf.append(0.90)
        diarizer_speaker_conf.append(0.90) # active target speech
        macro_motion_energy.append(1.0)
        au1_intensities.append(1.5)
    elif idx < 40:
        # Momentary visual occlusion: low confidence, target silent, high noise spike (agitation anomaly)
        pose_yolo_conf.append(0.20)
        face_lock_conf.append(0.15)
        face_extractor_conf.append(0.10)
        diarizer_speaker_conf.append(1.0) # silent target
        macro_motion_energy.append(5.0) # noisy motion spike!
        au1_intensities.append(4.0) # noisy AU spike!
    else:
        # Severe tracking loss: complete tracking drop
        pose_yolo_conf.append(0.0)
        face_lock_conf.append(0.0)
        face_extractor_conf.append(0.0)
        diarizer_speaker_conf.append(1.0)
        macro_motion_energy.append(np.nan)
        au1_intensities.append(np.nan)

# Generate baseline pose and openface data dict
mock_data = {
    "timestamp": timestamps,
    "yolo_conf": pose_yolo_conf,
    "facelock_conf": face_lock_conf,
    "face_confidence": face_extractor_conf,
    "diarizer_conf": diarizer_speaker_conf,
    "left_wrist_velocity": [0.1] * 130,
    "right_wrist_velocity": [0.1] * 130,
    "macro_motion_energy": macro_motion_energy,
    "left_hand_face_distance": [100.0] * 130,
    "right_hand_face_distance": [100.0] * 130,
    "emotion_confidence": [0.85] * 130,
    "emotion_label": ["Happy"] * 130,
    "gaze_x": [0.05] * 130,
    "gaze_y": [-0.05] * 130,
    "gaze_z": [0.9] * 130,
    "head_yaw": [2.0] * 130,
    "head_pitch": [-1.0] * 130,
    "head_roll": [0.5] * 130,
    "AU1": au1_intensities,
    "AU2": [0.1] * 130,
    "AU4": [0.1] * 130,
    "AU6": [0.1] * 130,
    "AU9": [0.1] * 130,
    "AU12": [0.1] * 130,
    "AU25": [0.1] * 130,
    "AU26": [0.1] * 130,
}

df_raw = pd.DataFrame(mock_data)

# Compute joint confidence frame-by-frame:
# w_t = c_yolo * c_facelock * c_landmark * c_diarizer
df_raw["joint_confidence"] = (
    df_raw["yolo_conf"].fillna(0.0) *
    df_raw["facelock_conf"].fillna(0.0) *
    df_raw["face_confidence"].fillna(0.0) *
    df_raw["diarizer_conf"].fillna(0.0)
)

# Save to CSV
df_raw.to_csv(raw_csv_path, index=False)

print(f"\n📂 Created synthetic raw frame-level dataset at: {raw_csv_path}")

# ============================================================
# CHECK 1: Joint Confidence Vector Correctness
# ============================================================
print("\n🧮 Check 1: Joint Confidence Calculation:")
w_high = df_raw.loc[0, "joint_confidence"]
w_occ = df_raw.loc[25, "joint_confidence"]
w_drop = df_raw.loc[50, "joint_confidence"]

# High confidence math: 0.95 * 0.92 * 0.90 * 0.90 = 0.70794
check("High-confidence joint scalar computed correctly", np.allclose(w_high, 0.70794))
# Occlusion confidence math: 0.20 * 0.15 * 0.10 * 1.0 = 0.003
check("Low-confidence joint scalar computed correctly", np.allclose(w_occ, 0.003))
# Drop confidence math: 0.0 * 0.0 * 0.0 * 1.0 = 0.0
check("Complete tracking drop collapses to 0.0", w_drop == 0.0)

# ============================================================
# CHECK 2: Dynamic Window Engine Confidence aggregation
# ============================================================
print("\n🎛️ Check 2: DynamicWindowEngine (Milliseconds based):")

engine_ms = DynamicWindowEngineMs(window_size_ms=2000.0, stride_ms=1000.0, min_fill_rate=0.25)
output_ms = engine_ms.compile_sliding_windows(raw_csv_path)

df_windows_ms = pd.read_csv(output_ms)

check("DynamicWindowEngine outputs windows CSV successfully", len(df_windows_ms) > 0)

# First window spans 0ms to 2000ms. Expected: 60 frames inside.
# Cumulative confidence = sum of w_t.
# For frame 0-21: w_t = 0.70794. Sum = 15.57468
# For frame 22-39: w_t = 0.003. Sum = 0.054
# For frame 40-59: w_t = 0.0. Sum = 0.0
# Total cumulative = 15.62868
first_window = df_windows_ms.iloc[0]
check("Cumulative window confidence computed correctly", np.allclose(first_window["cumulative_confidence"], 15.62868))

# Test Z-regularization on raw noise spikes:
# The raw macro_motion_energy for frame 22-39 was 5.0 (agitation spike).
# But since joint confidence is 0.003 (below 0.35 threshold),
# the value is regularized: val_reg = 0.003 * 5.0 = 0.015
# While frame 0-21 macro_motion_energy is 1.0 (fully trusted since w_t = 0.70794 >= 0.35 threshold).
# The weighted mean is:
# sum(w_t * val_reg) / sum(w_t)
# = (22 * w_high * 1.0 + 18 * w_occ * 0.015) / 15.62868
# = (15.57468 + 0.00081) / 15.62868
# = 0.9966
w_mean_motion = first_window["macro_motion_energy_mean"]
check("Low-confidence noise spikes are regularized toward neutral baseline", np.allclose(w_mean_motion, 0.9966, rtol=1e-3), f"Expected ~0.9966, got {w_mean_motion:.4f}")

# Test dynamic occupancy drop rule:
# Let's slide a window that falls entirely inside the severe tracking drop region (frames 40-59).
# expected_frames = 60. 25% occupancy = 15 cumulative frames.
# For a window starting at 2000ms (end time 4000ms), all frames fall in 40-99 (Severe tracking loss).
# w_t for 40-119 is 0.0. Sum = 0.0.
# Cumulative sum of weights = 0.0. This is way below 15!
# Therefore, this window must be completely nullified (NaN/N/A)!
third_window = df_windows_ms.iloc[2] if len(df_windows_ms) > 2 else None
if third_window is not None:
    check("Window with <25% cumulative confidence is completely nullified", np.isnan(third_window["macro_motion_energy_mean"]))
    check("AU1 mean in corrupted window is NaN", np.isnan(third_window["AU1_mean"]))
else:
    check("Corrupted window retrieved", False, "Could not slide 3rd window.")

# ============================================================
# CHECK 3: Temporal Window Generator parity
# ============================================================
print("\n⏱️ Check 3: Temporal Window Generator (Seconds based):")

engine_sec = DynamicWindowEngineSec(window_size_sec=2.0, stride_sec=1.0)
output_sec = engine_sec.generate_windows(raw_csv_path)

df_windows_sec = pd.read_csv(output_sec)

check("temporal_window_generator outputs windows CSV successfully", len(df_windows_sec) > 0)

first_window_sec = df_windows_sec.iloc[0]
check("Parity check: Weighted mean matches dynamic engine", np.allclose(first_window_sec["macro_motion_energy_mean"], 0.9966, rtol=1e-3))

third_window_sec = df_windows_sec.iloc[2] if len(df_windows_sec) > 2 else None
if third_window_sec is not None:
    check("Parity check: Occupancy drop nullification matches", np.isnan(third_window_sec["macro_motion_energy_mean"]))
else:
    check("Parity corrupted window retrieved", False)

# ============================================================
# CHECK 4: Column Schema Parity Between Engines
# ============================================================
print("\n📐 Check 4: Column Schema Parity:")

ms_cols = set(df_windows_ms.columns)
sec_cols = set(df_windows_sec.columns)

# Both engines should output identical column sets
check("Column names are identical between engines", ms_cols == sec_cols,
      f"Only in ms engine: {ms_cols - sec_cols}. Only in sec engine: {sec_cols - ms_cols}.")

# Verify no _std columns remain (should all be _var now)
std_cols_ms = [c for c in df_windows_ms.columns if c.endswith("_std")]
std_cols_sec = [c for c in df_windows_sec.columns if c.endswith("_std")]
check("No _std columns in dynamic_window_engine output", len(std_cols_ms) == 0, f"Found: {std_cols_ms}")
check("No _std columns in temporal_window_generator output", len(std_cols_sec) == 0, f"Found: {std_cols_sec}")

# ============================================================
# CHECK 5: Co-occurrence NaN correctness
# ============================================================
print("\n🔗 Check 5: Co-occurrence Index NaN Safety:")

# In a valid window, AU values should produce numeric co-occurrence indices
check("duchenne_index is numeric in valid window", not np.isnan(first_window["duchenne_index"]))

# In a nullified window, co-occurrence indices should be NaN (not 0)
if third_window is not None:
    check("duchenne_index is NaN in nullified window", np.isnan(third_window["duchenne_index"]))
    check("cognitive_load_index is NaN in nullified window", np.isnan(third_window["cognitive_load_index"]))

# ============================================================
# CLEANUP & REPORT
# ============================================================
print("\n" + "=" * 70)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"VERIFICATION RESULTS: {passed}/{total} checks passed")
if passed == total:
    print("🏆 SUCCESS — Target #15 Confidence-Weighted ML Fusion mathematically verified.")
    sys.exit(0)
else:
    print("❌ FAILURE — Review failed checks above.")
    sys.exit(1)
