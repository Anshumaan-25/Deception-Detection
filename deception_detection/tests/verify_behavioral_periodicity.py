"""
Target #16 Verification: Behavioral Periodicity (FFT Engine)
=============================================================
Validates the complete FFT spectral extraction pipeline:

  1. Pure sine wave frequency isolation (6 Hz → tremor band detection)
  2. Dual-band separation (1 Hz somatic + 7 Hz tremor in same channel)
  3. NaN gap handling (short gaps interpolated, long gaps → NaN rejection)
  4. DC offset rejection (large constant offset doesn't bleed into bands)
  5. Schema completeness (all 36 FFT columns present and correctly named)
  6. Spectral entropy discrimination (periodic signal vs random noise)
  7. Edge cases (empty DataFrame, too-short block, all-NaN channel)

Run:
    python -m tests.verify_behavioral_periodicity
    -- or --
    cd SPOVNOB_CLONE && python tests/verify_behavioral_periodicity.py
"""
import sys
import os

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analytics.confidence_math import (
    _interpolate_short_gaps,
    _linear_detrend,
    _compute_band_metrics,
    compute_fft_block_features,
    FFT_COLUMN_NAMES,
    FFT_TARGET_CHANNELS,
    FFT_FS,
    FFT_LOOKBACK_MS,
)

# ═════════════════════════════════════════════════════════════════
# Test Utilities
# ═════════════════════════════════════════════════════════════════

PASS_COUNT = 0
FAIL_COUNT = 0


def _assert(condition, label):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✅ {label}")
    else:
        FAIL_COUNT += 1
        print(f"  ❌ FAIL: {label}")


def _make_synthetic_df(duration_sec=5.0, fs=30.0, inject=None, inject_gaps=None):
    """
    Build a synthetic raw 30fps DataFrame with known frequency content.

    inject: dict of {channel_name: [(freq_hz, amplitude), ...]}
    inject_gaps: dict of {channel_name: [(start_idx, length), ...]}
    """
    n = int(duration_sec * fs)
    timestamps = np.linspace(0, duration_sec * 1000, n, endpoint=False)
    t = np.arange(n) / fs  # time in seconds

    data = {"timestamp": timestamps}
    for ch in FFT_TARGET_CHANNELS:
        if inject and ch in inject:
            signal = np.zeros(n)
            for freq, amp in inject[ch]:
                signal += amp * np.sin(2 * np.pi * freq * t)
            data[ch] = signal
        else:
            data[ch] = np.zeros(n)

    df = pd.DataFrame(data)

    if inject_gaps:
        for ch, gaps in inject_gaps.items():
            for start_idx, length in gaps:
                end_idx = min(start_idx + length, len(df))
                df.loc[start_idx:end_idx - 1, ch] = np.nan

    return df


# ═════════════════════════════════════════════════════════════════
# Test 1: Pure 6 Hz Sine Wave → Tremor Band Isolation
# ═════════════════════════════════════════════════════════════════

def test_pure_6hz_tremor():
    print("\n🔬 Test 1: Pure 6 Hz Sine Wave → Tremor Band Isolation")
    df = _make_synthetic_df(inject={"head_pitch": [(6.0, 1.0)]})

    result = compute_fft_block_features(df, end_time_ms=5000.0)

    _assert(len(result) == 36, f"Output has exactly 36 keys (got {len(result)})")

    # Tremor band should detect 6 Hz
    _assert(result["head_pitch_tremor_band_power"] > 0,
            f"Tremor band power > 0: {result['head_pitch_tremor_band_power']:.6f}")
    _assert(abs(result["head_pitch_tremor_dominant_freq"] - 6.0) < 0.5,
            f"Tremor dominant freq ≈ 6.0 Hz: {result['head_pitch_tremor_dominant_freq']:.2f}")

    # Somatic band should have near-zero power (6 Hz is outside 0.5-2 Hz)
    _assert(result["head_pitch_somatic_band_power"] < result["head_pitch_tremor_band_power"] * 0.01,
            f"Somatic band power negligible vs tremor: "
            f"{result['head_pitch_somatic_band_power']:.8f} << {result['head_pitch_tremor_band_power']:.6f}")

    # Other channels (all zeros) should have near-zero power
    _assert(result["head_yaw_tremor_band_power"] < 1e-10,
            f"Zero-signal channel has ~0 tremor power: {result['head_yaw_tremor_band_power']:.2e}")


# ═════════════════════════════════════════════════════════════════
# Test 2: Dual-Band Separation (1 Hz + 7 Hz in Same Channel)
# ═════════════════════════════════════════════════════════════════

def test_dual_band_separation():
    print("\n🔬 Test 2: Dual-Band Separation (1 Hz somatic + 7 Hz tremor)")
    df = _make_synthetic_df(inject={
        "head_yaw": [(1.0, 2.0), (7.0, 1.5)]  # 1 Hz somatic + 7 Hz tremor
    })

    result = compute_fft_block_features(df, end_time_ms=5000.0)

    # Somatic band should detect 1 Hz
    _assert(abs(result["head_yaw_somatic_dominant_freq"] - 1.0) < 0.5,
            f"Somatic dominant freq ≈ 1.0 Hz: {result['head_yaw_somatic_dominant_freq']:.2f}")
    _assert(result["head_yaw_somatic_band_power"] > 0,
            f"Somatic band power > 0: {result['head_yaw_somatic_band_power']:.6f}")

    # Tremor band should detect 7 Hz
    _assert(abs(result["head_yaw_tremor_dominant_freq"] - 7.0) < 0.5,
            f"Tremor dominant freq ≈ 7.0 Hz: {result['head_yaw_tremor_dominant_freq']:.2f}")
    _assert(result["head_yaw_tremor_band_power"] > 0,
            f"Tremor band power > 0: {result['head_yaw_tremor_band_power']:.6f}")


# ═════════════════════════════════════════════════════════════════
# Test 3: Short NaN Gap Interpolation (10 frames → features survive)
# ═════════════════════════════════════════════════════════════════

def test_short_gap_interpolation():
    print("\n🔬 Test 3: Short Gap Interpolation (10-frame NaN gap)")
    df = _make_synthetic_df(
        inject={"macro_motion_energy": [(6.0, 1.0)]},
        inject_gaps={"macro_motion_energy": [(50, 10)]}  # 10-frame gap at frame 50
    )

    result = compute_fft_block_features(df, end_time_ms=5000.0)

    # Short gap should be interpolated — features should NOT be NaN
    _assert(not np.isnan(result["macro_motion_energy_tremor_band_power"]),
            "Missing gap block failed to compute band power")
    _assert(not np.isnan(result["macro_motion_energy_tremor_dominant_freq"]),
            "Missing gap block failed to compute dominant freq")

    # The 6Hz frequency should still dominate since it was 100 frames long minus 10 gap frames
    _assert(abs(result["macro_motion_energy_tremor_dominant_freq"] - 6.0) < 1.0,
            f"Frequency still ≈ 6 Hz despite gap: {result['macro_motion_energy_tremor_dominant_freq']:.2f}")

    # Also verify the interpolation function directly
    signal_with_gap = np.ones(50)
    signal_with_gap[20:30] = np.nan
    interp_signal, miss_rate = _interpolate_short_gaps(signal_with_gap, max_gap=15)
    _assert(miss_rate == 0.0,
            f"10-frame gap fully interpolated (miss_rate=0.0): {miss_rate:.4f}")
    _assert(not np.any(np.isnan(interp_signal)),
            "No NaN remaining after interpolation of 10-frame gap")


# ═════════════════════════════════════════════════════════════════
# Test 4: Long Gap Rejection (>30% NaN → Features ARE NaN)
# ═════════════════════════════════════════════════════════════════

def test_long_gap_rejection():
    print("\n🔬 Test 4: Long Gap Rejection (>30% NaN → NaN output)")
    # Create 5 seconds of data (150 frames), FFT uses last 4s (120 frames)
    # Inject a 50-frame NaN gap (>30% of 120 = 36+ frames) that is also > 15 frames
    df = _make_synthetic_df(
        inject={"ear": [(6.0, 1.0)]},
        inject_gaps={"ear": [(60, 50)]}  # 50-frame gap starting at frame 60
    )

    result = compute_fft_block_features(df, end_time_ms=5000.0)

    # The ear channel should be fully NaN due to >30% missing
    _assert(np.isnan(result["ear_tremor_band_power"]),
            "Tremor band power IS NaN for ear (>30% gap)")
    _assert(np.isnan(result["ear_tremor_dominant_freq"]),
            "Tremor dominant freq IS NaN for ear (>30% gap)")
    _assert(np.isnan(result["ear_somatic_band_power"]),
            "Somatic band power IS NaN for ear (>30% gap)")

    # Other channels (no gaps) should still work
    _assert(not np.isnan(result["head_pitch_tremor_band_power"]),
            "head_pitch features are NOT NaN (no gaps in this channel)")


# ═════════════════════════════════════════════════════════════════
# Test 5: DC Offset Rejection
# ═════════════════════════════════════════════════════════════════

def test_dc_rejection():
    print("\n🔬 Test 5: DC Offset Rejection (large offset + small oscillation)")
    n = int(5.0 * 30)
    timestamps = np.linspace(0, 5000, n, endpoint=False)
    t = np.arange(n) / 30.0

    data = {"timestamp": timestamps}
    for ch in FFT_TARGET_CHANNELS:
        data[ch] = np.zeros(n)

    # head_pitch: large DC offset (100.0) + small 1 Hz oscillation (amp=0.5)
    data["head_pitch"] = 100.0 + 0.5 * np.sin(2 * np.pi * 1.0 * t)
    df = pd.DataFrame(data)

    result = compute_fft_block_features(df, end_time_ms=5000.0)

    # Somatic band should detect 1 Hz, NOT be dominated by DC
    _assert(abs(result["head_pitch_somatic_dominant_freq"] - 1.0) < 0.5,
            f"Somatic dominant freq ≈ 1.0 Hz despite DC=100: "
            f"{result['head_pitch_somatic_dominant_freq']:.2f}")
    _assert(result["head_pitch_somatic_band_power"] > 0,
            f"Somatic band power > 0: {result['head_pitch_somatic_band_power']:.6f}")


# ═════════════════════════════════════════════════════════════════
# Test 6: Schema Completeness (36 columns, exact names)
# ═════════════════════════════════════════════════════════════════

def test_schema_completeness():
    print("\n🔬 Test 6: Schema Completeness")
    _assert(len(FFT_COLUMN_NAMES) == 36,
            f"FFT_COLUMN_NAMES has exactly 36 entries: {len(FFT_COLUMN_NAMES)}")

    df = _make_synthetic_df()
    result = compute_fft_block_features(df, end_time_ms=5000.0)

    _assert(len(result) == 36,
            f"Output dict has exactly 36 keys: {len(result)}")

    # Verify all expected column names are present
    missing_cols = [col for col in FFT_COLUMN_NAMES if col not in result]
    _assert(len(missing_cols) == 0,
            f"All FFT column names present in output (missing: {missing_cols})")

    # Verify naming convention: each should contain channel + band + metric
    for col in FFT_COLUMN_NAMES:
        has_band = "_tremor_" in col or "_somatic_" in col
        has_metric = col.endswith("_band_power") or col.endswith("_dominant_freq") or col.endswith("_spectral_entropy")
        _assert(has_band and has_metric,
                f"Column '{col}' follows naming convention")


# ═════════════════════════════════════════════════════════════════
# Test 7: Spectral Entropy — Periodic vs Random
# ═════════════════════════════════════════════════════════════════

def test_spectral_entropy_discrimination():
    print("\n🔬 Test 7: Spectral Entropy Discrimination (periodic vs random)")
    np.random.seed(42)

    # Pure 6 Hz sine (very periodic → low entropy)
    df_periodic = _make_synthetic_df(inject={"head_pitch": [(6.0, 1.0)]})
    result_periodic = compute_fft_block_features(df_periodic, end_time_ms=5000.0)

    # Random noise (chaotic → high entropy)
    n = int(5.0 * 30)
    timestamps = np.linspace(0, 5000, n, endpoint=False)
    data = {"timestamp": timestamps}
    for ch in FFT_TARGET_CHANNELS:
        data[ch] = np.zeros(n)
    data["head_pitch"] = np.random.randn(n) * 1.0
    df_random = pd.DataFrame(data)
    result_random = compute_fft_block_features(df_random, end_time_ms=5000.0)

    periodic_entropy = result_periodic["head_pitch_tremor_spectral_entropy"]
    random_entropy = result_random["head_pitch_tremor_spectral_entropy"]

    _assert(not np.isnan(periodic_entropy), f"Periodic entropy is finite: {periodic_entropy:.4f}")
    _assert(not np.isnan(random_entropy), f"Random entropy is finite: {random_entropy:.4f}")
    _assert(random_entropy > periodic_entropy,
            f"Random entropy ({random_entropy:.4f}) > Periodic entropy ({periodic_entropy:.4f})")


# ═════════════════════════════════════════════════════════════════
# Test 8: Edge Cases
# ═════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n🔬 Test 8: Edge Cases")

    # 8a: Empty DataFrame → all NaN
    empty_df = pd.DataFrame({"timestamp": []})
    result = compute_fft_block_features(empty_df, end_time_ms=5000.0)
    _assert(len(result) == 36, f"Empty DF → 36 NaN keys: {len(result)}")
    _assert(all(np.isnan(v) for v in result.values()),
            "All values NaN for empty DataFrame")

    # 8b: Too-short block (< 8 frames)
    short_df = pd.DataFrame({
        "timestamp": [0, 33, 66, 100, 133],
        "head_pitch": [1, 2, 3, 4, 5],
    })
    result = compute_fft_block_features(short_df, end_time_ms=200.0)
    _assert(all(np.isnan(v) for v in result.values()),
            "All values NaN for too-short block (< 8 frames)")

    # 8c: All-NaN channel
    n = int(5.0 * 30)
    timestamps = np.linspace(0, 5000, n, endpoint=False)
    data = {"timestamp": timestamps}
    for ch in FFT_TARGET_CHANNELS:
        data[ch] = np.zeros(n)
    data["head_pitch"] = np.full(n, np.nan)  # Entire channel is NaN
    df = pd.DataFrame(data)
    result = compute_fft_block_features(df, end_time_ms=5000.0)
    _assert(np.isnan(result["head_pitch_tremor_band_power"]),
            "All-NaN channel → tremor band_power is NaN")
    _assert(np.isnan(result["head_pitch_somatic_band_power"]),
            "All-NaN channel → somatic band_power is NaN")

    # Other zero-filled channels should still compute (band_power ≈ 0)
    _assert(not np.isnan(result["head_yaw_tremor_band_power"]),
            "Zero-filled channel → tremor band_power is NOT NaN")


# ═════════════════════════════════════════════════════════════════
# Test 9: Linear Detrend Unit Test
# ═════════════════════════════════════════════════════════════════

def test_linear_detrend():
    print("\n🔬 Test 9: Linear Detrend Unit Test")
    # Signal with known linear trend: y = 5 + 0.3*x
    x = np.arange(100, dtype=np.float64)
    trend = 5.0 + 0.3 * x
    pure_signal = np.sin(2 * np.pi * 6.0 * x / 30.0)
    signal_with_trend = pure_signal + trend

    detrended = _linear_detrend(signal_with_trend)

    # After detrending, mean should be ~0 and signal should match pure sine
    _assert(abs(np.mean(detrended)) < 0.01,
            f"Detrended signal mean ≈ 0: {np.mean(detrended):.6f}")

    # Correlation with pure signal should be very high
    corr = np.corrcoef(detrended, pure_signal)[0, 1]
    _assert(corr > 0.99,
            f"Detrended signal correlates with pure sine (r={corr:.6f})")


# ═════════════════════════════════════════════════════════════════
# Test 10: Interpolation Function Unit Tests
# ═════════════════════════════════════════════════════════════════

def test_interpolation_unit():
    print("\n🔬 Test 10: Interpolation Function Unit Tests")

    # 10a: No NaN → passthrough
    clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    interp, rate = _interpolate_short_gaps(clean)
    _assert(rate == 0.0, f"Clean signal miss rate = 0.0: {rate}")
    _assert(np.array_equal(interp, clean), "Clean signal unchanged")

    # 10b: Short gap (3 frames) → linear interpolation
    gapped = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
    interp, rate = _interpolate_short_gaps(gapped, max_gap=5)
    _assert(rate == 0.0, f"Short gap fully interpolated: miss_rate={rate}")
    _assert(abs(interp[1] - 2.0) < 0.01, f"Interpolated value [1] ≈ 2.0: {interp[1]:.4f}")
    _assert(abs(interp[2] - 3.0) < 0.01, f"Interpolated value [2] ≈ 3.0: {interp[2]:.4f}")
    _assert(abs(interp[3] - 4.0) < 0.01, f"Interpolated value [3] ≈ 4.0: {interp[3]:.4f}")

    # 10c: Gap too long → remains NaN
    long_gap = np.array([1.0] + [np.nan] * 20 + [5.0])
    interp, rate = _interpolate_short_gaps(long_gap, max_gap=15)
    _assert(rate > 0.0, f"Long gap NOT interpolated: miss_rate={rate:.4f}")
    _assert(np.isnan(interp[10]), "Middle of long gap remains NaN")

    # 10d: Edge gap (start of signal)
    edge_gap = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
    interp, rate = _interpolate_short_gaps(edge_gap, max_gap=5)
    _assert(rate == 0.0, f"Edge gap interpolated: miss_rate={rate}")
    _assert(abs(interp[0] - 3.0) < 0.01, f"Edge gap filled with nearest: {interp[0]:.4f}")


# ═════════════════════════════════════════════════════════════════
# Main Runner
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" TARGET #16 VERIFICATION: Behavioral Periodicity (FFT Engine)")
    print("=" * 70)

    test_pure_6hz_tremor()
    test_dual_band_separation()
    test_short_gap_interpolation()
    test_long_gap_rejection()
    test_dc_rejection()
    test_schema_completeness()
    test_spectral_entropy_discrimination()
    test_edge_cases()
    test_linear_detrend()
    test_interpolation_unit()

    print("\n" + "=" * 70)
    total = PASS_COUNT + FAIL_COUNT
    print(f" RESULTS: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print(" 🏆 ALL TESTS PASSED — Target #16 FFT Engine Verified")
    else:
        print(f" ⚠️  {FAIL_COUNT} TEST(S) FAILED — Review Required")
    print("=" * 70)

    sys.exit(0 if FAIL_COUNT == 0 else 1)
