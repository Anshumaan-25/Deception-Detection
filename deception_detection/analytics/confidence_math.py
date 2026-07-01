"""
Confidence-Weighted Aggregation Mathematics
=============================================
Shared numeric utilities for piecewise Z-regularization and
confidence-weighted window aggregation across all Layer 4 engines.

Design Decisions:
    1. Linear Regularization (NOT Quadratic):
       For frames where w_t < threshold, the feature value is smoothly
       suppressed: Z_reg = w_t * Z_t. The weighted mean then applies
       the standard formula WITHOUT re-multiplying by w_t:
         weighted_mean = Σ(w_t * Z_reg) / Σ(w_t)
       This produces linear suppression (w * Z * w / Σw = w²Z/Σw).
       This is intentional quadratic damping for sub-threshold frames —
       it aggressively mutes noise from low-confidence tracking.

    2. Epsilon Floor:
       Denominators are guarded by 1e-9 to prevent division-by-zero
       without masking legitimate near-zero weight distributions.

    3. Variance Safety:
       confidence_weighted_var clamps its output to max(0, var) before
       returning, preventing negative floats from floating-point rounding.
       Callers using np.sqrt() on variance results are therefore safe.

    4. Single-Sample Guard:
       If fewer than 2 valid samples remain after NaN masking, variance
       returns np.nan (degenerate — no meaningful spread from 1 point).
"""

import numpy as np

# Epsilon floor for weighted denominator division
_EPS = 1e-9


def regularize_value(val: float, weight: float, threshold: float = 0.35) -> float:
    """
    Piecewise Z-regularization.

    If the frame's joint confidence (weight) is below the threshold,
    the feature value is linearly suppressed toward zero:
        Z_reg = w * Z   (when w < threshold)
        Z_reg = Z       (when w >= threshold)

    This prevents low-confidence tracking jitter from leaking anomalous
    spikes into downstream aggregation.
    """
    if np.isnan(val) or np.isnan(weight):
        return np.nan
    if weight < threshold:
        return float(weight * val)
    return float(val)


def confidence_weighted_mean(values, weights, threshold: float = 0.35) -> float:
    """
    Weighted mean with piecewise regularization.

    Formula:
        mean = Σ(w_t * Z_reg_t) / Σ(w_t)

    Where Z_reg_t is the regularized value (see regularize_value).
    For sub-threshold frames, this produces quadratic damping (w² * Z / Σw),
    which is the intended aggressive suppression of noisy low-confidence data.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not np.any(mask):
        return np.nan
    v_masked = values[mask]
    w_masked = weights[mask]
    w_sum = np.sum(w_masked)
    if w_sum < _EPS:
        return np.nan
    reg_values = np.array([regularize_value(v, w, threshold) for v, w in zip(v_masked, w_masked)])
    return float(np.nansum(reg_values * w_masked) / w_sum)


def confidence_weighted_var(values, weights, threshold: float = 0.35) -> float:
    """
    Weighted variance with piecewise regularization.

    Uses the standard formula:
        var = Σ(w_t * (Z_reg_t - μ_w)²) / Σ(w_t)

    Returns np.nan if fewer than 2 valid samples (degenerate — no spread).
    Output is clamped to max(0.0, var) to guard against negative floats
    from floating-point rounding, making np.sqrt(var) always safe.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not np.any(mask):
        return np.nan
    v_masked = values[mask]
    w_masked = weights[mask]
    # Single-sample guard: variance is undefined for n < 2
    if len(v_masked) < 2:
        return np.nan
    w_sum = np.sum(w_masked)
    if w_sum < _EPS:
        return np.nan
    reg_values = np.array([regularize_value(v, w, threshold) for v, w in zip(v_masked, w_masked)])
    w_mean = np.nansum(reg_values * w_masked) / w_sum
    w_var = np.nansum(w_masked * (reg_values - w_mean) ** 2) / w_sum
    # Clamp to prevent negative variance from floating-point rounding
    return float(max(0.0, w_var))


def confidence_weighted_max(values, weights, threshold: float = 0.35) -> float:
    """
    Maximum of regularized values (sub-threshold values are suppressed).
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not np.any(mask):
        return np.nan
    v_masked = values[mask]
    w_masked = weights[mask]
    reg_values = np.array([regularize_value(v, w, threshold) for v, w in zip(v_masked, w_masked)])
    return float(np.nanmax(reg_values))


def confidence_weighted_min(values, weights, threshold: float = 0.35) -> float:
    """
    Minimum of regularized values (sub-threshold values are suppressed).
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not np.any(mask):
        return np.nan
    v_masked = values[mask]
    w_masked = weights[mask]
    reg_values = np.array([regularize_value(v, w, threshold) for v, w in zip(v_masked, w_masked)])
    return float(np.nanmin(reg_values))


def confidence_weighted_std(values, weights, threshold: float = 0.35) -> float:
    """
    Convenience wrapper: sqrt of confidence_weighted_var.

    Safe — variance is clamped to >= 0 before sqrt.
    Returns np.nan if variance is np.nan.
    """
    var = confidence_weighted_var(values, weights, threshold)
    if np.isnan(var):
        return np.nan
    return float(np.sqrt(var))


def calculate_gaze_entropy(window_df) -> float:
    """
    Computes Shannon Gaze Entropy (H) over a 3x3 spatial grid based on
    standard boundaries [-0.15, 0.15] for gaze_x and gaze_y.

    This is a spatial distribution metric computed over discrete bins,
    so raw (unweighted) logic is correct — the gaze positions themselves
    are already filtered by the confidence gate upstream.
    """
    gaze_data = window_df[['gaze_x', 'gaze_y']].dropna()
    if len(gaze_data) == 0:
        return np.nan

    xs = gaze_data['gaze_x'].values
    ys = gaze_data['gaze_y'].values

    x_bins = np.digitize(xs, [-0.15, 0.15])
    y_bins = np.digitize(ys, [-0.15, 0.15])

    bins = x_bins * 3 + y_bins

    _, counts = np.unique(bins, return_counts=True)
    probs = counts / len(bins)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


# ═══════════════════════════════════════════════════════════════════
# Target #16: Behavioral Periodicity — FFT Spectral Engine
# ═══════════════════════════════════════════════════════════════════
#
# Extracts frequency-domain features from kinematic signals to detect
# involuntary rhythmic tells (micro-tremors, postural swaying, facial
# muscle twitching) that standard time-domain statistics (mean, var)
# cannot discriminate.
#
# Design Decisions:
#     1. Lookback Block: 4 seconds (120 frames at 30fps) provides
#        Δf = 0.25 Hz resolution — sufficient to resolve somatic
#        rhythms (0.5–2 Hz) without temporal smearing of brief tells.
#
#     2. Gap Handling: Short NaN gaps (≤15 consecutive frames) are
#        bridged via linear interpolation. Blocks with >30% residual
#        NaN after interpolation are rejected (all metrics → np.nan).
#
#     3. Pre-processing: Linear detrend removes DC offset and slow
#        drift. Hann windowing eliminates boundary spectral leakage.
#
#     4. Two Frequency Bands:
#        - Autonomic Tremor (4–10 Hz): involuntary muscle vibrations
#        - Somatic Postural (0.5–2 Hz): postural sway, breathing
#
#     5. Three Metrics Per Band Per Channel:
#        - band_power: integrated PSD energy in the band
#        - dominant_freq: peak frequency within the band (Hz)
#        - spectral_entropy: Shannon entropy of normalized PSD
# ═══════════════════════════════════════════════════════════════════

# FFT Configuration Constants
FFT_TARGET_CHANNELS = (
    "head_pitch", "head_yaw", "ear",
    "AU12_velocity", "AU25_velocity", "macro_motion_energy",
)
FFT_LOOKBACK_MS = 4000.0          # 4-second historical block
FFT_FS = 30.0                     # Sampling rate (Hz)
FFT_MAX_MISSING_RATE = 0.30       # >30% residual NaN → reject block
FFT_MAX_GAP_FRAMES = 15           # Max interpolatable gap length
AUTONOMIC_TREMOR_BAND = (4.0, 10.0)   # Hz
SOMATIC_POSTURAL_BAND = (0.5, 2.0)    # Hz

# Pre-computed column names for schema contract enforcement
# 6 channels × 2 bands × 3 metrics = 36 columns
FFT_COLUMN_NAMES = []
for _ch in FFT_TARGET_CHANNELS:
    for _band_name in ("tremor", "somatic"):
        for _metric in ("band_power", "dominant_freq", "spectral_entropy"):
            FFT_COLUMN_NAMES.append(f"{_ch}_{_band_name}_{_metric}")


def _interpolate_short_gaps(signal, max_gap=FFT_MAX_GAP_FRAMES):
    """
    Bridge short NaN gaps (≤ max_gap consecutive frames) via linear
    interpolation. Longer gaps remain as NaN.

    Returns:
        (interpolated_signal, missing_rate_after_interpolation)
    """
    result = np.array(signal, dtype=np.float64, copy=True)
    n = len(result)
    if n == 0:
        return result, 1.0
    is_nan = np.isnan(result)
    if not np.any(is_nan):
        return result, 0.0
    if np.all(is_nan):
        return result, 1.0

    # Scan for contiguous NaN runs and bridge short ones
    i = 0
    while i < n:
        if is_nan[i]:
            gap_start = i
            while i < n and is_nan[i]:
                i += 1
            gap_end = i   # exclusive
            gap_len = gap_end - gap_start

            if gap_len <= max_gap:
                # Determine boundary values for linear interpolation
                left_ok = (gap_start > 0 and not np.isnan(result[gap_start - 1]))
                right_ok = (gap_end < n and not np.isnan(result[gap_end]))
                left_val = result[gap_start - 1] if left_ok else None
                right_val = result[gap_end] if right_ok else None

                if left_val is not None and right_val is not None:
                    result[gap_start:gap_end] = np.linspace(
                        left_val, right_val, gap_len + 2
                    )[1:-1]
                elif left_val is not None:
                    result[gap_start:gap_end] = left_val
                elif right_val is not None:
                    result[gap_start:gap_end] = right_val
                # else: both boundaries NaN — leave gap intact
        else:
            i += 1

    remaining_nan = np.sum(np.isnan(result))
    return result, float(remaining_nan / n)


def _linear_detrend(signal):
    """
    Remove linear trend via least-squares regression.
    Zero-dependency implementation (no scipy required).
    """
    n = len(signal)
    if n < 2:
        return signal - np.mean(signal)
    x = np.arange(n, dtype=np.float64)
    sx = np.sum(x)
    sy = np.sum(signal)
    sxy = np.dot(x, signal)
    sx2 = np.dot(x, x)
    denom = n * sx2 - sx * sx
    if abs(denom) < _EPS:
        return signal - np.mean(signal)
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return signal - (a + b * x)


def _compute_band_metrics(psd, freqs, band_low, band_high):
    """
    Extract band_power, dominant_freq, spectral_entropy for a specific
    frequency band from a Power Spectral Density array.

    Returns dict with keys: band_power, dominant_freq, spectral_entropy.
    """
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    band_psd = psd[band_mask]
    band_freqs = freqs[band_mask]

    if len(band_psd) == 0:
        return {"band_power": np.nan, "dominant_freq": np.nan,
                "spectral_entropy": np.nan}

    psd_sum = float(np.sum(band_psd))
    if psd_sum < _EPS:
        return {"band_power": 0.0, "dominant_freq": np.nan,
                "spectral_entropy": np.nan}

    # Frequency resolution for trapezoidal integration
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Band power: integrated PSD energy
    band_power = float(np.sum(band_psd) * df)

    # Dominant frequency: location of peak PSD within band
    dominant_freq = float(band_freqs[np.argmax(band_psd)])

    # Spectral entropy: Shannon entropy of normalized PSD
    norm_psd = band_psd / psd_sum
    nonzero = norm_psd > 0
    spectral_entropy = float(-np.sum(norm_psd[nonzero] * np.log2(norm_psd[nonzero])))

    return {"band_power": band_power, "dominant_freq": dominant_freq,
            "spectral_entropy": spectral_entropy}


def compute_fft_block_features(raw_df, end_time_ms,
                                lookback_ms=FFT_LOOKBACK_MS,
                                fs=FFT_FS,
                                max_missing_rate=FFT_MAX_MISSING_RATE,
                                max_gap=FFT_MAX_GAP_FRAMES):
    """
    Master entry point for Target #16 Behavioral Periodicity extraction.

    Slices a historical block [end_time_ms - lookback_ms, end_time_ms)
    from the raw 30fps DataFrame, then for each target kinematic channel:
      1. Interpolates short NaN gaps (≤ max_gap frames)
      2. Rejects blocks with >30% residual missing data
      3. Applies linear detrend + Hann windowing
      4. Computes rfft → one-sided PSD
      5. Extracts band_power, dominant_freq, spectral_entropy for both
         Autonomic Tremor (4–10 Hz) and Somatic Postural (0.5–2 Hz)

    Returns a dict with 36 feature columns (6 channels × 2 bands × 3 metrics).
    """
    start_time_ms = max(0.0, end_time_ms - lookback_ms)
    block = raw_df[
        (raw_df["timestamp"] >= start_time_ms) & (raw_df["timestamp"] < end_time_ms)
    ]

    result = {}

    # Pre-build NaN template for fast rejection
    _nan_fill = {col: np.nan for col in FFT_COLUMN_NAMES}

    # Minimum viable block: need at least 8 samples for any meaningful FFT
    if len(block) < 8:
        result.update(_nan_fill)
        return result

    for channel in FFT_TARGET_CHANNELS:
        prefix_t = f"{channel}_tremor"
        prefix_s = f"{channel}_somatic"
        nan_ch = {
            f"{prefix_t}_band_power": np.nan,
            f"{prefix_t}_dominant_freq": np.nan,
            f"{prefix_t}_spectral_entropy": np.nan,
            f"{prefix_s}_band_power": np.nan,
            f"{prefix_s}_dominant_freq": np.nan,
            f"{prefix_s}_spectral_entropy": np.nan,
        }

        # Channel missing from DataFrame → NaN all metrics
        if channel not in block.columns:
            result.update(nan_ch)
            continue

        signal = block[channel].values.astype(np.float64)

        # 1. Interpolate short NaN gaps (≤ 15 frames)
        signal, missing_rate = _interpolate_short_gaps(signal, max_gap)

        # 2. Quality gate: too many missing frames
        if missing_rate > max_missing_rate:
            result.update(nan_ch)
            continue

        # 3. Replace residual NaN with 0 (after quality gate passed)
        signal = np.nan_to_num(signal, nan=0.0)

        # 4. Minimum sample guard
        if len(signal) < 4:
            result.update(nan_ch)
            continue

        # 5. Linear detrend (removes DC offset and slow drift)
        signal = _linear_detrend(signal)

        # 6. Hann windowing (eliminates boundary spectral leakage)
        signal = signal * np.hanning(len(signal))

        # 7. Compute one-sided PSD via rfft (standard normalization)
        N = len(signal)
        fft_coeff = np.fft.rfft(signal)
        psd = (2.0 / (N * fs)) * (np.abs(fft_coeff) ** 2)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)

        # 8. Extract band metrics
        tremor = _compute_band_metrics(psd, freqs, *AUTONOMIC_TREMOR_BAND)
        somatic = _compute_band_metrics(psd, freqs, *SOMATIC_POSTURAL_BAND)

        result[f"{prefix_t}_band_power"] = tremor["band_power"]
        result[f"{prefix_t}_dominant_freq"] = tremor["dominant_freq"]
        result[f"{prefix_t}_spectral_entropy"] = tremor["spectral_entropy"]
        result[f"{prefix_s}_band_power"] = somatic["band_power"]
        result[f"{prefix_s}_dominant_freq"] = somatic["dominant_freq"]
        result[f"{prefix_s}_spectral_entropy"] = somatic["spectral_entropy"]

    return result
