"""
Frame Alignment — WavLM latent → 30 fps master-clock alignment (pure math)
==========================================================================
Timestamp-bucketed alignment of the WavLM latent sequence (one frame per
20 ms conv hop) onto the video master clock (one frame per 33.33 ms), plus
the window-level feature formulas, all as pure numpy functions.

Design doctrine — drift-free by construction: every video frame's bucket is
resolved by ABSOLUTE timestamp lookup (searchsorted over latent-frame center
times), never by index ratio multiplication. Alignment error is bounded by
half a hop (±10 ms) at every frame and does not accumulate, regardless of
recording length. Both streams already share one PTS-anchored clock (SPOVNOB
is PTS-true; CanonicalStreamReader reads CAP_PROP_POS_MSEC), so there is no
inter-stream drift for index math to hide.

This module is deliberately torch/transformers/sklearn-free so the alignment
math is unit-testable on any machine (tests/verify_frame_acoustics.py). The
GPU wrapper (acoustic_extractor.WavLMAcousticExtractor) owns model inference
and delegates every formula here.
"""

import numpy as np

# ── Canonical constants ────────────────────────────────────────────────────
# WavLM's conv feature encoder strides 320 samples at 16 kHz → one latent
# frame per 20 ms, with a 400-sample (25 ms) receptive field. Frame i is
# stamped at its hop center: i*20 + 10 ms.
WAVLM_FRAME_HOP_MS = 20.0
WAVLM_FRAME_CENTER_OFFSET_MS = 10.0

LATENT_CHANNELS = 16
SILENCE_RMS_FLOOR = 0.005  # below this RMS the audio is diarizer-attenuated silence

VIDEO_FRAME_DURATION_MS = 1000.0 / 30.0  # canonicalizer enforces 30 fps CFR

# Frame-level acoustic schema (Phase 2 raw-CSV injection; ST-GAE input).
# `frame_` prefix keeps these disjoint from the 20-column window-level schema
# so DynamicWindowEngine's explicit column list is unaffected.
FRAME_ACOUSTIC_COLUMN_NAMES = (
    [f"frame_wavlm_latent_{i}" for i in range(LATENT_CHANNELS)]
    + ["frame_prosodic_velocity", "frame_acoustic_energy_rms"]
)

# Window-level schema lives in acoustic_extractor.ACOUSTIC_COLUMN_NAMES (the
# canonical import point for downstream code); window_features_from_latents
# below produces exactly those 20 keys.
_WINDOW_COLUMN_NAMES = (
    ["acoustic_volatility", "prosodic_velocity"]
    + [f"wavlm_latent_{i}" for i in range(LATENT_CHANNELS)]
    + ["vocal_entropy", "acoustic_energy_rms"]
)


def latent_frame_centers_ms(n_frames: int, start_offset_ms: float = 0.0) -> np.ndarray:
    """Center timestamps (ms) of a run of WavLM latent frames whose first
    sample sits at start_offset_ms on the recording clock."""
    return (start_offset_ms
            + np.arange(n_frames, dtype=np.float64) * WAVLM_FRAME_HOP_MS
            + WAVLM_FRAME_CENTER_OFFSET_MS)


def pool_latents_to_intervals(latents: np.ndarray, centers_ms: np.ndarray,
                              starts_ms: np.ndarray, duration_ms: float):
    """
    Mean-pool latent frames into per-interval vectors by absolute timestamp.

    Interval k is [starts_ms[k], starts_ms[k] + duration_ms); a latent frame
    belongs to it iff its center lies inside. Intervals that trap no latent
    frame yield an all-NaN row (count 0).

    Args:
        latents:    [T, H] float latent sequence (may be empty, T == 0).
        centers_ms: [T] ascending latent-frame center times.
        starts_ms:  [K] interval start times (any order).
        duration_ms: shared interval width.

    Returns:
        pooled [K, H] float64 (NaN rows where empty), counts [K] int64.

    Non-finite guard: a single inf/NaN latent (e.g. an fp16 autocast overflow)
    would otherwise poison every later interval forever — cs[k] goes to +-inf
    past the bad index, so inf-inf=NaN for every subsequent segment sum, with
    no signal that anything went wrong. Non-finite rows are excluded from the
    running sum and any interval whose bucket contains one is explicitly
    nulled (and dropped from its count), confining the damage to the actually
    poisoned intervals instead of cascading to the end of the recording.
    """
    starts_ms = np.asarray(starts_ms, dtype=np.float64)
    K = len(starts_ms)
    H = latents.shape[1] if latents.ndim == 2 else 0

    if latents.shape[0] == 0:
        return np.full((K, max(H, 1)), np.nan), np.zeros(K, dtype=np.int64)

    lo = np.searchsorted(centers_ms, starts_ms, side="left")
    hi = np.searchsorted(centers_ms, starts_ms + duration_ms, side="left")
    counts = (hi - lo).astype(np.int64)

    finite_row = np.isfinite(latents).all(axis=1)
    latents_clean = np.where(finite_row[:, None], latents, 0.0)

    # Segment sums via cumulative sum: sum(latents[lo:hi]) = cs[hi] - cs[lo].
    cs = np.zeros((latents.shape[0] + 1, latents.shape[1]), dtype=np.float64)
    np.cumsum(latents_clean, axis=0, dtype=np.float64, out=cs[1:])

    bad_cs = np.zeros(latents.shape[0] + 1, dtype=np.int64)
    np.cumsum(~finite_row, out=bad_cs[1:])

    pooled = np.full((K, latents.shape[1]), np.nan)
    counts = counts.copy()
    valid = counts > 0
    if valid.any():
        idx = np.flatnonzero(valid)
        had_nonfinite = (bad_cs[hi[valid]] - bad_cs[lo[valid]]) > 0

        means = (cs[hi[valid]] - cs[lo[valid]]) / counts[valid, None]
        means[had_nonfinite] = np.nan
        pooled[valid] = means

        # An interval that trapped a non-finite latent has zero *usable* data
        # — report it the same as count==0 (no latent frame) rather than
        # silently retaining a stale positive count.
        counts[idx[had_nonfinite]] = 0
    return pooled, counts


def interval_rms(audio: np.ndarray, sample_rate: float,
                 starts_ms: np.ndarray, duration_ms: float) -> np.ndarray:
    """RMS of the raw waveform over each [start, start+duration) interval,
    NaN where the interval traps no samples. Same cumulative-sum approach as
    the latent pooling — absolute sample indices, no drift.

    Same non-finite guard as pool_latents_to_intervals: a single inf/NaN
    sample is excluded from the running sum and poisons only the interval(s)
    that actually trap it, rather than every interval from that point on.
    """
    starts_ms = np.asarray(starts_ms, dtype=np.float64)
    lo = np.clip((starts_ms / 1000.0 * sample_rate).astype(np.int64), 0, len(audio))
    hi = np.clip(((starts_ms + duration_ms) / 1000.0 * sample_rate).astype(np.int64), 0, len(audio))
    counts = hi - lo

    finite_sample = np.isfinite(audio)
    audio_clean = np.where(finite_sample, audio, 0.0)

    cs = np.zeros(len(audio) + 1, dtype=np.float64)
    np.cumsum(np.square(audio_clean, dtype=np.float64), out=cs[1:])

    bad_cs = np.zeros(len(audio) + 1, dtype=np.int64)
    np.cumsum(~finite_sample, out=bad_cs[1:])

    rms = np.full(len(starts_ms), np.nan)
    valid = counts > 0
    if valid.any():
        had_nonfinite = (bad_cs[hi[valid]] - bad_cs[lo[valid]]) > 0
        vals = np.sqrt((cs[hi[valid]] - cs[lo[valid]]) / counts[valid])
        vals[had_nonfinite] = np.nan
        rms[valid] = vals
    return rms


# Matches torch.nn.functional.normalize's default eps — the original
# per-window implementation normalized via F.normalize before taking cosine
# similarity, which clamps a zero-norm row to the zero vector (cosine 0,
# distance 1.0) instead of raising 0/0=NaN. Both cosine helpers below
# eps-clamp their norms to preserve that parity at exact-zero rows.
_NORMALIZE_EPS = 1e-12


def cosine_velocity(pooled: np.ndarray) -> np.ndarray:
    """Frame-to-frame prosodic velocity over pooled vectors: 1 - cosine
    similarity between consecutive rows. Row 0 is NaN (no predecessor); a
    pair touching a NaN row is NaN; a zero-norm row degrades to distance 1.0
    (F.normalize parity — see _NORMALIZE_EPS)."""
    K = pooled.shape[0]
    vel = np.full(K, np.nan)
    if K < 2:
        return vel
    a, b = pooled[:-1], pooled[1:]
    na = np.maximum(np.linalg.norm(a, axis=1), _NORMALIZE_EPS)
    nb = np.maximum(np.linalg.norm(b, axis=1), _NORMALIZE_EPS)
    with np.errstate(invalid="ignore"):
        cos = (a * b).sum(axis=1) / (na * nb)
    vel[1:] = 1.0 - cos
    # NaN can still arise here only from a NaN-containing row (propagates
    # through the sum); zero-norm rows no longer produce it.
    nan_row = ~np.isfinite(a).all(axis=1) | ~np.isfinite(b).all(axis=1)
    vel[1:][nan_row] = np.nan
    return vel


def reduce_to_latent_groups(vectors: np.ndarray, latent_channels: int = LATENT_CHANNELS) -> np.ndarray:
    """[K, H] → [K, latent_channels]: mean over each contiguous group of
    H / latent_channels hidden dims (the canonical latent-profile reduction)."""
    K, H = vectors.shape
    if H % latent_channels != 0:
        raise ValueError(f"hidden size {H} not divisible by {latent_channels}")
    return vectors.reshape(K, latent_channels, H // latent_channels).mean(axis=2)


def frame_features_from_latents(latents: np.ndarray, centers_ms: np.ndarray,
                                audio: np.ndarray, sample_rate: float,
                                timestamps_ms: np.ndarray,
                                frame_duration_ms: float = VIDEO_FRAME_DURATION_MS,
                                latent_channels: int = LATENT_CHANNELS,
                                silence_rms_floor: float = SILENCE_RMS_FLOOR) -> dict:
    """
    The frame-level acoustic block: align WavLM latents + waveform energy to
    the 30 fps master clock.

    Args:
        latents:      [T, H] full-clip latent sequence (layer WAVLM_LAYER_INDEX).
        centers_ms:   [T] latent frame center times on the recording clock.
        audio:        raw isolated waveform (float32, target-only).
        sample_rate:  waveform sample rate (Hz).
        timestamps_ms: [K] video frame START times (master clock).
        frame_duration_ms: video frame width (33.33 ms at 30 fps CFR).

    Returns:
        {column: np.ndarray[K]} for every FRAME_ACOUSTIC_COLUMN_NAMES entry.
        A video frame is NaN across the board when it traps no latent frame
        OR its waveform RMS sits below the diarizer-attenuation floor —
        attenuated-silence latents are not behavioral signal. (Cross-modal
        masking by is_audio_active happens downstream in main_pipeline.)
    """
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)

    pooled, counts = pool_latents_to_intervals(latents, centers_ms, timestamps_ms, frame_duration_ms)
    rms = interval_rms(audio, sample_rate, timestamps_ms, frame_duration_ms)

    # Silence tripwire, per video frame (same doctrine as the window path).
    # A frame is nulled across the board (latents AND rms) if EITHER it trapped
    # no usable latent frame (counts==0 — including one poisoned by a
    # non-finite value, see pool_latents_to_intervals) OR its waveform RMS
    # sits below the attenuation floor. Folding both conditions into one mask
    # is required: rms is computed independently of the latents, so without
    # this a frame with valid audio but zero trapped latents would otherwise
    # keep a stray non-NaN rms while every latent column is NaN.
    silent = (counts == 0) | ~(rms >= silence_rms_floor)   # True for NaN rms too
    pooled[silent] = np.nan
    rms = np.where(silent, np.nan, rms)

    velocity = cosine_velocity(pooled)            # full-H cosine, then reduce
    groups = (reduce_to_latent_groups(pooled, latent_channels)
              if latents.shape[0] > 0 and pooled.shape[1] % latent_channels == 0
              else np.full((len(timestamps_ms), latent_channels), np.nan))

    result = {f"frame_wavlm_latent_{i}": groups[:, i] for i in range(latent_channels)}
    result["frame_prosodic_velocity"] = velocity
    result["frame_acoustic_energy_rms"] = rms
    return result


def window_features_from_latents(latents: np.ndarray, centers_ms: np.ndarray,
                                 audio: np.ndarray, sample_rate: float,
                                 start_ms: float, end_ms: float,
                                 quantize_fn,
                                 latent_channels: int = LATENT_CHANNELS,
                                 silence_rms_floor: float = SILENCE_RMS_FLOOR) -> dict:
    """
    The 20-column window-level acoustic block, computed from the cached
    full-clip latent sequence (single-pass architecture) instead of a
    dedicated per-window WavLM forward. Formulas are identical to the
    original per-window implementation (unbiased variance, cosine velocity,
    16-group latent profile, codebook Shannon entropy, waveform RMS); what
    changed is the transformer context — latents now come from ~30 s chunks
    rather than an isolated 2 s forward, which removes the per-window
    boundary effect. (No real-audio values were ever validated against the
    old path — see MASTER_REFERENCE §12 validation debt.)

    quantize_fn: [T, H] → [T] codebook assignments (the extractor passes its
    fitted MiniBatchKMeans .predict).
    """
    null_result = {col: np.nan for col in _WINDOW_COLUMN_NAMES}

    start_sample = max(0, int((start_ms / 1000.0) * sample_rate))
    end_sample = min(len(audio), int((end_ms / 1000.0) * sample_rate))
    if end_sample <= start_sample:
        return null_result

    raw_chunk = audio[start_sample:end_sample]
    rms_energy = float(np.sqrt(np.mean(np.square(raw_chunk, dtype=np.float64))))
    if rms_energy < silence_rms_floor:
        return null_result

    lo = int(np.searchsorted(centers_ms, start_ms, side="left"))
    hi = int(np.searchsorted(centers_ms, end_ms, side="left"))
    window = latents[lo:hi].astype(np.float64, copy=False)
    T = window.shape[0]
    if T < 2:
        return null_result

    # Acoustic volatility: L2 norm of the per-dim temporal variance (ddof=1,
    # matching torch.var's unbiased default in the original implementation).
    temporal_var = window.var(axis=0, ddof=1)
    acoustic_volatility = float(np.linalg.norm(temporal_var))

    # Prosodic velocity: mean consecutive cosine distance over normalized rows.
    # eps-clamped (matches the original F.normalize-based formula: a
    # zero-norm row degrades to the zero vector, i.e. cosine 0 / distance
    # 1.0, rather than 0/0=NaN — see _NORMALIZE_EPS).
    norms = np.maximum(np.linalg.norm(window, axis=1, keepdims=True), _NORMALIZE_EPS)
    normalized = window / norms
    cos = (normalized[:-1] * normalized[1:]).sum(axis=1)
    prosodic_velocity = float(np.nanmean(1.0 - cos)) if np.isfinite(cos).any() else np.nan

    # 16-channel latent profile: temporal mean → contiguous group means.
    latent_profile = reduce_to_latent_groups(window.mean(axis=0)[None, :], latent_channels)[0]

    # Vocal entropy: Shannon entropy of codebook assignments.
    assignments = quantize_fn(window.astype(np.float32, copy=False))
    _, counts = np.unique(assignments, return_counts=True)
    probs = counts / len(assignments)
    vocal_entropy = float(-np.sum(probs * np.log2(probs)))

    result = {
        "acoustic_volatility": acoustic_volatility,
        "prosodic_velocity": prosodic_velocity,
    }
    for i in range(latent_channels):
        result[f"wavlm_latent_{i}"] = float(latent_profile[i])
    result["vocal_entropy"] = vocal_entropy
    result["acoustic_energy_rms"] = rms_energy
    return result
