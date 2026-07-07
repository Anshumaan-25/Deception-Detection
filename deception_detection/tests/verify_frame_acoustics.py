"""
verify_frame_acoustics.py — frame-level WavLM ↔ 30 fps master-clock alignment
==============================================================================
Verifies audio_isolation/core/frame_alignment.py: the timestamp-bucketed
pooling that puts WavLM latents (20 ms hop) onto the video master clock
(33.33 ms frames), the drift-free property that motivated the design, the
silence/masking doctrine, and the window-level formula parity of the
single-pass refactor.

Pure numpy on synthetic latents — no GPU, no torch, no real audio.
Run from deception_detection/:  python tests/verify_frame_acoustics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_isolation.core.frame_alignment import (  # noqa: E402
    LATENT_CHANNELS,
    SILENCE_RMS_FLOOR,
    WAVLM_FRAME_HOP_MS,
    WAVLM_FRAME_CENTER_OFFSET_MS,
    VIDEO_FRAME_DURATION_MS,
    FRAME_ACOUSTIC_COLUMN_NAMES,
    latent_frame_centers_ms,
    pool_latents_to_intervals,
    interval_rms,
    cosine_velocity,
    reduce_to_latent_groups,
    frame_features_from_latents,
    window_features_from_latents,
)

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


SR = 16000
H = 64          # synthetic hidden size (64 / 16 channels = groups of 4)
FPS_DUR = VIDEO_FRAME_DURATION_MS


def make_latents(n, value_fn):
    """Latents whose row i is filled with value_fn(i) — value-coded frames."""
    lat = np.zeros((n, H), dtype=np.float32)
    for i in range(n):
        lat[i, :] = value_fn(i)
    return lat


def expected_bucket(t_start, n_latents):
    """Ground-truth bucket [lo, hi) of latent indices for one video frame,
    computed by exact arithmetic over centers c_i = 20 i + 10."""
    lo = int(np.ceil((t_start - WAVLM_FRAME_CENTER_OFFSET_MS) / WAVLM_FRAME_HOP_MS))
    hi = int(np.ceil((t_start + FPS_DUR - WAVLM_FRAME_CENTER_OFFSET_MS) / WAVLM_FRAME_HOP_MS))
    # searchsorted 'left' keeps centers == interval start; ceil matches that
    # for non-lattice points; adjust exact-hit edges the same way.
    if (t_start - WAVLM_FRAME_CENTER_OFFSET_MS) % WAVLM_FRAME_HOP_MS == 0:
        lo = int((t_start - WAVLM_FRAME_CENTER_OFFSET_MS) / WAVLM_FRAME_HOP_MS)
    if (t_start + FPS_DUR - WAVLM_FRAME_CENTER_OFFSET_MS) % WAVLM_FRAME_HOP_MS == 0:
        hi = int((t_start + FPS_DUR - WAVLM_FRAME_CENTER_OFFSET_MS) / WAVLM_FRAME_HOP_MS)
    return max(0, lo), min(n_latents, max(0, hi))


def main():
    # ── 1. Basic pooling: value-coded latents, known buckets ────────────
    n_lat = 150  # 3 s of audio
    lat = make_latents(n_lat, lambda i: float(i))
    centers = latent_frame_centers_ms(n_lat)
    ts = np.arange(0, 3000 - FPS_DUR, FPS_DUR)  # video frames fully inside audio

    pooled, counts = pool_latents_to_intervals(lat, centers, ts, FPS_DUR)
    ok = True
    for k, t in enumerate(ts):
        lo, hi = expected_bucket(t, n_lat)
        expect = np.mean(np.arange(lo, hi)) if hi > lo else np.nan
        got = pooled[k, 0]
        if hi > lo:
            ok &= np.isclose(got, expect)
            ok &= counts[k] == hi - lo
        else:
            ok &= np.isnan(got)
    check("1. pooled means match exact-arithmetic buckets on every video frame", ok)
    check("   every 33.3ms frame traps 1-2 latent frames (20ms hop)",
          set(np.unique(counts)).issubset({1, 2}))

    # ── 2. Drift-free property: 60 minutes out, bucket still exact ──────
    n_lat_long = 180_000                      # 1 hour of latents
    centers_long = latent_frame_centers_ms(n_lat_long)
    lat_long = np.zeros((n_lat_long, 1), dtype=np.float32)
    lat_long[:, 0] = np.arange(n_lat_long)
    far_ts = np.array([3_599_000.0 + k * FPS_DUR for k in range(20)])  # ~59m59s
    pooled_far, counts_far = pool_latents_to_intervals(lat_long, centers_long, far_ts, FPS_DUR)
    ok = True
    for k, t in enumerate(far_ts):
        lo, hi = expected_bucket(t, n_lat_long)
        expect = np.mean(np.arange(lo, hi)) if hi > lo else np.nan
        ok &= np.isclose(pooled_far[k, 0], expect, rtol=0, atol=1e-6)
    check("2. one hour into the recording, buckets are still exact (no accumulated drift)", ok)

    # Contrast — the rejected index-ratio approach: assume "50 latents per 30
    # video frames" and multiply. If the true camera rate is NTSC 29.97 fps
    # but the ratio assumes 30.0, the k-th video frame's latent index drifts
    # linearly with k. At the 1-hour mark that is seconds of misalignment;
    # timestamp lookup (above) is immune because each bucket is absolute.
    k_1h = int(3600 * 29.97)                      # video frame count at 1 h, true rate
    idx_ratio = int(k_1h * (50.0 / 30.0))         # index math with assumed 30 fps
    t_true_ms = k_1h / 29.97 * 1000.0             # that frame's true clock time
    lo_true, _ = expected_bucket(t_true_ms, n_lat_long)
    drift_frames = abs(idx_ratio - lo_true)
    check(f"   (contrast: index math assuming 30 fps on a 29.97 fps stream is "
          f"{drift_frames} latent frames ≈ {drift_frames * 20 / 1000:.1f}s off at 1 hour)",
          drift_frames > 100)

    # ── 3. Edge semantics ────────────────────────────────────────────────
    short_ts = np.array([0.0, 2990.0, 3500.0])  # last two beyond/straddling 3 s audio
    pooled_e, counts_e = pool_latents_to_intervals(lat, centers, short_ts, FPS_DUR)
    check("3. video frame past end of audio → NaN pooled row, count 0",
          np.isnan(pooled_e[2]).all() and counts_e[2] == 0)

    empty_pooled, empty_counts = pool_latents_to_intervals(
        np.zeros((0, H), dtype=np.float32), np.zeros(0), short_ts, FPS_DUR)
    check("   zero-latent clip (TARGET_SILENT) → all-NaN, all counts 0",
          np.isnan(empty_pooled).all() and (empty_counts == 0).all())

    # ── 4. Interval RMS on a known waveform ──────────────────────────────
    audio = np.ones(SR * 3, dtype=np.float32) * 0.25       # constant 0.25 → RMS 0.25
    audio[SR:2 * SR] = -0.25                               # sign flip, same RMS
    rms = interval_rms(audio, SR, np.array([0.0, 1500.0, 5000.0]), FPS_DUR)
    check("4. interval RMS exact on constant-amplitude signal; NaN past audio end",
          np.isclose(rms[0], 0.25) and np.isclose(rms[1], 0.25) and np.isnan(rms[2]))

    # ── 5. Cosine velocity semantics ─────────────────────────────────────
    v = np.zeros((4, H))
    v[0, 0] = 1.0          # e0
    v[1, 1] = 1.0          # e1 (orthogonal)   → vel 1.0
    v[2, 1] = 1.0          # e1 again          → vel 0.0
    v[3, :] = np.nan       # gap               → vel NaN
    vel = cosine_velocity(v)
    check("5. cosine velocity: first=NaN, orthogonal=1, identical=0, NaN-propagating",
          np.isnan(vel[0]) and np.isclose(vel[1], 1.0)
          and np.isclose(vel[2], 0.0) and np.isnan(vel[3]))

    # ── 6. Latent group reduction ────────────────────────────────────────
    g = np.arange(H, dtype=np.float64)[None, :]            # 0..63
    groups = reduce_to_latent_groups(g, LATENT_CHANNELS)   # groups of 4
    check("6. group reduction: mean of each contiguous 4-dim block",
          np.allclose(groups[0], [np.mean(np.arange(H)[i * 4:(i + 1) * 4])
                                  for i in range(LATENT_CHANNELS)]))

    # ── 7. frame_features_from_latents end-to-end ────────────────────────
    n = 150
    lat7 = make_latents(n, lambda i: 1.0 + 0.001 * i)
    centers7 = latent_frame_centers_ms(n)
    audio7 = np.ones(SR * 3, dtype=np.float32) * 0.2
    audio7[SR:SR + SR // 2] = 0.0                          # 1000-1500ms attenuated silence
    ts7 = np.arange(0, 2900, FPS_DUR)

    feats = frame_features_from_latents(lat7, centers7, audio7, SR, ts7)
    check("7. output schema == FRAME_ACOUSTIC_COLUMN_NAMES, all arrays length K",
          sorted(feats.keys()) == sorted(FRAME_ACOUSTIC_COLUMN_NAMES)
          and all(len(a) == len(ts7) for a in feats.values()))

    silent_zone = (ts7 >= 1000.0) & (ts7 + FPS_DUR <= 1500.0)
    loud_zone = ts7 + FPS_DUR <= 1000.0
    check("   silence floor: frames fully inside the attenuated zone are NaN "
          "(latents, velocity, rms)",
          np.isnan(feats["frame_wavlm_latent_0"][silent_zone]).all()
          and np.isnan(feats["frame_acoustic_energy_rms"][silent_zone]).all())
    check("   loud zone carries finite latents and rms == 0.2",
          np.isfinite(feats["frame_wavlm_latent_0"][loud_zone]).all()
          and np.allclose(feats["frame_acoustic_energy_rms"][loud_zone], 0.2))

    vel7 = feats["frame_prosodic_velocity"]
    interior_loud = loud_zone.copy()
    interior_loud[0] = False                                # first frame has no predecessor
    check("   velocity: NaN at t=0 and across the silence boundary, finite inside loud zone",
          np.isnan(vel7[0]) and np.isfinite(vel7[interior_loud]).all())

    # ── 8. Window-path parity with the original formulas ─────────────────
    rng = np.random.default_rng(7)
    latw = rng.normal(size=(100, H)).astype(np.float32)    # 2 s of latents
    centersw = latent_frame_centers_ms(100)
    audiow = rng.normal(scale=0.1, size=SR * 2).astype(np.float32)

    identity_quantize = lambda x: np.arange(len(x)) % 5    # deterministic fake codebook
    wf = window_features_from_latents(latw, centersw, audiow, SR, 0.0, 2000.0,
                                      quantize_fn=identity_quantize)

    win = latw.astype(np.float64)
    check("8. acoustic_volatility == L2(per-dim unbiased variance)",
          np.isclose(wf["acoustic_volatility"],
                     np.linalg.norm(win.var(axis=0, ddof=1))))
    normalized = win / np.linalg.norm(win, axis=1, keepdims=True)
    cos = (normalized[:-1] * normalized[1:]).sum(axis=1)
    check("   prosodic_velocity == mean consecutive cosine distance",
          np.isclose(wf["prosodic_velocity"], np.mean(1.0 - cos)))
    profile = win.mean(axis=0).reshape(LATENT_CHANNELS, -1).mean(axis=1)
    check("   16-channel latent profile matches reshape-mean",
          all(np.isclose(wf[f"wavlm_latent_{i}"], profile[i]) for i in range(LATENT_CHANNELS)))
    probs = np.bincount(identity_quantize(latw)) / 100.0
    probs = probs[probs > 0]
    check("   vocal_entropy == Shannon entropy of assignments",
          np.isclose(wf["vocal_entropy"], -np.sum(probs * np.log2(probs))))
    check("   acoustic_energy_rms == waveform RMS over the window",
          np.isclose(wf["acoustic_energy_rms"],
                     np.sqrt(np.mean(np.square(audiow.astype(np.float64))))))

    # Null semantics preserved from the original implementation:
    wf_silent = window_features_from_latents(
        latw, centersw, np.zeros(SR * 2, dtype=np.float32), SR, 0.0, 2000.0,
        quantize_fn=identity_quantize)
    wf_oob = window_features_from_latents(latw, centersw, audiow, SR, 5000.0, 7000.0,
                                          quantize_fn=identity_quantize)
    wf_thin = window_features_from_latents(latw[:1], centersw[:1], audiow, SR, 0.0, 2000.0,
                                           quantize_fn=identity_quantize)
    check("   null semantics: silence / out-of-range / <2 latent frames → all-NaN 20-col dict",
          all(np.isnan(v) for v in wf_silent.values())
          and all(np.isnan(v) for v in wf_oob.values())
          and all(np.isnan(v) for v in wf_thin.values()))

    # ── 9. Non-zero-origin timestamps (offset clocks) ────────────────────
    off_ts = np.array([500.0 + k * FPS_DUR for k in range(10)])
    pooled_off, counts_off = pool_latents_to_intervals(lat, centers, off_ts, FPS_DUR)
    ok = True
    for k, t in enumerate(off_ts):
        lo, hi = expected_bucket(t, n_lat)
        ok &= np.isclose(pooled_off[k, 0], np.mean(np.arange(lo, hi)))
    check("9. buckets exact for a clock that does not start at 0 (offset_ms rebase-safe)", ok)

    # ── 10. Non-finite poisoning containment (bugs found + fixed 2026-07-07) ─
    n_lat10 = 5000
    lat10 = np.ones((n_lat10, H), dtype=np.float64)
    lat10[2500, 0] = np.inf                                 # single poisoned latent
    centers10 = latent_frame_centers_ms(n_lat10)
    ts10 = np.arange(0, 100_000 - FPS_DUR, FPS_DUR)         # 100s of video frames
    pooled10, counts10 = pool_latents_to_intervals(lat10, centers10, ts10, FPS_DUR)

    poison_bucket = (centers10[2500] >= ts10) & (centers10[2500] < ts10 + FPS_DUR)
    poisoned_frame = np.flatnonzero(poison_bucket)
    check("10. a single non-finite latent poisons ONLY the frame(s) that trap it "
          "(not every later frame via inf-inf cumsum cancellation)",
          len(poisoned_frame) >= 1
          and np.isnan(pooled10[poisoned_frame]).all()
          and np.isfinite(pooled10[poisoned_frame[-1] + 1:]).all())
    check("    the poisoned frame's count is zeroed (reported as 'no usable latent', "
          "not a stale positive count)",
          (counts10[poisoned_frame] == 0).all())

    audio10 = np.ones(SR * 100, dtype=np.float32) * 0.2
    audio10[SR] = np.inf                                     # single poisoned sample at t=1000ms
    rms10 = interval_rms(audio10, SR, ts10, FPS_DUR)
    poison_frame_audio = np.flatnonzero(np.isnan(rms10))
    check("    same containment for interval_rms: EXACTLY ONE frame goes NaN "
          "(the one trapping the poisoned sample) — not every later window",
          len(poison_frame_audio) == 1
          and np.isfinite(rms10[:poison_frame_audio[0]]).all()
          and np.isfinite(rms10[poison_frame_audio[0] + 1:]).all())

    # ── 11. frame_features_from_latents nulls rms when zero latents trapped ─
    empty_lat = np.zeros((0, H), dtype=np.float32)
    ts11 = np.array([0.0, FPS_DUR, 2 * FPS_DUR])
    audio11 = np.array([0.0] * int(SR * FPS_DUR / 1000) +
                       [0.5] * int(SR * FPS_DUR / 1000) +
                       [0.0] * int(SR * FPS_DUR / 1000), dtype=np.float32)
    feats11 = frame_features_from_latents(empty_lat, np.zeros(0), audio11, SR, ts11)
    check("11. zero-latent clip: frame_acoustic_energy_rms is nulled alongside the "
          "latent columns (not left as a stray non-NaN value) — the 'NaN across "
          "the board' contract holds even when only counts==0, not the RMS floor, "
          "triggers it",
          np.isnan(feats11["frame_acoustic_energy_rms"]).all()
          and np.isnan(feats11["frame_wavlm_latent_0"]).all())

    # ── 12. Zero-norm-row cosine parity with the original F.normalize formula ─
    v12 = np.zeros((3, H))
    v12[0, 0] = 1.0
    v12[1, :] = 0.0                                          # exact zero-norm row
    v12[2, 0] = 1.0
    vel12 = cosine_velocity(v12)
    check("12. cosine_velocity: a zero-norm row degrades to distance 1.0 "
          "(F.normalize-eps parity), not NaN",
          np.isclose(vel12[1], 1.0) and np.isclose(vel12[2], 1.0))

    identity_quantize12 = lambda x: np.arange(len(x)) % 3
    lat12 = np.zeros((10, H), dtype=np.float64)
    lat12[:, 0] = 1.0
    lat12[5, :] = 0.0                                        # zero-norm row mid-window
    centers12 = latent_frame_centers_ms(10)
    audio12 = np.ones(SR * 2, dtype=np.float32) * 0.1
    wf12 = window_features_from_latents(lat12, centers12, audio12, SR, 0.0, 2000.0,
                                        quantize_fn=identity_quantize12)
    check("    window_features_from_latents' prosodic_velocity carries the same "
          "eps-clamped parity (finite, not NaN-contaminated by the zero row)",
          np.isfinite(wf12["prosodic_velocity"]))

    # ── 13. pool_latents_to_intervals empty-latents preserves [K, H] shape ──
    empty_wide = np.zeros((0, 1024), dtype=np.float32)
    pooled13, counts13 = pool_latents_to_intervals(empty_wide, np.zeros(0),
                                                    np.array([0.0, 100.0, 200.0]), FPS_DUR)
    check("13. empty latents of hidden_size=1024 pool to shape (K, 1024), not (K, 1) "
          "— the documented [K, H] contract holds even for T==0",
          pooled13.shape == (3, 1024) and np.isnan(pooled13).all())

    # ── 14. Chunk-seam alignment guard (acoustic_extractor.validate_chunk_alignment) ─
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audio_isolation.core.acoustic_extractor import (  # noqa: E402
        validate_chunk_alignment, WAVLM_CHUNK_SECONDS, WAVLM_CHUNK_OVERLAP_SECONDS,
    )
    raised_chunk = raised_overlap = False
    try:
        validate_chunk_alignment(WAVLM_CHUNK_SECONDS, WAVLM_CHUNK_OVERLAP_SECONDS)
        defaults_ok = True
    except ValueError:
        defaults_ok = False
    try:
        validate_chunk_alignment(3.33, WAVLM_CHUNK_OVERLAP_SECONDS)  # not a 20ms multiple
    except ValueError:
        raised_chunk = True
    try:
        validate_chunk_alignment(WAVLM_CHUNK_SECONDS, 0.95)          # not a 20ms multiple
    except ValueError:
        raised_overlap = True
    check("14. production defaults (30s chunk, 1s overlap) pass the alignment guard",
          defaults_ok)
    check("    a non-20ms-multiple chunk_seconds is rejected (fail-fast, not silent "
          "seam corruption)", raised_chunk)
    check("    a non-20ms-multiple chunk_overlap_seconds is rejected", raised_overlap)

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"VERIFICATION RESULTS: {PASS}/{PASS + FAIL} checks passed")
    if FAIL == 0:
        print("🏆 SUCCESS — frame-level WavLM alignment mathematically verified.")
        return 0
    print("❌ FAILURES PRESENT — do not wire further until resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
