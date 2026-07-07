import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from scipy.io import wavfile
from sklearn.cluster import MiniBatchKMeans
import logging

from audio_isolation.core.frame_alignment import (
    LATENT_CHANNELS,
    SILENCE_RMS_FLOOR,
    WAVLM_FRAME_HOP_MS,
    FRAME_ACOUSTIC_COLUMN_NAMES,
    VIDEO_FRAME_DURATION_MS,
    latent_frame_centers_ms,
    frame_features_from_latents,
    window_features_from_latents,
)

# --- Constants ---
WAVLM_MODEL_NAME = "microsoft/wavlm-large"
WAVLM_SAMPLE_RATE = 16000
# wavlm-large has 24 transformer layers (hidden_states indices 0..24, where 0
# is the embedding output). HuBERT-base's layer index 7 was chosen at ~58% of
# its 12-layer stack; 24 * 7/12 = 14 lands at the same proportional depth.
# This is a reasoned placeholder, not an empirically re-tuned value — dry
# runs on real footage are on hold for now (see
# RECORDING_TIMELINE_AND_ACOUSTIC_UPGRADE_PLAN.md §4.2), so re-validate this
# index once real-audio validation resumes.
WAVLM_LAYER_INDEX = 14
CODEBOOK_SIZE = 64
CODEBOOK_FIT_FRAMES = 512  # evenly-spaced latent frames sampled for KMeans

# WavLM/Wav2Vec2 conv feature-encoder total stride @ 16kHz (20ms/frame hop —
# derived from the shared frame_alignment constant, not duplicated as a
# magic number). Chunk boundaries must land on multiples of this or the
# chunk-seam stitch in _compute_full_latents can silently corrupt the grid.
_WAVLM_STRIDE_SAMPLES = int(round(WAVLM_FRAME_HOP_MS / 1000.0 * WAVLM_SAMPLE_RATE))

# The 20 acoustic column names (canonical schema contract)
ACOUSTIC_COLUMN_NAMES = (
    ["acoustic_volatility", "prosodic_velocity"]
    + [f"wavlm_latent_{i}" for i in range(LATENT_CHANNELS)]
    + ["vocal_entropy", "acoustic_energy_rms"]
)

# ── Production hardware profile ─────────────────────────────────────────────
# Tuned for the DIPR work desktop: RTX 6000 Ada (48 GB VRAM), 44-core CPU,
# 512 GB RAM, Ubuntu, R580 driver / CUDA 12.x (torch cu121 wheels run fine on
# R580 — drivers are backward-compatible with older CUDA runtimes).
#   CHUNK 30 s  → 1500 latent frames/chunk: transformer attention is O(T²),
#                 1500² is cheap; full-clip single-window attention would not be.
#   OVERLAP 1 s → context halo so chunk-edge frames see real neighbors; halo
#                 frames are trimmed on stitching (each kept frame appears once).
#   BATCH 4     → 4×32 s per forward keeps Ada tensor cores fed; peak activation
#                 memory stays well under 2 GB of the 48 GB budget.
#   AMP fp16    → Ada tensor-core path, ~2× throughput; latents are cast back
#                 to float32 before caching. Set False for strict-fp32 parity runs.
#   TRUNCATE    → wavlm-large uses the StableLayerNorm encoder, which applies
#                 one unconditional final LayerNorm after its layer loop and
#                 bakes it into whichever hidden_states entry ends up last.
#                 Naively dropping layers above WAVLM_LAYER_INDEX would land
#                 that extra norm exactly on the entry we read — a real,
#                 systematic corruption, not fp16 noise (found + verified by
#                 review 2026-07-07). The fix: keep WAVLM_LAYER_INDEX + 1
#                 layers (one throwaway extra) so the final norm lands on the
#                 entry AFTER the one we read; hidden_states[WAVLM_LAYER_INDEX]
#                 is then the same raw (unnormed) tensor the full 24-layer
#                 stack would produce at that index. Still tracks
#                 WAVLM_LAYER_INDEX automatically if the layer is re-tuned —
#                 see _load_wavlm.
WAVLM_CHUNK_SECONDS = 30.0
WAVLM_CHUNK_OVERLAP_SECONDS = 1.0
WAVLM_CHUNK_BATCH = 4
WAVLM_USE_AMP = True
WAVLM_TRUNCATE_ENCODER = True

# Process-wide model singleton: one recording = N clips = N extractor
# instances in one GPU worker process; the frozen weights are shared instead
# of re-read from disk per clip. Keyed on (model, truncation cut).
_MODEL_CACHE = {}


def validate_chunk_alignment(chunk_seconds: float, chunk_overlap_seconds: float) -> None:
    """
    Raise ValueError unless both chunk boundaries land on the 20ms WavLM
    latent-frame grid (see WavLMAcousticExtractor.__init__ chunk-seam
    doctrine comment). Pure function — no model/audio load — so it's
    unit-testable without GPU or network (tests/verify_frame_acoustics.py).
    """
    chunk_samples = int(chunk_seconds * WAVLM_SAMPLE_RATE)
    overlap_samples = int(chunk_overlap_seconds * WAVLM_SAMPLE_RATE)
    grid_seconds = _WAVLM_STRIDE_SAMPLES / WAVLM_SAMPLE_RATE
    if chunk_samples % _WAVLM_STRIDE_SAMPLES != 0:
        raise ValueError(
            f"chunk_seconds={chunk_seconds} → {chunk_samples} samples is not a "
            f"multiple of the {_WAVLM_STRIDE_SAMPLES}-sample ({WAVLM_FRAME_HOP_MS:.0f}ms) "
            f"WavLM conv stride; chunk seams would silently gap or double-count "
            f"latent frames. Pick a chunk_seconds that is a multiple of {grid_seconds:.3f}s."
        )
    if overlap_samples % _WAVLM_STRIDE_SAMPLES != 0:
        raise ValueError(
            f"chunk_overlap_seconds={chunk_overlap_seconds} → {overlap_samples} samples "
            f"is not a multiple of the {_WAVLM_STRIDE_SAMPLES}-sample "
            f"({WAVLM_FRAME_HOP_MS:.0f}ms) WavLM conv stride; chunk seams would silently "
            f"gap or double-count latent frames. Pick a chunk_overlap_seconds that is a "
            f"multiple of {grid_seconds:.3f}s."
        )


def _load_wavlm(device: torch.device, truncate_encoder: bool):
    """
    Load WavLM, optionally truncating unused encoder layers for speed.

    StableLayerNorm encoder mechanics (WavLMEncoderStableLayerNorm.forward,
    wavlm-large has do_stable_layer_norm=True): each loop iteration appends
    the PRE-layer hidden state as the next hidden_states entry, THEN — after
    the whole loop — applies one unconditional `self.layer_norm` and appends
    ITS output as one final entry. So with L layers kept, hidden_states has
    L+1 entries: 0..L-1 are raw (pre-layer) states, entry L is normed.

    hidden_states[WAVLM_LAYER_INDEX] in the FULL 24-layer stack is the raw
    (unnormed) output of layer WAVLM_LAYER_INDEX-1 — entry WAVLM_LAYER_INDEX
    is never the last entry there (last is 24), so it never receives the
    final norm. To read the SAME raw tensor from a truncated stack, entry
    WAVLM_LAYER_INDEX must likewise not be the truncated stack's last entry —
    which requires keeping WAVLM_LAYER_INDEX + 1 layers (last entry becomes
    WAVLM_LAYER_INDEX + 1, one past the index we actually read).
    Naively truncating to exactly WAVLM_LAYER_INDEX layers makes
    hidden_states[WAVLM_LAYER_INDEX] the LAST entry — silently applying an
    extra LayerNorm the full-stack computation never applies at that index.
    """
    key = (WAVLM_MODEL_NAME, WAVLM_LAYER_INDEX if truncate_encoder else -1, str(device))
    if key not in _MODEL_CACHE:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        model = WavLMModel.from_pretrained(
            WAVLM_MODEL_NAME, output_hidden_states=True
        ).to(device).eval()
        if truncate_encoder:
            model.encoder.layers = model.encoder.layers[:WAVLM_LAYER_INDEX + 1]
        _MODEL_CACHE[key] = (feature_extractor, model)
    return _MODEL_CACHE[key]


class WavLMAcousticExtractor:
    """
    Decoupled Acoustic Microservice — Layer 5 Audio Feature Extraction.

    Single-pass architecture: the full isolated-target WAV is pushed through
    WavLM once (chunked + batched, fp16 autocast on Ada), and the layer-14
    latent sequence is cached in RAM (float32, ~750 MB/hour — trivial against
    the 512 GB desktop budget). Both consumers then read the cache:

      - extract_window_features(start_ms, end_ms): the canonical 20-column
        window block consumed by DynamicWindowEngine (unchanged contract).
        Previously this ran a dedicated 2 s WavLM forward per window — with
        the 1 s stride every second of audio crossed the transformer twice;
        now it is a numpy slice of the cache.
      - frame_features_for_timestamps(ts): the frame-level acoustic block
        aligned to the 30 fps master clock (Phase 2 injection; ST-GAE input),
        drift-free by absolute-timestamp bucketing (frame_alignment.py).

    Context-semantics note: latents now carry ~30 s of transformer context
    instead of an isolated 2 s window, removing the per-window boundary
    effect. No real-audio feature values were ever validated against the old
    path (real-footage validation is still pending), so this is a semantics
    improvement, not a regression risk against golden data.
    """

    def __init__(self, isolated_wav_path: str, *,
                 device: str = "cuda",
                 use_amp: bool = WAVLM_USE_AMP,
                 truncate_encoder: bool = WAVLM_TRUNCATE_ENCODER,
                 chunk_seconds: float = WAVLM_CHUNK_SECONDS,
                 chunk_overlap_seconds: float = WAVLM_CHUNK_OVERLAP_SECONDS,
                 chunk_batch: int = WAVLM_CHUNK_BATCH):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("WavLM_Acoustic_Extractor")

        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.chunk_samples = int(chunk_seconds * WAVLM_SAMPLE_RATE)
        self.overlap_samples = int(chunk_overlap_seconds * WAVLM_SAMPLE_RATE)
        self.chunk_batch = max(1, int(chunk_batch))

        # Chunk-seam alignment guard: _compute_full_latents' halo-trim keep
        # mask only stitches to a gap-free, non-duplicated 20ms latent grid
        # when chunk/overlap boundaries land on the grid — a value that
        # doesn't divide evenly desyncs every chunk seam in the recording
        # with no error anywhere downstream (found + verified by review
        # 2026-07-07). Fail fast here instead.
        validate_chunk_alignment(chunk_seconds, chunk_overlap_seconds)

        self.logger.info(f"🧠 Loading {WAVLM_MODEL_NAME} onto {self.device} "
                         f"(truncate@{WAVLM_LAYER_INDEX}={truncate_encoder}, amp={self.use_amp})...")
        self.feature_extractor, self.model = _load_wavlm(self.device, truncate_encoder)

        # Hidden size varies by checkpoint (768 for base/base-plus, 1024 for
        # large) — the latent-profile reshape must track it.
        self.hidden_size = self.model.config.hidden_size
        if self.hidden_size % LATENT_CHANNELS != 0:
            raise ValueError(
                f"{WAVLM_MODEL_NAME} hidden_size={self.hidden_size} is not divisible by "
                f"LATENT_CHANNELS={LATENT_CHANNELS}; the latent-profile reshape requires it."
            )

        # ── Load and prepare the full audio signal ──────────────────────
        sample_rate, audio_signal = wavfile.read(isolated_wav_path)

        if audio_signal.ndim > 1:  # enforce mono
            audio_signal = audio_signal[:, 0]

        if audio_signal.dtype == np.int16:
            audio_signal = audio_signal.astype(np.float32) / 32768.0
        elif audio_signal.dtype == np.int32:
            audio_signal = audio_signal.astype(np.float32) / 2147483648.0
        else:
            audio_signal = audio_signal.astype(np.float32)

        if sample_rate != WAVLM_SAMPLE_RATE:
            self.logger.info(f"Resampling from {sample_rate}Hz to {WAVLM_SAMPLE_RATE}Hz...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=WAVLM_SAMPLE_RATE
            )
            audio_signal = resampler(torch.from_numpy(audio_signal).unsqueeze(0)).squeeze(0).numpy()

        self.audio_signal = audio_signal
        self.sample_rate = WAVLM_SAMPLE_RATE
        self.total_duration_ms = (len(self.audio_signal) / self.sample_rate) * 1000.0

        self.logger.info(
            f"📼 Audio loaded: {len(self.audio_signal)} samples "
            f"({self.total_duration_ms:.0f}ms @ {self.sample_rate}Hz)"
        )

        # ── Single full-clip pass → latent cache → codebook ─────────────
        self.latents, self.latent_centers_ms = self._compute_full_latents()
        self.latent_frame_count = int(self.latents.shape[0])
        self._fit_codebook()

    # ------------------------------------------------------------------
    # Full-clip latent computation
    # ------------------------------------------------------------------

    def _normalized_signal(self) -> np.ndarray:
        """Whole-clip zero-mean/unit-variance normalization (the checkpoint's
        Wav2Vec2FeatureExtractor semantics, applied globally so chunk
        boundaries share one scaling instead of per-chunk statistics)."""
        if not getattr(self.feature_extractor, "do_normalize", True):
            return self.audio_signal
        x = self.audio_signal.astype(np.float64)
        return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)

    def _compute_full_latents(self):
        """Chunked, batched, AMP forward over the whole clip. Returns
        (latents [T, H] float32, centers_ms [T] float64). Overlap halo frames
        are trimmed by absolute center time so every kept frame appears
        exactly once, on a uniform 20 ms grid."""
        L = len(self.audio_signal)
        empty = (np.zeros((0, self.hidden_size), dtype=np.float32),
                 np.zeros(0, dtype=np.float64))

        # Conv encoder receptive field is 400 samples — anything shorter
        # yields zero latent frames.
        if L < 400:
            self.logger.warning("Audio shorter than one WavLM receptive field — no latents.")
            return empty

        # Whole-clip silence fast-path (TARGET_SILENT clips): the isolation
        # stage attenuated everything, features would be nulled anyway.
        global_rms = float(np.sqrt(np.mean(np.square(self.audio_signal, dtype=np.float64))))
        if global_rms < SILENCE_RMS_FLOOR:
            self.logger.warning("Whole clip below silence floor — skipping WavLM pass.")
            return empty

        signal = self._normalized_signal()

        # Build the chunk plan: core [s, e) + context halo, trim by center.
        plan = []  # (order, padded_start, padded_end, core_start_ms, core_end_ms)
        for order, s in enumerate(range(0, L, self.chunk_samples)):
            e = min(s + self.chunk_samples, L)
            ps = max(0, s - self.overlap_samples)
            pe = min(L, e + self.overlap_samples)
            plan.append((order, ps, pe,
                         s / self.sample_rate * 1000.0,
                         e / self.sample_rate * 1000.0))

        # Group equal-length chunks so they batch without padding.
        by_len = {}
        for item in plan:
            by_len.setdefault(item[2] - item[1], []).append(item)

        results = {}
        with torch.inference_mode():
            for _length, items in by_len.items():
                for i in range(0, len(items), self.chunk_batch):
                    batch = items[i:i + self.chunk_batch]
                    x = torch.from_numpy(
                        np.stack([signal[ps:pe] for (_o, ps, pe, _s, _e) in batch])
                    ).to(self.device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                        out = self.model(input_values=x, output_hidden_states=True)
                    layer = out.hidden_states[WAVLM_LAYER_INDEX].float().cpu().numpy()
                    for (order, ps, _pe, s_ms, e_ms), lat in zip(batch, layer):
                        centers = latent_frame_centers_ms(
                            lat.shape[0], start_offset_ms=ps / self.sample_rate * 1000.0)
                        keep = (centers >= s_ms) & (centers < e_ms)
                        results[order] = (lat[keep].astype(np.float32), centers[keep])

        parts = [results[k] for k in sorted(results)]
        latents = np.concatenate([p[0] for p in parts], axis=0)
        centers = np.concatenate([p[1] for p in parts], axis=0)

        self.logger.info(
            f"✅ Single-pass latents cached: {latents.shape[0]} frames × {latents.shape[1]} dims "
            f"({latents.nbytes / 1e6:.0f} MB) from {len(plan)} chunk(s)."
        )
        return latents, centers

    # ------------------------------------------------------------------
    # Codebook (vocal entropy vocabulary)
    # ------------------------------------------------------------------

    def _fit_codebook(self):
        """Fit the MiniBatchKMeans codebook on up to CODEBOOK_FIT_FRAMES
        latent frames sampled evenly across the WHOLE clip (the old path used
        only the first 10 s; stratified sampling costs nothing now that the
        latents are cached)."""
        T = self.latents.shape[0]
        if T == 0:
            self.codebook = None
            self.logger.warning("No latents — vocal_entropy will be NaN everywhere.")
            return

        idx = np.linspace(0, T - 1, num=min(T, CODEBOOK_FIT_FRAMES)).astype(np.int64)
        sample = self.latents[idx]
        n_clusters = min(CODEBOOK_SIZE, sample.shape[0])

        self.codebook = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=256,
            n_init=3,
            random_state=42,
        )
        self.codebook.fit(sample)
        self.logger.info(f"✅ Codebook fitted: K={n_clusters} on {sample.shape[0]} "
                         f"stratified frames (of {T}).")

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------

    def extract_window_features(self, start_ms: float, end_ms: float) -> dict:
        """The canonical 20-column window block (DynamicWindowEngine contract,
        unchanged). Cache slice — no GPU work."""
        if self.latents.shape[0] == 0 or self.codebook is None:
            return {col: np.nan for col in ACOUSTIC_COLUMN_NAMES}
        return window_features_from_latents(
            self.latents, self.latent_centers_ms,
            self.audio_signal, self.sample_rate,
            start_ms, end_ms,
            quantize_fn=self.codebook.predict,
        )

    def frame_features_for_timestamps(self, timestamps_ms,
                                      frame_duration_ms: float = VIDEO_FRAME_DURATION_MS) -> dict:
        """Frame-level acoustic block aligned to the 30 fps master clock:
        {column: np.ndarray} over FRAME_ACOUSTIC_COLUMN_NAMES, one value per
        video frame timestamp. Drift-free absolute-timestamp bucketing; NaN
        where no latent lands or the waveform is attenuated silence."""
        return frame_features_from_latents(
            self.latents, self.latent_centers_ms,
            self.audio_signal, self.sample_rate,
            np.asarray(timestamps_ms, dtype=np.float64),
            frame_duration_ms=frame_duration_ms,
        )
