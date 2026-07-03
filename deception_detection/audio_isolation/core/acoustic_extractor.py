import numpy as np
import torch
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from scipy.io import wavfile
from sklearn.cluster import MiniBatchKMeans
import logging


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
LATENT_CHANNELS = 16
CODEBOOK_SIZE = 64
SILENCE_RMS_FLOOR = 0.005  # Below this RMS, the chunk is diarizer-attenuated silence

# The 20 acoustic column names (canonical schema contract)
ACOUSTIC_COLUMN_NAMES = (
    ["acoustic_volatility", "prosodic_velocity"]
    + [f"wavlm_latent_{i}" for i in range(LATENT_CHANNELS)]
    + ["vocal_entropy", "acoustic_energy_rms"]
)


class WavLMAcousticExtractor:
    """
    Decoupled Acoustic Microservice — Layer 5 Audio Feature Extraction.

    Loads microsoft/wavlm-large onto CUDA, ingests the isolated target
    audio WAV, and provides per-window (2-second) paralinguistic feature
    extraction from a fixed transformer-layer's hidden states
    (WAVLM_LAYER_INDEX).

    This module is designed as a self-contained Domain-Driven service within
    the audio_isolation domain. It has zero coupling to the visual pipeline.
    """

    def __init__(self, isolated_wav_path: str):
        """
        Boot the WavLM inference engine and pre-load the full isolated WAV.

        Args:
            isolated_wav_path: Path to the diarizer-isolated target audio WAV.
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("WavLM_Acoustic_Extractor")

        self.device = torch.device("cuda")
        self.logger.info(f"🧠 Loading {WAVLM_MODEL_NAME} onto CUDA...")

        # Load feature extractor (no tokenizer needed — this checkpoint is
        # self-supervised, not an ASR fine-tune) and model with hidden state
        # capture enabled.
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        self.model = WavLMModel.from_pretrained(
            WAVLM_MODEL_NAME,
            output_hidden_states=True
        ).to(self.device).eval()

        # Hidden size varies by checkpoint (768 for base/base-plus, 1024 for
        # large) — the latent-profile reshape below must track it rather than
        # assume HuBERT-base's 768.
        self.hidden_size = self.model.config.hidden_size
        if self.hidden_size % LATENT_CHANNELS != 0:
            raise ValueError(
                f"{WAVLM_MODEL_NAME} hidden_size={self.hidden_size} is not divisible by "
                f"LATENT_CHANNELS={LATENT_CHANNELS}; the latent-profile reshape requires it."
            )
        self._latent_group_size = self.hidden_size // LATENT_CHANNELS

        self.logger.info("✅ WavLM model loaded and locked to eval() mode on CUDA.")

        # Load and prepare the full audio signal
        sample_rate, audio_signal = wavfile.read(isolated_wav_path)

        # Enforce mono
        if audio_signal.ndim > 1:
            audio_signal = audio_signal[:, 0]

        # Normalize int16 to float32 [-1.0, 1.0]
        if audio_signal.dtype == np.int16:
            audio_signal = audio_signal.astype(np.float32) / 32768.0
        elif audio_signal.dtype == np.int32:
            audio_signal = audio_signal.astype(np.float32) / 2147483648.0
        else:
            audio_signal = audio_signal.astype(np.float32)

        # Resample to 16kHz if necessary
        if sample_rate != WAVLM_SAMPLE_RATE:
            self.logger.info(f"Resampling from {sample_rate}Hz to {WAVLM_SAMPLE_RATE}Hz...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=WAVLM_SAMPLE_RATE
            )
            audio_tensor = torch.from_numpy(audio_signal).unsqueeze(0)
            audio_tensor = resampler(audio_tensor)
            audio_signal = audio_tensor.squeeze(0).numpy()

        self.audio_signal = audio_signal
        self.sample_rate = WAVLM_SAMPLE_RATE
        self.total_duration_ms = (len(self.audio_signal) / self.sample_rate) * 1000.0

        self.logger.info(
            f"📼 Audio loaded: {len(self.audio_signal)} samples "
            f"({self.total_duration_ms:.0f}ms @ {self.sample_rate}Hz)"
        )

        # Pre-fit the KMeans codebook for vocal entropy quantization
        self._fit_codebook()

    def _fit_codebook(self):
        """
        Pre-fit a MiniBatchKMeans codebook (K=64) on a stratified sample of
        WAVLM_LAYER_INDEX frames from the full audio. This avoids re-fitting
        per window and provides a stable quantization vocabulary.
        """
        self.logger.info(f"📊 Fitting KMeans codebook (K=64) on layer {WAVLM_LAYER_INDEX} sample...")

        # Sample up to 10 seconds of audio for codebook fitting
        max_samples = min(len(self.audio_signal), WAVLM_SAMPLE_RATE * 10)
        sample_chunk = self.audio_signal[:max_samples]

        # Run forward pass on the sample
        inputs = self.feature_extractor(
            sample_chunk, sampling_rate=WAVLM_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        layer_sample = outputs.hidden_states[WAVLM_LAYER_INDEX].squeeze(0).cpu().numpy()

        # Fit codebook
        self.codebook = MiniBatchKMeans(
            n_clusters=CODEBOOK_SIZE,
            batch_size=256,
            n_init=3,
            random_state=42
        )
        self.codebook.fit(layer_sample)
        self.logger.info(f"✅ Codebook fitted on {layer_sample.shape[0]} temporal frames.")

    def extract_window_features(self, start_ms: float, end_ms: float) -> dict:
        """
        Extract 20 paralinguistic features from a single temporal window.

        Args:
            start_ms: Window start time in milliseconds.
            end_ms: Window end time in milliseconds.

        Returns:
            Dictionary with 20 acoustic feature columns. All values are np.nan
            if the window falls outside the audio range or contains only
            diarizer-attenuated silence.
        """
        null_result = {col: np.nan for col in ACOUSTIC_COLUMN_NAMES}

        # --- Boundary guard ---
        start_sample = int((start_ms / 1000.0) * self.sample_rate)
        end_sample = int((end_ms / 1000.0) * self.sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_signal), end_sample)

        if end_sample <= start_sample:
            return null_result

        raw_chunk = self.audio_signal[start_sample:end_sample]

        # --- Silence tripwire: detect diarizer-attenuated segments ---
        rms_energy = float(np.sqrt(np.mean(raw_chunk ** 2)))
        if rms_energy < SILENCE_RMS_FLOOR:
            return null_result

        # --- WavLM forward pass ---
        inputs = self.feature_extractor(
            raw_chunk, sampling_rate=WAVLM_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        # Isolate the target layer's hidden states: [1, T, hidden_size] → [T, hidden_size]
        layer = outputs.hidden_states[WAVLM_LAYER_INDEX].squeeze(0)

        # Temporal frame count T (approximately 99 for 2-second chunks)
        T = layer.shape[0]

        if T < 2:
            return null_result

        # ================================================================
        # FEATURE 1: Acoustic Volatility
        # L2 norm of the temporal variance vector across hidden_size dims
        # ================================================================
        temporal_var = layer.var(dim=0)  # → [hidden_size]
        acoustic_volatility = torch.linalg.norm(temporal_var).item()

        # ================================================================
        # FEATURE 2: Prosodic Velocity
        # Mean frame-to-frame cosine distance across temporal embeddings
        # ================================================================
        normalized = F.normalize(layer, dim=1)  # → [T, hidden_size]
        cosine_sim = (normalized[:-1] * normalized[1:]).sum(dim=1)  # → [T-1]
        prosodic_velocity = (1.0 - cosine_sim).mean().item()

        # ================================================================
        # FEATURES 3-18: Latent Profile (16 channels)
        # Temporal mean → reshape hidden_size into 16 groups → mean each
        # ================================================================
        temporal_mean = layer.mean(dim=0)  # → [hidden_size]
        latent_profile = temporal_mean.reshape(LATENT_CHANNELS, self._latent_group_size).mean(dim=1)  # → [16]
        latent_values = latent_profile.cpu().numpy()

        # ================================================================
        # FEATURE 19: Vocal Entropy
        # Shannon entropy over KMeans-quantized hidden state frame assignments
        # ================================================================
        layer_np = layer.cpu().numpy()
        assignments = self.codebook.predict(layer_np)
        _, counts = np.unique(assignments, return_counts=True)
        probs = counts / len(assignments)
        vocal_entropy = float(-np.sum(probs * np.log2(probs)))

        # ================================================================
        # FEATURE 20: Acoustic Energy RMS
        # Root Mean Square of the raw waveform chunk
        # ================================================================
        acoustic_energy_rms = rms_energy  # Already computed above

        # --- Assemble the 20-column output dictionary ---
        result = {
            "acoustic_volatility": acoustic_volatility,
            "prosodic_velocity": prosodic_velocity,
        }
        for i in range(LATENT_CHANNELS):
            result[f"wavlm_latent_{i}"] = float(latent_values[i])

        result["vocal_entropy"] = vocal_entropy
        result["acoustic_energy_rms"] = acoustic_energy_rms

        return result
