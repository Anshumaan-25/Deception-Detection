import numpy as np
import torch
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2Processor
from scipy.io import wavfile
from sklearn.cluster import MiniBatchKMeans
import logging


# --- Constants ---
HUBERT_SAMPLE_RATE = 16000
HUBERT_LAYER_INDEX = 7
LATENT_CHANNELS = 16
CODEBOOK_SIZE = 64
SILENCE_RMS_FLOOR = 0.005  # Below this RMS, the chunk is diarizer-attenuated silence

# The 20 acoustic column names (canonical schema contract)
ACOUSTIC_COLUMN_NAMES = (
    ["acoustic_volatility", "prosodic_velocity"]
    + [f"hubert_latent_{i}" for i in range(LATENT_CHANNELS)]
    + ["vocal_entropy", "acoustic_energy_rms"]
)


class HuBERTAcousticExtractor:
    """
    Decoupled Acoustic Microservice — Layer 5 Audio Feature Extraction.

    Loads facebook/hubert-base-ls960 onto CUDA, ingests the isolated target
    audio WAV, and provides per-window (2-second) paralinguistic feature
    extraction from Layer 7 hidden states.

    This module is designed as a self-contained Domain-Driven service within
    the audio_isolation domain. It has zero coupling to the visual pipeline.
    """

    def __init__(self, isolated_wav_path: str):
        """
        Boot the HuBERT inference engine and pre-load the full isolated WAV.

        Args:
            isolated_wav_path: Path to the diarizer-isolated target audio WAV.
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("HuBERT_Acoustic_Extractor")

        self.device = torch.device("cuda")
        self.logger.info("🧠 Loading HuBERT-base-ls960 onto CUDA...")

        # Load processor and model with hidden state capture enabled
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            output_hidden_states=True
        ).to(self.device).eval()

        self.logger.info("✅ HuBERT model loaded and locked to eval() mode on CUDA.")

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
        if sample_rate != HUBERT_SAMPLE_RATE:
            self.logger.info(f"Resampling from {sample_rate}Hz to {HUBERT_SAMPLE_RATE}Hz...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=HUBERT_SAMPLE_RATE
            )
            audio_tensor = torch.from_numpy(audio_signal).unsqueeze(0)
            audio_tensor = resampler(audio_tensor)
            audio_signal = audio_tensor.squeeze(0).numpy()

        self.audio_signal = audio_signal
        self.sample_rate = HUBERT_SAMPLE_RATE
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
        Layer 7 frames from the full audio. This avoids re-fitting per window
        and provides a stable quantization vocabulary.
        """
        self.logger.info("📊 Fitting KMeans codebook (K=64) on Layer 7 sample...")

        # Sample up to 10 seconds of audio for codebook fitting
        max_samples = min(len(self.audio_signal), HUBERT_SAMPLE_RATE * 10)
        sample_chunk = self.audio_signal[:max_samples]

        # Run forward pass on the sample
        inputs = self.processor(
            sample_chunk, sampling_rate=HUBERT_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        layer7_sample = outputs.hidden_states[HUBERT_LAYER_INDEX].squeeze(0).cpu().numpy()

        # Fit codebook
        self.codebook = MiniBatchKMeans(
            n_clusters=CODEBOOK_SIZE,
            batch_size=256,
            n_init=3,
            random_state=42
        )
        self.codebook.fit(layer7_sample)
        self.logger.info(f"✅ Codebook fitted on {layer7_sample.shape[0]} temporal frames.")

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

        # --- HuBERT forward pass ---
        inputs = self.processor(
            raw_chunk, sampling_rate=HUBERT_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        # Isolate Layer 7 hidden states: [1, T, 768] → [T, 768]
        layer7 = outputs.hidden_states[HUBERT_LAYER_INDEX].squeeze(0)

        # Temporal frame count T (approximately 99 for 2-second chunks)
        T = layer7.shape[0]

        if T < 2:
            return null_result

        # ================================================================
        # FEATURE 1: Acoustic Volatility
        # L2 norm of the temporal variance vector across 768 dimensions
        # ================================================================
        temporal_var = layer7.var(dim=0)  # → [768]
        acoustic_volatility = torch.linalg.norm(temporal_var).item()

        # ================================================================
        # FEATURE 2: Prosodic Velocity
        # Mean frame-to-frame cosine distance across temporal embeddings
        # ================================================================
        normalized = F.normalize(layer7, dim=1)  # → [T, 768]
        cosine_sim = (normalized[:-1] * normalized[1:]).sum(dim=1)  # → [T-1]
        prosodic_velocity = (1.0 - cosine_sim).mean().item()

        # ================================================================
        # FEATURES 3-18: Latent Profile (16 channels)
        # Temporal mean → reshape 768 into 16 groups of 48 → mean each
        # ================================================================
        temporal_mean = layer7.mean(dim=0)  # → [768]
        latent_profile = temporal_mean.reshape(LATENT_CHANNELS, 48).mean(dim=1)  # → [16]
        latent_values = latent_profile.cpu().numpy()

        # ================================================================
        # FEATURE 19: Vocal Entropy
        # Shannon entropy over KMeans-quantized hidden state frame assignments
        # ================================================================
        layer7_np = layer7.cpu().numpy()
        assignments = self.codebook.predict(layer7_np)
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
            result[f"hubert_latent_{i}"] = float(latent_values[i])

        result["vocal_entropy"] = vocal_entropy
        result["acoustic_energy_rms"] = acoustic_energy_rms

        return result
