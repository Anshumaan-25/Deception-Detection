# ============================================================================
# ARCHIVED — SUPERSEDED (2026-07-06). DO NOT WIRE THIS IN.
# ============================================================================
# This file held the Tensor Fusion Network (TFN) supervised classifier — the
# old "Temporal Sequence Model -> Classification Head -> verdict" end-stage
# from the original prototype design. That approach has been superseded:
#
#   The pipeline's end-stage is attribution, not classification. Current
#   deliverable: the recording-level deviation report produced by
#   BaselineCalibrator.fit/apply + assemble_recording (baseline-clip
#   calibration). Planned future end-stage: an ST-GAE (spatio-temporal graph
#   autoencoder) fitted per subject on the baseline clip, emitting a Temporal
#   Anomaly Attribution Report — never a truth/lie verdict.
#
# See MASTER_REFERENCE.md (repo root) sections 12 and 14, and
# PIPELINE_ARCHITECTURE.md. This file was never wired into main_pipeline.py.
# Kept commented-out (not deleted) as an archival record of the approach.
# ============================================================================

# #!/usr/bin/env python3
# """
# SPOVNOB Predictive Decision Engine — Quadrant B Downstream Classifier
# =====================================================================
# Defense-Grade Multi-Modal Behavioral Fusion & Classification.
#
# This module implements the complete machine learning decision framework:
#     1. Data ingestion from calibrated feature matrices with confidence imputation
#     2. CMU-style Tensor Fusion Network (Cartesian Outer Product V⊗A)
#     3. CUDA-accelerated 2D Convolutional Interaction Classifier
#     4. Context-modulated Hidden Markov Model with Viterbi Decoding
#     5. Structured session report serialization
#
# Target Hardware: 44-Core CPU / 512GB ECC RAM / RTX 6000 Ada (48GB VRAM)
# """
#
# import os
# import sys
# import json
# import logging
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from datetime import datetime, timezone
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
#
# # Canonical acoustic schema — imported rather than duplicated so a future
# # acoustic-model swap only needs to change acoustic_extractor.py.
# from audio_isolation.core.acoustic_extractor import ACOUSTIC_COLUMN_NAMES
#
# # ═══════════════════════════════════════════════════════════════════
# # 0. HARD CUDA GATE — No Fallbacks, No CPU Alternatives
# # ═══════════════════════════════════════════════════════════════════
#
# if not torch.cuda.is_available():
#     raise RuntimeError(
#         "FATAL: CUDA is not available on this system. "
#         "The SPOVNOB Predictive Engine requires an NVIDIA RTX 6000 Ada (48GB VRAM) "
#         "with native CUDA drivers. No CPU fallback is implemented by design. "
#         "Verify: nvidia-smi && python -c 'import torch; print(torch.cuda.get_device_name(0))'"
#     )
#
# DEVICE = torch.device("cuda")
# torch.backends.cudnn.benchmark = True  # Auto-tune conv kernel selection for RTX 6000 Ada
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger("PredictiveEngine")
#
# # ═══════════════════════════════════════════════════════════════════
# # 1. CANONICAL FEATURE SCHEMA CONTRACT
# # ═══════════════════════════════════════════════════════════════════
# # Exact column names from the calibrated CSV produced by:
# #   DynamicWindowEngine → BaselineCalibrator → _calibrated_features.csv
# #
# # The schema is partitioned into two modality vectors for the TFN:
# #   VISUAL:   Kinematic + Facial AU + Blink + Co-occurrence + FFT (114 cols)
# #   ACOUSTIC: WavLM paralinguistic embeddings (20 cols)
# #
# # Both are linearly projected into a shared 70-dimensional latent space
# # before the Cartesian Outer Product to yield the [71, 71] interaction matrix.
# # ═══════════════════════════════════════════════════════════════════
#
# # --- Metadata columns (excluded from feature vectors) ---
# METADATA_COLUMNS = [
#     "window_id", "start_time_ms", "end_time_ms",
#     "frame_count", "cumulative_confidence",
#     "context_phase", "question_id", "phase_elapsed_ms",
#     "emotion_label_mode", "blink_count",
#     "deviation_magnitude", "deviation_percentile",
#     "target_ground_truth",
# ]
#
# # --- Visual modality: 114 columns ---
# _AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
#
# _KINEMATIC_COLUMNS = [
#     "left_wrist_velocity_mean", "left_wrist_velocity_max",
#     "right_wrist_velocity_mean", "right_wrist_velocity_max",
#     "motion_energy_mean", "motion_energy_var",
#     "left_hand_face_distance_min", "right_hand_face_distance_min",
#     "emotion_confidence_mean",
#     "gaze_x_mean", "gaze_x_var", "gaze_y_mean", "gaze_y_var",
#     "gaze_z_mean", "gaze_z_var", "gaze_entropy",
#     "gaze_velocity_mean", "gaze_velocity_var",
#     "head_yaw_mean", "head_yaw_var",
#     "head_pitch_mean", "head_pitch_var",
#     "head_roll_mean", "head_roll_var",
#     "macro_motion_energy_mean", "macro_motion_energy_var",
#     "postural_stillness_mean", "postural_stillness_var",
#     "mismatch_ratio", "silent_speech_duration_ms",
# ]
#
# _AU_COLUMNS = []
# for _au in _AU_NAMES:
#     for _suffix in ("_mean", "_max", "_var", "_velocity_max", "_velocity_mean"):
#         _AU_COLUMNS.append(f"{_au}{_suffix}")
#
# _BLINK_COLUMNS = ["blink_rate", "ear_mean", "ear_var"]
#
# _COOCCURRENCE_COLUMNS = [
#     "duchenne_index", "cognitive_load_index", "speech_hesitation_index",
#     "disgust_leak", "postural_freeze_index",
# ]
#
# _FFT_TARGET_CHANNELS = (
#     "head_pitch", "head_yaw", "ear",
#     "AU12_velocity", "AU25_velocity", "macro_motion_energy",
# )
# _FFT_COLUMNS = []
# for _ch in _FFT_TARGET_CHANNELS:
#     for _band in ("tremor", "somatic"):
#         for _metric in ("band_power", "dominant_freq", "spectral_entropy"):
#             _FFT_COLUMNS.append(f"{_ch}_{_band}_{_metric}")
#
# VISUAL_COLUMNS = (
#     _KINEMATIC_COLUMNS + _AU_COLUMNS + _BLINK_COLUMNS
#     + _COOCCURRENCE_COLUMNS + _FFT_COLUMNS
# )
#
# # --- Acoustic modality: 20 columns (canonical schema imported above) ---
# ACOUSTIC_COLUMNS = list(ACOUSTIC_COLUMN_NAMES)
#
# VISUAL_DIM = len(VISUAL_COLUMNS)    # 114
# ACOUSTIC_DIM = len(ACOUSTIC_COLUMNS)  # 20
# PROJECTION_DIM = 70                   # Shared latent projection target
# TFN_DIM = PROJECTION_DIM + 1          # 71 (with bias scalar appended)
# INTERACTION_SHAPE = (1, TFN_DIM, TFN_DIM)  # (C=1, H=71, W=71) for 2D CNN
#
# # Confidence imputation threshold (Target #15)
# CONFIDENCE_THRESHOLD = 0.35
#
# # HMM State Labels
# STATE_STABLE = 0
# STATE_FRICTION = 1
# STATE_NAMES = {STATE_STABLE: "stable_context", STATE_FRICTION: "high_cognitive_load"}
# NUM_STATES = 2
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 2. DATA INGESTION & CONFIDENCE IMPUTATION GATE
# # ═══════════════════════════════════════════════════════════════════
#
# class CalibratedSessionLoader:
#     """
#     Ingests the _calibrated_features.csv from the batch factory and
#     partitions it into visual + acoustic modality vectors with
#     confidence-gated NaN imputation.
#     """
#
#     def __init__(self, calibrated_csv_path: str):
#         self.csv_path = Path(calibrated_csv_path)
#         if not self.csv_path.exists():
#             raise FileNotFoundError(
#                 f"Calibrated feature matrix not found: {self.csv_path}"
#             )
#
#         logger.info(f"Loading calibrated features: {self.csv_path.name}")
#         self.df = pd.read_csv(self.csv_path)
#         logger.info(
#             f"Loaded {len(self.df)} windows × {len(self.df.columns)} columns"
#         )
#
#         # Validate required columns exist
#         self._validate_schema()
#
#     def _validate_schema(self):
#         """
#         Verify that both modality column sets are present in the CSV.
#         Missing columns are fatal — the upstream pipeline produced incomplete data.
#         """
#         missing_visual = [c for c in VISUAL_COLUMNS if c not in self.df.columns]
#         missing_acoustic = [c for c in ACOUSTIC_COLUMNS if c not in self.df.columns]
#
#         if missing_visual:
#             raise RuntimeError(
#                 f"FATAL: {len(missing_visual)} visual columns missing from calibrated CSV. "
#                 f"First 5: {missing_visual[:5]}. "
#                 f"This indicates an incomplete upstream pipeline run."
#             )
#         if missing_acoustic:
#             raise RuntimeError(
#                 f"FATAL: {len(missing_acoustic)} acoustic columns missing from calibrated CSV. "
#                 f"First 5: {missing_acoustic[:5]}. "
#                 f"This indicates the WavLM extractor did not execute."
#             )
#
#         logger.info(
#             f"Schema validated: {VISUAL_DIM} visual + {ACOUSTIC_DIM} acoustic columns"
#         )
#
#     def extract_tensors(self):
#         """
#         Extract visual and acoustic modality tensors with confidence-gated
#         NaN imputation.
#
#         The Confidence Imputation Gate:
#             If cumulative_confidence < CONFIDENCE_THRESHOLD for a window,
#             ALL NaN values in that window are clamped to 0.0 (the Z-score
#             neutral point). This prevents NaN propagation in the Cartesian
#             outer product without introducing false-positive anomalies.
#
#             For windows above the confidence threshold, NaN values are also
#             clamped to 0.0 to guarantee tensor integrity, but these windows
#             are expected to have minimal NaN presence due to high tracker
#             occupancy.
#
#         Returns:
#             visual_tensor:      np.ndarray  [N, 115]
#             acoustic_tensor:    np.ndarray  [N, 20]
#             confidence_weights: np.ndarray  [N]
#             metadata_df:        pd.DataFrame (window_id, timestamps, context, etc.)
#         """
#         # Extract raw modality matrices
#         visual_raw = self.df[VISUAL_COLUMNS].values.astype(np.float64)
#         acoustic_raw = self.df[ACOUSTIC_COLUMNS].values.astype(np.float64)
#
#         # Extract confidence weights for sample weighting in loss function
#         confidence_weights = self.df["cumulative_confidence"].values.astype(np.float64)
#
#         # ── Confidence Imputation Gate ────────────────────────────────
#         # Phase 1: Low-confidence windows — aggressive clamp to Z-score neutral (0.0)
#         low_conf_mask = confidence_weights < CONFIDENCE_THRESHOLD
#         low_conf_count = np.sum(low_conf_mask)
#         if low_conf_count > 0:
#             logger.warning(
#                 f"Confidence Gate: {low_conf_count}/{len(confidence_weights)} windows "
#                 f"below threshold ({CONFIDENCE_THRESHOLD}). Clamping NaN → 0.0"
#             )
#
#         # Phase 2: Universal NaN → 0.0 clamp (safe because data is Z-score normalized)
#         nan_count_visual = np.sum(np.isnan(visual_raw))
#         nan_count_acoustic = np.sum(np.isnan(acoustic_raw))
#
#         if nan_count_visual > 0 or nan_count_acoustic > 0:
#             logger.info(
#                 f"Imputation Gate: Clamping {nan_count_visual} visual NaN + "
#                 f"{nan_count_acoustic} acoustic NaN → 0.0 (Z-score neutral)"
#             )
#
#         visual_clean = np.nan_to_num(visual_raw, nan=0.0, posinf=0.0, neginf=0.0)
#         acoustic_clean = np.nan_to_num(acoustic_raw, nan=0.0, posinf=0.0, neginf=0.0)
#
#         # Normalize confidence weights to [0, 1] range for sample weighting
#         conf_max = np.max(confidence_weights) if np.max(confidence_weights) > 0 else 1.0
#         confidence_weights_normalized = np.clip(confidence_weights / conf_max, 0.0, 1.0)
#
#         # Extract metadata for context-modulated HMM and report generation
#         metadata_cols_present = [c for c in METADATA_COLUMNS if c in self.df.columns]
#         metadata_df = self.df[metadata_cols_present].copy()
#
#         logger.info(
#             f"Tensor extraction complete: visual={visual_clean.shape}, "
#             f"acoustic={acoustic_clean.shape}"
#         )
#
#         return visual_clean, acoustic_clean, confidence_weights_normalized, metadata_df
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 3. CMU-STYLE TENSOR FUSION NETWORK (Cartesian Outer Product)
# # ═══════════════════════════════════════════════════════════════════
#
# class TensorFusionBlock(nn.Module):
#     """
#     CMU Tensor Fusion Network — Modality Projection + Cartesian Outer Product.
#
#     Architecture:
#         1. Linear projection: Visual [N, 114] → [N, 70]
#         2. Linear projection: Acoustic [N, 20] → [N, 70]
#         3. Append bias scalar 1.0 to both: → [N, 71] each
#         4. Cartesian Outer Product: V ⊗ A → [N, 71, 71]
#
#     The resulting [71, 71] matrix preserves:
#         - [0:70, 0:70] block: 4,900 cross-modal interaction nodes
#         - [0:70, 70]   column: 70 unimodal visual features × bias
#         - [70, 0:70]   row:    70 unimodal acoustic features × bias
#         - [70, 70]     scalar: bias × bias = 1.0 (intercept term)
#     """
#
#     def __init__(self):
#         super().__init__()
#
#         # Linear projection layers: map raw modality dims → shared 70-dim latent space
#         self.visual_proj = nn.Linear(VISUAL_DIM, PROJECTION_DIM, bias=False)
#         self.acoustic_proj = nn.Linear(ACOUSTIC_DIM, PROJECTION_DIM, bias=False)
#
#         # Xavier initialization for stable gradient flow through the outer product
#         nn.init.xavier_uniform_(self.visual_proj.weight)
#         nn.init.xavier_uniform_(self.acoustic_proj.weight)
#
#         logger.info(
#             f"TFN Block initialized: "
#             f"Visual [{VISUAL_DIM}→{PROJECTION_DIM}] | "
#             f"Acoustic [{ACOUSTIC_DIM}→{PROJECTION_DIM}] | "
#             f"Outer Product → [{TFN_DIM}×{TFN_DIM}]"
#         )
#
#     def forward(self, visual: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             visual:   [B, 114] raw visual modality tensor
#             acoustic: [B, 20]  raw acoustic modality tensor
#
#         Returns:
#             interaction_image: [B, 1, 71, 71] single-channel interaction tensor
#         """
#         batch_size = visual.shape[0]
#
#         # Step 1: Project into shared 70-dim latent space
#         v_proj = self.visual_proj(visual)      # [B, 70]
#         a_proj = self.acoustic_proj(acoustic)  # [B, 70]
#
#         # Step 2: Append bias scalar 1.0 for unimodal preservation (CMU TFN contract)
#         ones = torch.ones(batch_size, 1, device=visual.device, dtype=visual.dtype)
#         v_augmented = torch.cat([v_proj, ones], dim=1)  # [B, 71]
#         a_augmented = torch.cat([a_proj, ones], dim=1)  # [B, 71]
#
#         # Step 3: Cartesian Outer Product V ⊗ A
#         # torch.bmm requires [B, N, 1] × [B, 1, M] → [B, N, M]
#         v_col = v_augmented.unsqueeze(2)   # [B, 71, 1]
#         a_row = a_augmented.unsqueeze(1)   # [B, 1, 71]
#         interaction = torch.bmm(v_col, a_row)  # [B, 71, 71]
#
#         # Step 4: Reshape to single-channel image for Conv2d: [B, 1, 71, 71]
#         interaction_image = interaction.unsqueeze(1)
#
#         return interaction_image
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 4. CUDA-ACCELERATED 2D CONVOLUTIONAL INTERACTION CLASSIFIER
# # ═══════════════════════════════════════════════════════════════════
#
# class SPOVNOBInteractionClassifier(nn.Module):
#     """
#     2D Convolutional Neural Network operating on the [1, 71, 71]
#     Tensor Fusion interaction image.
#
#     Architecture:
#         Block 1: Conv2d(1→32,  k=3, p=1) → BN → ReLU → MaxPool(2)   → [32, 35, 35]
#         Block 2: Conv2d(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)   → [64, 17, 17]
#         Block 3: Conv2d(64→128, k=3, p=1) → BN → ReLU → AdaptiveAvgPool(4) → [128, 4, 4]
#         Flatten → FC(2048→256) → ReLU → Dropout(0.4) → FC(256→2)
#
#     Output: [B, 2] raw logits for binary classification
#         State 0: Stable Context (baseline-congruent behavior)
#         State 1: High Cognitive Load / Friction State (anomalous deviation)
#     """
#
#     def __init__(self):
#         super().__init__()
#
#         # Tensor Fusion Block (modality projection + outer product)
#         self.tfn = TensorFusionBlock()
#
#         # ── Convolutional Feature Extraction Backbone ──────────────
#         self.conv_block_1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 35, 35]
#         )
#
#         self.conv_block_2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 17, 17]
#         )
#
#         self.conv_block_3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((4, 4)),  # [128, 4, 4]
#         )
#
#         # ── Classification Head ───────────────────────────────────
#         self.classifier = nn.Sequential(
#             nn.Flatten(),                          # [128 * 4 * 4] = [2048]
#             nn.Linear(128 * 4 * 4, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.4),
#             nn.Linear(256, NUM_STATES),            # [2] raw logits
#         )
#
#         self._init_weights()
#
#         total_params = sum(p.numel() for p in self.parameters())
#         logger.info(
#             f"SPOVNOBInteractionClassifier initialized: "
#             f"{total_params:,} parameters | Input: {INTERACTION_SHAPE}"
#         )
#
#     def _init_weights(self):
#         """Kaiming initialization for convolutional layers, Xavier for linear."""
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d):
#                 nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             elif isinstance(module, nn.BatchNorm2d):
#                 nn.init.ones_(module.weight)
#                 nn.init.zeros_(module.bias)
#             elif isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#     def forward(self, visual: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
#         """
#         Full forward pass: modality tensors → TFN → Conv2d → logits.
#
#         Args:
#             visual:   [B, 114] visual modality tensor
#             acoustic: [B, 20]  acoustic modality tensor
#
#         Returns:
#             logits: [B, 2] raw class logits
#         """
#         # Tensor Fusion: [B, 115] + [B, 20] → [B, 1, 71, 71]
#         interaction = self.tfn(visual, acoustic)
#
#         # Convolutional feature extraction
#         x = self.conv_block_1(interaction)  # [B, 32, 35, 35]
#         x = self.conv_block_2(x)           # [B, 64, 17, 17]
#         x = self.conv_block_3(x)           # [B, 128, 4, 4]
#
#         # Classification
#         logits = self.classifier(x)        # [B, 2]
#         return logits
#
#     def predict_probabilities(
#         self, visual: torch.Tensor, acoustic: torch.Tensor
#     ) -> np.ndarray:
#         """
#         Inference-only: returns softmax probabilities as numpy array.
#
#         Returns:
#             probs: [N, 2] numpy array of class probabilities
#         """
#         self.eval()
#         with torch.no_grad():
#             logits = self.forward(visual, acoustic)
#             probs = F.softmax(logits, dim=1)
#         return probs.cpu().numpy()
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 5. CONFIDENCE-WEIGHTED CROSS-ENTROPY LOSS
# # ═══════════════════════════════════════════════════════════════════
#
# class ConfidenceWeightedCrossEntropy(nn.Module):
#     """
#     Custom loss function that incorporates the per-window cumulative
#     confidence as sample weights. Windows with low tracking confidence
#     (occluded frames, lost face-lock) contribute proportionally less
#     to gradient updates, preventing corrupted data from warping model
#     convergence.
#
#     Formula:
#         L = -Σ(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)]) / Σ(w_i)
#
#     Where w_i = normalized cumulative_confidence for window i.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.base_loss = nn.CrossEntropyLoss(reduction="none")
#
#     def forward(
#         self,
#         logits: torch.Tensor,
#         targets: torch.Tensor,
#         confidence_weights: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Args:
#             logits:             [B, 2] raw class logits
#             targets:            [B]    integer class labels (0 or 1)
#             confidence_weights: [B]    normalized confidence weights [0, 1]
#         """
#         # Per-sample cross-entropy loss (unreduced)
#         per_sample_loss = self.base_loss(logits, targets)  # [B]
#
#         # Apply confidence weighting
#         weighted_loss = per_sample_loss * confidence_weights  # [B]
#
#         # Normalize by total weight mass (prevents scale drift)
#         weight_sum = confidence_weights.sum().clamp(min=1e-9)
#         return weighted_loss.sum() / weight_sum
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 6. PYTORCH DATASET FOR WINDOW-LEVEL TRAINING
# # ═══════════════════════════════════════════════════════════════════
#
# class InteractionWindowDataset(Dataset):
#     """
#     PyTorch Dataset wrapping the visual + acoustic modality tensors
#     with per-window confidence weights and ground truth labels.
#     """
#
#     def __init__(
#         self,
#         visual: np.ndarray,
#         acoustic: np.ndarray,
#         confidence_weights: np.ndarray,
#         labels: np.ndarray,
#     ):
#         self.visual = torch.from_numpy(visual).float()
#         self.acoustic = torch.from_numpy(acoustic).float()
#         self.confidence = torch.from_numpy(confidence_weights).float()
#         self.labels = torch.from_numpy(labels).long()
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return (
#             self.visual[idx],
#             self.acoustic[idx],
#             self.confidence[idx],
#             self.labels[idx],
#         )
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 7. MODEL TRAINING LOOP
# # ═══════════════════════════════════════════════════════════════════
#
# def train_classifier(
#     model: SPOVNOBInteractionClassifier,
#     dataset: InteractionWindowDataset,
#     num_epochs: int = 80,
#     learning_rate: float = 1e-3,
#     batch_size: int = 32,
#     weight_decay: float = 1e-4,
# ) -> dict:
#     """
#     Full training loop with confidence-weighted loss, AdamW optimizer,
#     and cosine annealing learning rate schedule.
#
#     Pins tensors to page-locked host memory for async CUDA transfers
#     via pin_memory=True on the DataLoader.
#
#     Returns:
#         training_log: dict with per-epoch loss and accuracy history
#     """
#     model.to(DEVICE)
#     model.train()
#
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,        # Page-locked memory for async DMA transfers
#         persistent_workers=True,
#         drop_last=False,
#     )
#
#     criterion = ConfidenceWeightedCrossEntropy().to(DEVICE)
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=learning_rate,
#         weight_decay=weight_decay,
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=num_epochs, eta_min=1e-6
#     )
#
#     training_log = {"epochs": [], "losses": [], "accuracies": []}
#
#     logger.info(
#         f"Training: {num_epochs} epochs | batch={batch_size} | "
#         f"lr={learning_rate} | weight_decay={weight_decay}"
#     )
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         epoch_correct = 0
#         epoch_total = 0
#
#         for visual, acoustic, confidence, labels in loader:
#             visual = visual.to(DEVICE, non_blocking=True)
#             acoustic = acoustic.to(DEVICE, non_blocking=True)
#             confidence = confidence.to(DEVICE, non_blocking=True)
#             labels = labels.to(DEVICE, non_blocking=True)
#
#             optimizer.zero_grad(set_to_none=True)
#
#             logits = model(visual, acoustic)
#             loss = criterion(logits, labels, confidence)
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#
#             epoch_loss += loss.item() * visual.shape[0]
#             predictions = logits.argmax(dim=1)
#             epoch_correct += (predictions == labels).sum().item()
#             epoch_total += visual.shape[0]
#
#         scheduler.step()
#
#         avg_loss = epoch_loss / max(epoch_total, 1)
#         accuracy = epoch_correct / max(epoch_total, 1)
#
#         training_log["epochs"].append(epoch)
#         training_log["losses"].append(avg_loss)
#         training_log["accuracies"].append(accuracy)
#
#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             lr_current = scheduler.get_last_lr()[0]
#             logger.info(
#                 f"  Epoch {epoch + 1:3d}/{num_epochs} | "
#                 f"Loss: {avg_loss:.4f} | Acc: {accuracy:.3f} | "
#                 f"LR: {lr_current:.6f}"
#             )
#
#     logger.info(
#         f"Training complete. Final accuracy: {training_log['accuracies'][-1]:.3f}"
#     )
#     return training_log
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 8. CONTEXT-MODULATED HIDDEN MARKOV MODEL
# # ═══════════════════════════════════════════════════════════════════
#
# class ContextModulatedHMM:
#     """
#     Hidden Markov Model with dynamic state transition matrices
#     modulated by Target #14 interview context phase indicators.
#
#     States:
#         0: Stable Context (baseline-congruent behavior)
#         1: High Cognitive Load / Friction State
#
#     The transition matrix A[i][j] = P(state_t = j | state_{t-1} = i)
#     is dynamically adjusted based on the active interview phase:
#
#         baseline_neutral / briefing_instruction:
#             Heavy self-transition penalty → state switches are extremely unlikely.
#             The subject is expected to be calm; any micro-fluctuation in CNN output
#             is treated as noise, not a genuine state shift.
#
#         question_delivery:
#             Moderate relaxation → allows detection of anticipatory stress onset,
#             but still requires sustained signal momentum.
#
#         subject_response / investigative_confrontation:
#             Full relaxation → the Viterbi decoder responds rapidly to cognitive
#             friction indicators. This is where genuine deceptive signatures emerge.
#     """
#
#     # Context phase categories and their transition matrix profiles
#     PHASE_PROFILES = {
#         # Phase name → (P(stable→stable), P(friction→friction))
#         # The off-diagonal is 1.0 - diagonal by construction
#         "baseline_neutral":        (0.98, 0.85),
#         "briefing_instruction":    (0.97, 0.85),
#         "question_delivery":       (0.90, 0.88),
#         "subject_response":        (0.80, 0.92),
#         "investigative_confrontation": (0.75, 0.94),
#     }
#     DEFAULT_PROFILE = (0.85, 0.90)  # Fallback for unrecognized phases
#
#     def __init__(self):
#         # Initial state distribution: strongly biased toward Stable
#         self.pi = np.array([0.95, 0.05])
#
#         logger.info("Context-Modulated HMM initialized")
#         logger.info(f"  Initial distribution π = {self.pi}")
#         logger.info(f"  Phase profiles: {len(self.PHASE_PROFILES)} registered")
#
#     def _get_transition_matrix(self, phase_label: str) -> np.ndarray:
#         """
#         Construct the 2×2 transition matrix for a given interview phase.
#
#         A[i][j] = P(state_t = j | state_{t-1} = i)
#         Rows sum to 1.0.
#         """
#         if isinstance(phase_label, float) and np.isnan(phase_label):
#             phase_label = None
#
#         profile = self.PHASE_PROFILES.get(phase_label, self.DEFAULT_PROFILE)
#         p_ss, p_ff = profile
#
#         transition = np.array([
#             [p_ss,       1.0 - p_ss],    # From Stable:  stay vs. switch to Friction
#             [1.0 - p_ff, p_ff],           # From Friction: switch to Stable vs. stay
#         ])
#         return transition
#
#     def _compute_emission_matrix(
#         self, cnn_probabilities: np.ndarray
#     ) -> np.ndarray:
#         """
#         Convert CNN softmax outputs into HMM emission probabilities.
#
#         The CNN produces P(class | observation). The HMM needs
#         P(observation | state), which is the emission probability.
#
#         For this binary state model:
#             B[t, state=0] = P(CNN predicts Stable   | true state = Stable)   ≈ cnn_prob[t, 0]
#             B[t, state=1] = P(CNN predicts Friction  | true state = Friction) ≈ cnn_prob[t, 1]
#
#         This direct mapping is valid because the CNN was trained with
#         within-subject Z-score normalized features, so its probability
#         outputs are already conditioned on the subject's personal baseline.
#         """
#         # Clamp to prevent log(0) in Viterbi
#         emissions = np.clip(cnn_probabilities, 1e-12, 1.0 - 1e-12)
#         return emissions
#
#     def viterbi_decode(
#         self,
#         cnn_probabilities: np.ndarray,
#         context_phases: list,
#     ) -> tuple:
#         """
#         Complete Viterbi decoding with context-modulated transition matrices.
#
#         Finds the single most probable state sequence across the entire session
#         by computing the global maximum-likelihood path through the HMM lattice.
#
#         Args:
#             cnn_probabilities: [T, 2] softmax outputs from CNN
#             context_phases:    [T] list of phase label strings from ContextMapper
#
#         Returns:
#             optimal_path:    [T] array of state indices (0=Stable, 1=Friction)
#             path_probability: float, log-probability of the optimal path
#             trellis:         [T, 2] Viterbi trellis (log probabilities)
#         """
#         T = len(cnn_probabilities)
#         if T == 0:
#             return np.array([], dtype=np.int64), 0.0, np.array([])
#
#         emissions = self._compute_emission_matrix(cnn_probabilities)
#
#         # ── Viterbi Trellis Construction ──────────────────────────────
#         # V[t, s] = max log-probability of reaching state s at time t
#         # B[t, s] = backpointer: which state at t-1 led to max at (t, s)
#         V = np.full((T, NUM_STATES), -np.inf, dtype=np.float64)
#         backptr = np.zeros((T, NUM_STATES), dtype=np.int64)
#
#         # ── Initialization (t=0) ──────────────────────────────────────
#         for s in range(NUM_STATES):
#             V[0, s] = np.log(self.pi[s] + 1e-300) + np.log(emissions[0, s])
#
#         # ── Recursion (t=1..T-1) ──────────────────────────────────────
#         for t in range(1, T):
#             # Get context-modulated transition matrix for this timestep
#             phase = context_phases[t] if t < len(context_phases) else None
#             A = self._get_transition_matrix(phase)
#             log_A = np.log(A + 1e-300)
#
#             for s in range(NUM_STATES):
#                 # Candidate scores: V[t-1, prev_s] + log A[prev_s, s]
#                 candidates = V[t - 1, :] + log_A[:, s]
#
#                 best_prev = np.argmax(candidates)
#                 V[t, s] = candidates[best_prev] + np.log(emissions[t, s])
#                 backptr[t, s] = best_prev
#
#         # ── Termination ───────────────────────────────────────────────
#         optimal_path = np.zeros(T, dtype=np.int64)
#         optimal_path[T - 1] = np.argmax(V[T - 1, :])
#         path_probability = float(V[T - 1, optimal_path[T - 1]])
#
#         # ── Backtracking ──────────────────────────────────────────────
#         for t in range(T - 2, -1, -1):
#             optimal_path[t] = backptr[t + 1, optimal_path[t + 1]]
#
#         logger.info(
#             f"Viterbi decoding complete: {T} windows | "
#             f"Stable: {np.sum(optimal_path == STATE_STABLE)} | "
#             f"Friction: {np.sum(optimal_path == STATE_FRICTION)} | "
#             f"Log-probability: {path_probability:.2f}"
#         )
#
#         return optimal_path, path_probability, V
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 9. SESSION REPORT GENERATOR
# # ═══════════════════════════════════════════════════════════════════
#
# def generate_session_report(
#     session_id: str,
#     metadata_df: pd.DataFrame,
#     cnn_probabilities: np.ndarray,
#     hmm_states: np.ndarray,
#     viterbi_log_prob: float,
#     training_log: dict,
#     output_path: Path,
# ) -> dict:
#     """
#     Serializes the complete predictive analysis into a structured
#     session_report.json containing:
#         - Global stress index (session-level aggregate)
#         - Phase-by-phase behavioral summaries
#         - Precise millisecond timestamps for peak anomaly events
#         - CNN probability traces
#         - HMM state timeline
#         - Training metrics
#     """
#     report = {
#         "session_id": session_id,
#         "generated_at": datetime.now(timezone.utc).isoformat(),
#         "engine_version": "1.0.0",
#         "hardware": {
#             "device": torch.cuda.get_device_name(0),
#             "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1),
#         },
#     }
#
#     # ── Global Stress Index ───────────────────────────────────────
#     # Proportion of session windows classified as Friction State
#     total_windows = len(hmm_states)
#     friction_count = int(np.sum(hmm_states == STATE_FRICTION))
#     stable_count = total_windows - friction_count
#
#     global_stress_index = float(friction_count / max(total_windows, 1))
#     mean_friction_probability = float(
#         np.mean(cnn_probabilities[:, STATE_FRICTION])
#     ) if total_windows > 0 else 0.0
#
#     report["global_metrics"] = {
#         "total_windows": total_windows,
#         "friction_windows": friction_count,
#         "stable_windows": stable_count,
#         "global_stress_index": round(global_stress_index, 4),
#         "mean_friction_probability": round(mean_friction_probability, 4),
#         "viterbi_log_probability": round(viterbi_log_prob, 4),
#     }
#
#     # ── Phase-by-Phase Summaries ──────────────────────────────────
#     phase_summaries = []
#     if "context_phase" in metadata_df.columns and "start_time_ms" in metadata_df.columns:
#         # Group windows by context phase
#         phases = metadata_df["context_phase"].values
#         unique_phases = []
#         current_phase = None
#         phase_start_idx = 0
#
#         for i, phase in enumerate(phases):
#             phase_str = str(phase) if not (isinstance(phase, float) and np.isnan(phase)) else "unknown"
#             if phase_str != current_phase:
#                 if current_phase is not None:
#                     unique_phases.append((current_phase, phase_start_idx, i))
#                 current_phase = phase_str
#                 phase_start_idx = i
#
#         if current_phase is not None:
#             unique_phases.append((current_phase, phase_start_idx, len(phases)))
#
#         for phase_label, start_idx, end_idx in unique_phases:
#             phase_states = hmm_states[start_idx:end_idx]
#             phase_probs = cnn_probabilities[start_idx:end_idx]
#
#             start_ms = float(metadata_df.iloc[start_idx]["start_time_ms"])
#             end_ms = float(metadata_df.iloc[end_idx - 1]["end_time_ms"])
#
#             phase_friction = int(np.sum(phase_states == STATE_FRICTION))
#             phase_total = len(phase_states)
#
#             phase_summaries.append({
#                 "phase_label": phase_label,
#                 "start_ms": start_ms,
#                 "end_ms": end_ms,
#                 "window_count": phase_total,
#                 "friction_windows": phase_friction,
#                 "phase_stress_index": round(
#                     float(phase_friction / max(phase_total, 1)), 4
#                 ),
#                 "mean_friction_probability": round(
#                     float(np.mean(phase_probs[:, STATE_FRICTION])), 4
#                 ),
#                 "peak_friction_probability": round(
#                     float(np.max(phase_probs[:, STATE_FRICTION])), 4
#                 ),
#             })
#
#     report["phase_summaries"] = phase_summaries
#
#     # ── Peak Anomaly Events ───────────────────────────────────────
#     # Find the top-N windows with highest friction probability
#     top_n = min(10, total_windows)
#     if total_windows > 0:
#         friction_probs = cnn_probabilities[:, STATE_FRICTION]
#         top_indices = np.argsort(friction_probs)[-top_n:][::-1]
#
#         peak_events = []
#         for idx in top_indices:
#             idx = int(idx)
#             event = {
#                 "window_id": int(metadata_df.iloc[idx].get("window_id", idx)),
#                 "start_time_ms": float(metadata_df.iloc[idx]["start_time_ms"]),
#                 "end_time_ms": float(metadata_df.iloc[idx]["end_time_ms"]),
#                 "friction_probability": round(float(friction_probs[idx]), 6),
#                 "hmm_state": STATE_NAMES[int(hmm_states[idx])],
#             }
#             if "context_phase" in metadata_df.columns:
#                 phase_val = metadata_df.iloc[idx]["context_phase"]
#                 event["context_phase"] = (
#                     str(phase_val)
#                     if not (isinstance(phase_val, float) and np.isnan(phase_val))
#                     else "unknown"
#                 )
#             if "question_id" in metadata_df.columns:
#                 q_val = metadata_df.iloc[idx]["question_id"]
#                 event["question_id"] = int(q_val) if not np.isnan(q_val) else -1
#             peak_events.append(event)
#
#         report["peak_anomaly_events"] = peak_events
#     else:
#         report["peak_anomaly_events"] = []
#
#     # ── Window-Level Timeline ─────────────────────────────────────
#     timeline = []
#     for i in range(total_windows):
#         window_entry = {
#             "window_id": int(metadata_df.iloc[i].get("window_id", i)),
#             "start_time_ms": float(metadata_df.iloc[i]["start_time_ms"]),
#             "end_time_ms": float(metadata_df.iloc[i]["end_time_ms"]),
#             "cnn_stable_prob": round(float(cnn_probabilities[i, STATE_STABLE]), 6),
#             "cnn_friction_prob": round(float(cnn_probabilities[i, STATE_FRICTION]), 6),
#             "hmm_state": STATE_NAMES[int(hmm_states[i])],
#         }
#         timeline.append(window_entry)
#
#     report["timeline"] = timeline
#
#     # ── Training Metrics ──────────────────────────────────────────
#     if training_log:
#         report["training_metrics"] = {
#             "total_epochs": len(training_log.get("epochs", [])),
#             "final_loss": round(training_log["losses"][-1], 6) if training_log.get("losses") else None,
#             "final_accuracy": round(training_log["accuracies"][-1], 4) if training_log.get("accuracies") else None,
#         }
#
#     # ── Serialize to disk ─────────────────────────────────────────
#     with open(output_path, "w") as f:
#         json.dump(report, f, indent=2, default=str)
#
#     logger.info(f"📋 Session report written: {output_path}")
#     logger.info(
#         f"   Global Stress Index: {global_stress_index:.4f} | "
#         f"Peak Events: {len(report['peak_anomaly_events'])}"
#     )
#
#     return report
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 10. LABEL DERIVATION FROM GROUND TRUTH / DEVIATION MAGNITUDE
# # ═══════════════════════════════════════════════════════════════════
#
# def derive_training_labels(metadata_df: pd.DataFrame) -> np.ndarray:
#     """
#     Derives binary training labels for the classifier.
#
#     Strategy (priority order):
#         1. If `target_ground_truth` column exists (from ELAN injection):
#            - "lying" → 1 (Friction State)
#            - "truth" / "neutral" / "unlabeled" → 0 (Stable Context)
#
#         2. If no ground truth available, falls back to deviation_magnitude:
#            - Windows above the 75th percentile → 1 (anomalous deviation)
#            - Windows at or below the 75th percentile → 0 (baseline-congruent)
#            This is a self-supervised proxy label, not a true ground truth.
#
#     Returns:
#         labels: np.ndarray [N] of int64 (0 or 1)
#     """
#     n = len(metadata_df)
#     labels = np.zeros(n, dtype=np.int64)
#
#     if "target_ground_truth" in metadata_df.columns:
#         gt_values = metadata_df["target_ground_truth"].fillna("unlabeled").str.lower()
#         labels[gt_values == "lying"] = STATE_FRICTION
#         labeled_count = int(np.sum(labels == STATE_FRICTION))
#         logger.info(
#             f"Ground truth labels derived from ELAN: "
#             f"{labeled_count}/{n} friction, {n - labeled_count}/{n} stable"
#         )
#     elif "deviation_magnitude" in metadata_df.columns:
#         dev_mag = metadata_df["deviation_magnitude"].values
#         valid_mask = ~np.isnan(dev_mag)
#         if np.sum(valid_mask) > 4:
#             threshold = np.percentile(dev_mag[valid_mask], 75)
#             labels[valid_mask & (dev_mag > threshold)] = STATE_FRICTION
#             logger.warning(
#                 f"No ELAN ground truth found. Using deviation_magnitude proxy "
#                 f"(threshold={threshold:.3f}, p75). "
#                 f"Friction: {np.sum(labels == STATE_FRICTION)}/{n}"
#             )
#         else:
#             logger.warning(
#                 "Insufficient valid deviation_magnitude values. "
#                 "Defaulting all labels to STATE_STABLE (0)."
#             )
#     else:
#         logger.warning(
#             "No ground truth or deviation_magnitude available. "
#             "All labels defaulted to STATE_STABLE (0)."
#         )
#
#     return labels
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 11. MASTER ORCHESTRATOR — FULL PIPELINE EXECUTION
# # ═══════════════════════════════════════════════════════════════════
#
# class PredictiveEngineOrchestrator:
#     """
#     Top-level orchestrator that executes the complete predictive pipeline:
#         1. Load calibrated CSV → extract modality tensors
#         2. Derive training labels (ELAN ground truth or deviation proxy)
#         3. Train the CNN classifier with confidence-weighted loss
#         4. Run inference to produce per-window probability scores
#         5. Apply context-modulated HMM + Viterbi decoding
#         6. Generate structured session report
#     """
#
#     def __init__(self, output_root: str = "pipeline_system_outputs"):
#         self.output_root = Path(output_root)
#         self.model = SPOVNOBInteractionClassifier().to(DEVICE)
#         self.hmm = ContextModulatedHMM()
#
#     def process_session(
#         self,
#         session_id: str,
#         calibrated_csv_path: str,
#         num_epochs: int = 80,
#         learning_rate: float = 1e-3,
#         batch_size: int = 32,
#     ) -> dict:
#         """
#         Execute the complete predictive pipeline for a single session.
#
#         Args:
#             session_id: Unique session identifier
#             calibrated_csv_path: Path to _calibrated_features.csv
#             num_epochs: CNN training epochs
#             learning_rate: AdamW learning rate
#             batch_size: Training batch size
#
#         Returns:
#             session_report: dict (also serialized to session_report.json)
#         """
#         logger.info("═" * 60)
#         logger.info(f"  PREDICTIVE ENGINE — SESSION: {session_id}")
#         logger.info("═" * 60)
#
#         session_output = self.output_root / session_id
#         session_output.mkdir(parents=True, exist_ok=True)
#
#         # ── Phase 1: Data Ingestion ───────────────────────────────
#         logger.info("── PHASE 1: DATA INGESTION & CONFIDENCE IMPUTATION ──")
#         loader = CalibratedSessionLoader(calibrated_csv_path)
#         visual, acoustic, confidence_weights, metadata_df = loader.extract_tensors()
#
#         # ── Phase 2: Label Derivation ─────────────────────────────
#         logger.info("── PHASE 2: TRAINING LABEL DERIVATION ──")
#         labels = derive_training_labels(metadata_df)
#
#         # ── Phase 3: CNN Training ─────────────────────────────────
#         logger.info("── PHASE 3: CNN CLASSIFIER TRAINING ──")
#         dataset = InteractionWindowDataset(
#             visual, acoustic, confidence_weights, labels
#         )
#
#         # Re-initialize model weights for each session (within-subject training)
#         self.model = SPOVNOBInteractionClassifier().to(DEVICE)
#
#         training_log = train_classifier(
#             model=self.model,
#             dataset=dataset,
#             num_epochs=num_epochs,
#             learning_rate=learning_rate,
#             batch_size=batch_size,
#         )
#
#         # Save trained model checkpoint
#         checkpoint_path = session_output / f"{session_id}_classifier.pt"
#         torch.save({
#             "model_state_dict": self.model.state_dict(),
#             "session_id": session_id,
#             "visual_dim": VISUAL_DIM,
#             "acoustic_dim": ACOUSTIC_DIM,
#             "projection_dim": PROJECTION_DIM,
#             "training_log": training_log,
#         }, checkpoint_path)
#         logger.info(f"Model checkpoint saved: {checkpoint_path}")
#
#         # ── Phase 4: Full-Session Inference ───────────────────────
#         logger.info("── PHASE 4: FULL-SESSION CNN INFERENCE ──")
#         visual_tensor = torch.from_numpy(visual).float().to(DEVICE)
#         acoustic_tensor = torch.from_numpy(acoustic).float().to(DEVICE)
#
#         cnn_probabilities = self.model.predict_probabilities(
#             visual_tensor, acoustic_tensor
#         )
#         logger.info(
#             f"CNN inference complete: {len(cnn_probabilities)} windows scored"
#         )
#
#         # ── Phase 5: Context-Modulated HMM + Viterbi ─────────────
#         logger.info("── PHASE 5: HMM VITERBI DECODING ──")
#         context_phases = []
#         if "context_phase" in metadata_df.columns:
#             for val in metadata_df["context_phase"].values:
#                 if isinstance(val, float) and np.isnan(val):
#                     context_phases.append(None)
#                 else:
#                     context_phases.append(str(val))
#         else:
#             context_phases = [None] * len(cnn_probabilities)
#
#         hmm_states, viterbi_log_prob, trellis = self.hmm.viterbi_decode(
#             cnn_probabilities, context_phases
#         )
#
#         # ── Phase 6: Report Generation ────────────────────────────
#         logger.info("── PHASE 6: SESSION REPORT GENERATION ──")
#         report_path = session_output / "session_report.json"
#
#         report = generate_session_report(
#             session_id=session_id,
#             metadata_df=metadata_df,
#             cnn_probabilities=cnn_probabilities,
#             hmm_states=hmm_states,
#             viterbi_log_prob=viterbi_log_prob,
#             training_log=training_log,
#             output_path=report_path,
#         )
#
#         # ── Save probability trace CSV for dashboard visualization ─
#         trace_df = pd.DataFrame({
#             "window_id": metadata_df["window_id"].values if "window_id" in metadata_df.columns else np.arange(len(hmm_states)),
#             "start_time_ms": metadata_df["start_time_ms"].values,
#             "end_time_ms": metadata_df["end_time_ms"].values,
#             "cnn_stable_prob": cnn_probabilities[:, STATE_STABLE],
#             "cnn_friction_prob": cnn_probabilities[:, STATE_FRICTION],
#             "hmm_state": [STATE_NAMES[int(s)] for s in hmm_states],
#             "hmm_state_id": hmm_states,
#         })
#         trace_path = session_output / f"{session_id}_prediction_trace.csv"
#         trace_df.to_csv(trace_path, index=False)
#         logger.info(f"Prediction trace saved: {trace_path}")
#
#         logger.info("═" * 60)
#         logger.info(f"  🏆 PREDICTIVE ENGINE COMPLETE — {session_id}")
#         logger.info(
#             f"     Global Stress Index: {report['global_metrics']['global_stress_index']:.4f}"
#         )
#         logger.info(
#             f"     Friction Windows:    {report['global_metrics']['friction_windows']}/{report['global_metrics']['total_windows']}"
#         )
#         logger.info(f"     Report:              {report_path}")
#         logger.info("═" * 60)
#
#         return report
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 12. STANDALONE EXECUTION ENTRY POINT
# # ═══════════════════════════════════════════════════════════════════
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="SPOVNOB Predictive Decision Engine — Session Classifier",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     parser.add_argument(
#         "calibrated_csv",
#         type=str,
#         help="Path to the _calibrated_features.csv file",
#     )
#     parser.add_argument(
#         "--session-id",
#         type=str,
#         default=None,
#         help="Session identifier (auto-detected from CSV path if not provided)",
#     )
#     parser.add_argument(
#         "--output-root",
#         type=str,
#         default="pipeline_system_outputs",
#         help="Root output directory (default: pipeline_system_outputs)",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=80,
#         help="Number of training epochs (default: 80)",
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=1e-3,
#         help="Learning rate (default: 1e-3)",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=32,
#         help="Training batch size (default: 32)",
#     )
#
#     args = parser.parse_args()
#
#     # Auto-detect session ID from CSV filename if not provided
#     if args.session_id is None:
#         csv_name = Path(args.calibrated_csv).stem
#         # Strip _calibrated_features suffix
#         args.session_id = csv_name.replace("_calibrated_features", "")
#         logger.info(f"Auto-detected session ID: {args.session_id}")
#
#     print("═" * 60)
#     print("  SPOVNOB PREDICTIVE DECISION ENGINE")
#     print(f"  Session:  {args.session_id}")
#     print(f"  Input:    {args.calibrated_csv}")
#     print(f"  GPU:      {torch.cuda.get_device_name(0)}")
#     print(f"  VRAM:     {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")
#     print(f"  Epochs:   {args.epochs}")
#     print(f"  LR:       {args.lr}")
#     print(f"  Batch:    {args.batch_size}")
#     print("═" * 60)
#
#     orchestrator = PredictiveEngineOrchestrator(output_root=args.output_root)
#     report = orchestrator.process_session(
#         session_id=args.session_id,
#         calibrated_csv_path=args.calibrated_csv,
#         num_epochs=args.epochs,
#         learning_rate=args.lr,
#         batch_size=args.batch_size,
#     )
