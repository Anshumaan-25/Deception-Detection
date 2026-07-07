import torch
import torch.nn.functional as F
import numpy as np
import logging
import sys
import os

# --- PATH INJECTION BRIDGE ---
# Dynamically resolve the legacy academic repository root relative to this file.
# NOTE (2026-07-07 relocation): this copy lives at the TOP-LEVEL
# deception_detection/openface_pipeline/detectors/ (not nested inside
# OpenFace-3.0/openface_pipeline/detectors/ like the original), so the
# academic repo is a sibling of the package: ../../OpenFace-3.0.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_OPENFACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "OpenFace-3.0"))

if LEGACY_OPENFACE_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_OPENFACE_ROOT)

# TensorRT imports — mandatory on production server
import tensorrt as trt

# Safely import the MLT architecture from the legacy folder (fallback reference only)
from model.MLT import MLT


class TensorRTMLTEngine:
    """
    Direct TensorRT runtime wrapper for the compiled MLT .engine plan.
    Bypasses PyTorch entirely — allocates CUDA buffers manually and executes
    via TensorRT's optimized CUDA kernels.

    This class is a zero-overhead inference executor. All post-processing
    (softmax, gaze conversion) happens downstream in numpy after the
    GPU→CPU transfer.
    """

    def __init__(self, engine_path: str):
        self.logger = logging.getLogger("TRT_MLT_Engine")

        if not os.path.exists(engine_path):
            raise RuntimeError(
                f"FATAL: Compiled MLT TensorRT engine not found at '{engine_path}'. "
                f"Run 'python tools/compile_pipeline_trt.py' on the production "
                f"server before executing the pipeline."
            )

        self.logger.info(f"Loading TensorRT engine: {engine_path} "
                         f"({os.path.getsize(engine_path) / 1e6:.1f} MB)")

        # Initialize TensorRT runtime
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                f"FATAL: Failed to deserialize TensorRT engine from '{engine_path}'. "
                f"Engine may be corrupted or compiled for a different GPU architecture."
            )

        self.context = self.engine.create_execution_context()

        # Resolve binding indices and shapes
        self.input_idx = self.engine.get_binding_index("input")
        self.emotion_idx = self.engine.get_binding_index("emotion")
        self.gaze_idx = self.engine.get_binding_index("gaze")
        self.au_idx = self.engine.get_binding_index("action_units")

        self.logger.info(
            f"✅ TensorRT MLT engine loaded. "
            f"Bindings: input={self.input_idx}, emotion={self.emotion_idx}, "
            f"gaze={self.gaze_idx}, au={self.au_idx}"
        )

    def infer(self, input_tensor: torch.Tensor):
        """
        Execute the MLT forward pass entirely on TensorRT.

        Args:
            input_tensor: CUDA tensor of shape [B, 3, 224, 224], float32.

        Returns:
            Tuple of (emotion_output, gaze_output, au_output) as numpy arrays.
        """
        batch_size = input_tensor.shape[0]

        # Set the dynamic batch dimension on the input binding
        self.context.set_binding_shape(
            self.input_idx, (batch_size, 3, 224, 224)
        )

        # Allocate output tensors on CUDA — shapes derived from model architecture
        # MLT outputs: emotion=[B,8], gaze=[B,2], au=[B,8]
        emotion_out = torch.empty(batch_size, 8, dtype=torch.float32, device="cuda")
        gaze_out = torch.empty(batch_size, 2, dtype=torch.float32, device="cuda")
        au_out = torch.empty(batch_size, 8, dtype=torch.float32, device="cuda")

        # Ensure input is contiguous in CUDA memory
        input_tensor = input_tensor.contiguous()

        # Bind device pointers — TensorRT operates directly on GPU memory
        bindings = [None] * self.engine.num_bindings
        bindings[self.input_idx] = input_tensor.data_ptr()
        bindings[self.emotion_idx] = emotion_out.data_ptr()
        bindings[self.gaze_idx] = gaze_out.data_ptr()
        bindings[self.au_idx] = au_out.data_ptr()

        # Execute the compiled engine — single fused CUDA kernel launch
        success = self.context.execute_v2(bindings=bindings)
        if not success:
            raise RuntimeError(
                "FATAL: TensorRT execute_v2() returned failure. "
                "Possible OOM or engine corruption."
            )

        # Transfer results to CPU numpy arrays
        return (
            emotion_out.cpu().numpy(),
            gaze_out.cpu().numpy(),
            au_out.cpu().numpy(),
        )


class UnifiedBehaviorDetector:
    def __init__(self, weights_path: str, device: str = "cuda"):
        """
        Production wrapper for the Unified Multi-Task Learning (MLT) backend.
        Extracts Emotion, Gaze, and Action Units (AUs) in a single batched pass.

        Runtime Routing:
          - If a compiled .engine file exists → TensorRT direct execution (zero PyTorch overhead)
          - If no .engine file exists → FATAL ERROR (production requires compiled engines)
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("Unified_Behavior_API")
        self.device = torch.device(device)

        self.emotion_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        self.au_labels = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]

        # ── TensorRT Engine Resolution ──────────────────────────────────
        # Resolve the .engine path from the project structure
        # Expected at: weights/trt_engines/MLT.engine (relative to project root)
        # NOTE (2026-07-07 relocation): from the top-level package location,
        # project root (deception_detection/) is two levels up, not three.
        project_root = os.path.abspath(
            os.path.join(CURRENT_DIR, "..", "..")
        )
        engine_path = os.path.join(project_root, "weights", "trt_engines", "MLT.engine")

        if os.path.exists(engine_path):
            self.logger.info(
                f"🚀 TensorRT engine detected. Routing MLT inference through "
                f"hardware-fused CUDA kernels (PyTorch bypassed entirely)."
            )
            self.trt_engine = TensorRTMLTEngine(engine_path)
            self.model = None  # PyTorch model is NOT loaded — zero memory overhead
            self._use_tensorrt = True
        else:
            raise RuntimeError(
                f"FATAL: Compiled TensorRT engine not found at '{engine_path}'. "
                f"Production pipeline requires pre-compiled engines. "
                f"Run: python tools/compile_pipeline_trt.py --project-root {project_root}"
            )

    @torch.no_grad()
    def analyze(self, batched_crops: torch.Tensor) -> list:
        """
        Executes the heavy CUDA inference pass.

        Args:
            batched_crops: Tensor of shape [B, 3, 224, 224] on GPU.

        Returns:
            List of dictionaries containing behavior metrics for each face in the batch.
        """
        # Ensure batched dimension exists
        if batched_crops.dim() == 3:
            batched_crops = batched_crops.unsqueeze(0)

        batched_crops = batched_crops.to(self.device)

        # ── Forward Pass ─────────────────────────────────────────────
        if self._use_tensorrt:
            emotion_probs_raw, gaze_numpy, au_numpy = self.trt_engine.infer(batched_crops)
            # TensorRT outputs raw logits — apply softmax in numpy
            # Numerically stable softmax
            emotion_max = emotion_probs_raw.max(axis=1, keepdims=True)
            emotion_exp = np.exp(emotion_probs_raw - emotion_max)
            emotion_probs = emotion_exp / emotion_exp.sum(axis=1, keepdims=True)
        else:
            # This branch should never execute in production
            raise RuntimeError(
                "FATAL: PyTorch fallback path reached in production. "
                "This indicates a configuration error."
            )

        # ── Post-process Gaze (Spherical → 3D Cartesian) ────────────
        yaw = gaze_numpy[:, 0]
        pitch = gaze_numpy[:, 1]

        Gx = -np.cos(pitch) * np.sin(yaw)
        Gy = -np.sin(pitch)
        Gz = -np.cos(pitch) * np.cos(yaw)
        gaze_3d = np.stack([Gx, Gy, Gz], axis=1)  # Shape: [B, 3]

        # ── Structure JSON-ready Results ─────────────────────────────
        batch_results = []
        for i in range(batched_crops.size(0)):
            top_emotion_idx = np.argmax(emotion_probs[i])

            res = {
                "emotion": {
                    "primary_label": self.emotion_labels[top_emotion_idx],
                    "confidence": float(emotion_probs[i][top_emotion_idx]),
                    "logits": emotion_probs[i].tolist()
                },
                "gaze_3d": gaze_3d[i].tolist(),
                "gaze_raw_rad": gaze_numpy[i].tolist(),
                "action_units": {self.au_labels[j]: float(au_numpy[i][j]) for j in range(8)}
            }
            batch_results.append(res)

        return batch_results
