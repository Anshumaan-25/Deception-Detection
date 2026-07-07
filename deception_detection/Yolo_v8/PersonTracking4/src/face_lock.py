import numpy as np
import os
import logging
from insightface.app import FaceAnalysis
from numpy.linalg import norm


class FaceLock:

    def __init__(self, engine_cache_dir: str = None):
        """
        InsightFace identity matcher with TensorRT acceleration.

        Args:
            engine_cache_dir: Absolute path to the TensorRT engine cache
                directory. If provided, InsightFace's ONNX models are
                JIT-compiled to FP16 TensorRT engines on first inference
                and cached for zero-latency loading on subsequent runs.
                If None, defaults to weights/trt_engines/insightface/.
        """
        self.logger = logging.getLogger("FaceLock_TRT")

        ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../weights"))

        # Resolve TRT engine cache directory
        if engine_cache_dir is None:
            # Default: project_root/weights/trt_engines/insightface/
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            engine_cache_dir = os.path.join(
                project_root, "weights", "trt_engines", "insightface")

        os.makedirs(engine_cache_dir, exist_ok=True)

        # CUDA first, CPU as genuine fallback. The TensorRT EP is deliberately
        # NOT listed: this box has no TensorRT runtime (Strategy B / PyTorch-
        # native), and InsightFace's session loader falls back to CPU-ONLY the
        # instant ANY requested provider errors (it retries with just
        # ['CPUExecutionProvider'], dropping CUDA too). So listing an
        # unavailable TensorRT EP silently forced every InsightFace model onto
        # CPU — ~15x slower than realtime. Re-add the TensorRT EP here only once
        # a TensorRT runtime is actually installed and verified on the box.
        providers = [
            ("CUDAExecutionProvider", {}),
            ("CPUExecutionProvider", {}),
        ]

        self.logger.info(
            f"🚀 Initializing InsightFace (CUDA EP, cache: {engine_cache_dir})")

        self.app = FaceAnalysis(
            name="buffalo_l", root=ROOT, providers=providers)

        self.app.prepare(ctx_id=0)
        self.target_embedding = None
        self.threshold = 0.45

    # --------------------------------------
    def cosine(self, a, b):
        return np.dot(a,b)/(norm(a)*norm(b))

    # --------------------------------------
    def lock(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        crop = frame[y1:y2, x1:x2]

        faces = self.app.get(crop)

        if len(faces) == 0:
            return False

        self.target_embedding = faces[0].embedding
        print("\n Face identity locked")
        return True

    # --------------------------------------
    def match(self, frame, detections):

        if self.target_embedding is None:
            return None, 0.0

        best_score = -1.0
        best_bbox = None

        for det in detections:

            x1,y1,x2,y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]

            faces = self.app.get(crop)
            if len(faces)==0:
                continue

            emb = faces[0].embedding
            score = self.cosine(self.target_embedding, emb)

            if score > best_score:
                best_score = score
                best_bbox = det["bbox"]

        # Clamp similarity score strictly between 0.0 and 1.0 for the joint weight product
        clamped_score = max(0.0, min(1.0, float(best_score))) if best_score != -1.0 else 0.0

        if best_score > self.threshold:
            return best_bbox, clamped_score

        return None, 0.0

