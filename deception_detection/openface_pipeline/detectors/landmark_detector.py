"""
LandmarkDetector — STAR (WFLW 98-point, STARLoss) production wrapper.
=====================================================================
RECONSTRUCTED 2026-07-07. The desktop original was lost in the July-2026
transfer incident; this is a clean-room rebuild against the call contract
fixed by api/extractor.py, porting the alignment flow from the academic
repo's own demo code (OpenFace-3.0/STAR/demo.py and OpenFace-3.0/demo.py).

Call contract (api/extractor.py lines 91 and 208):
    detect(frame_bgr, bbox_xyxy) -> np.ndarray [98, 2]
where frame_bgr is the full uint8 BGR frame/person-crop and bbox_xyxy is
the RetinaFace box; landmarks come back in frame pixel coordinates.
Scale/center derivation follows OpenFace-3.0/demo.py::landmark_detection:
    center = bbox midpoint;  scale = min(bbox_w, bbox_h) / 200 * 1.05

Why the transform classes are ported rather than imported:
STAR/demo.py imports `dlib` and `gradio` at module top — two heavyweight
packages the production pipeline must not depend on just to reach three
small numpy/cv2 transform classes and the Alignment container. The pieces
below (GetCropMatrix, TransformPerspective, the norm/denorm/postprocess
math) are verbatim ports of STAR/demo.py lines 15-179; the network itself
is still built by the real STAR code (STAR.lib.utility get_config/get_net),
so checkpoint compatibility is untouched.
"""

import argparse
import logging
import math

import cv2
import numpy as np
import torch

from STAR.lib import utility


class _GetCropMatrix:
    """Ported verbatim from STAR/demo.py::GetCropMatrix (lines 15-63)."""

    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)
        fx, fy = from_center
        tx, ty = to_center
        acos = scale * cosv
        asin = scale * sinv
        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]
        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]
        return np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size
        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        return self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])


class _TransformPerspective:
    """Ported verbatim from STAR/demo.py::TransformPerspective (lines 66-77)."""

    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class LandmarkDetector:
    def __init__(self, weights_path: str, device: str = "cuda"):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("STAR_Landmark_Detector")
        self.device = torch.device(device)
        self.input_size = 256          # STAR/demo.py::Alignment defaults
        self.target_face_scale = 1.0

        # Network construction exactly as STAR/demo.py::Alignment.__init__
        # (and OpenFace-3.0/demo.py lines 239-245): the 'alignment' config
        # from STAR/conf resolves the stackedHGnet_v1 WFLW architecture.
        args = argparse.Namespace(
            config_name="alignment",
            device_id=self.device.index if self.device.type == "cuda" else -1,
        )
        self.config = utility.get_config(args)
        self.config.device_id = args.device_id
        utility.set_environment(self.config)
        self.config.init_instance()

        net = utility.get_net(self.config)
        checkpoint = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        net = net.to(self.config.device_id if self.device.type == "cuda" else "cpu")
        net.eval()
        self.net = net

        self.get_crop_matrix = _GetCropMatrix(
            image_size=self.input_size,
            target_face_scale=self.target_face_scale,
            align_corners=True)
        self.transform_perspective = _TransformPerspective(image_size=self.input_size)

        self.logger.info(f"✅ STAR WFLW-98 landmark network loaded from {weights_path}")

    def _denorm_points(self, points, align_corners=True):
        # Ported from STAR/demo.py::Alignment.denorm_points (lines 136-142).
        if align_corners:
            return (points + 1) / 2 * torch.tensor(
                [self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2)
        return ((points + 1) * torch.tensor(
            [self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2

    @staticmethod
    def _postprocess(src_points, coeff):
        # Ported from STAR/demo.py::Alignment.postprocess (lines 155-163):
        # maps landmarks from crop space back through the inverse crop matrix.
        dst = np.zeros(src_points.shape, dtype=np.float32)
        for i in range(src_points.shape[0]):
            dst[i][0] = coeff[0][0] * src_points[i][0] + coeff[0][1] * src_points[i][1] + coeff[0][2]
            dst[i][1] = coeff[1][0] * src_points[i][0] + coeff[1][1] * src_points[i][1] + coeff[1][2]
        return dst

    @torch.no_grad()
    def detect(self, frame: np.ndarray, bbox) -> np.ndarray:
        """Full-frame + bbox → 98 WFLW landmarks in frame pixel coords."""
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        center_w = (x1 + x2) / 2.0
        center_h = (y1 + y2) / 2.0
        scale = min(x2 - x1, y2 - y1) / 200.0 * 1.05

        # Ported from STAR/demo.py::Alignment.preprocess/analyze (144-179).
        matrix = self.get_crop_matrix.process(scale, center_w, center_h)
        input_tensor = self.transform_perspective.process(frame, matrix)
        input_tensor = input_tensor[np.newaxis, :]
        input_tensor = torch.from_numpy(input_tensor).float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        input_tensor = input_tensor.to(
            self.config.device_id if self.device.type == "cuda" else "cpu")

        output = self.net(input_tensor)
        landmarks = output[-1][0]

        landmarks = self._denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        return self._postprocess(landmarks, np.linalg.inv(matrix))
