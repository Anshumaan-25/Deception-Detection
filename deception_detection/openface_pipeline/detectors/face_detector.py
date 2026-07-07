"""
FaceDetector — RetinaFace (mobilenet0.25) production wrapper.
=============================================================
RECONSTRUCTED 2026-07-07. The desktop original was lost in the July-2026
transfer incident; this is a clean-room rebuild against the call contract
fixed by api/extractor.py, with the decode flow ported from the academic
repo's own demo code (OpenFace-3.0/demo2.py::preprocess_image).

Call contract (api/extractor.py lines 64 and 190):
    detect(img_tensor, (im_height, im_width)) -> torch.FloatTensor [N, 5]
where
  - img_tensor is float32, shape [1, 3, H, W], BGR channel order,
    RetinaFace mean (104, 117, 123) already subtracted, already on the
    target device (extractor does all of that preprocessing itself);
  - each result row is (x1, y1, x2, y2, score) in input-pixel coordinates;
  - rows are sorted by descending score — extractor.process_batch() locks
    onto row 0 as the primary face of the person crop.

Deliberate differences from demo2.py:
  - `load_model` is inlined (with check_keys/remove_prefix) rather than
    imported from Pytorch_Retinaface/detect.py: that module calls
    argparse.parse_args() at import time and would abort any host process
    (e.g. the batch daemon) whose argv it doesn't recognize.
  - confidence_threshold defaults to 0.5 (demo2 filtered at 0.02 and let
    callers re-filter at 0.5; extractor.py never re-filters, so the
    production threshold lives here).
  - PriorBox anchors are cached per input resolution — person crops from a
    tracked stream reuse a handful of shapes, and anchor generation is the
    only CPU-heavy step of the decode path.
"""

import logging

import numpy as np
import torch

from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.box_utils import decode
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.data.config import cfg_mnet


def _check_keys(model, pretrained_state_dict, logger):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_keys = model_keys & ckpt_keys
    if len(used_keys) == 0:
        raise RuntimeError("FATAL: checkpoint and RetinaFace model share no keys.")
    missing = model_keys - ckpt_keys
    if missing:
        logger.warning(f"RetinaFace checkpoint missing {len(missing)} keys (strict=False load).")


def _remove_prefix(state_dict, prefix):
    strip = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {strip(key): value for key, value in state_dict.items()}


def _load_weights(model, weights_path, device, logger):
    # Inlined from Pytorch_Retinaface/detect.py::load_model (module not
    # importable in-process — see module docstring).
    if device.type == "cpu":
        pretrained = torch.load(weights_path, map_location=lambda storage, loc: storage)
    else:
        pretrained = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(device.index or 0))
    if "state_dict" in pretrained.keys():
        pretrained = _remove_prefix(pretrained["state_dict"], "module.")
    else:
        pretrained = _remove_prefix(pretrained, "module.")
    _check_keys(model, pretrained, logger)
    model.load_state_dict(pretrained, strict=False)
    return model


class FaceDetector:
    def __init__(self, weights_path: str, device: str = "cuda",
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("RetinaFace_Detector")
        self.device = torch.device(device)
        self.cfg = cfg_mnet
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        net = RetinaFace(cfg=self.cfg, phase="test")
        net = _load_weights(net, weights_path, self.device, self.logger)
        net.eval()
        self.net = net.to(self.device)
        self._prior_cache = {}

        self.logger.info(f"✅ RetinaFace mobilenet0.25 loaded from {weights_path}")

    def _priors_for(self, im_height: int, im_width: int) -> torch.Tensor:
        key = (im_height, im_width)
        if key not in self._prior_cache:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            self._prior_cache[key] = priorbox.forward().to(self.device)
        return self._prior_cache[key]

    @torch.no_grad()
    def detect(self, img_tensor: torch.Tensor, image_size: tuple) -> torch.Tensor:
        """RetinaFace forward pass + decode + NMS. See module docstring."""
        im_height, im_width = int(image_size[0]), int(image_size[1])

        loc, conf, _landms = self.net(img_tensor)

        priors = self._priors_for(im_height, im_width)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        scale = torch.tensor([im_width, im_height, im_width, im_height],
                             dtype=boxes.dtype, device=boxes.device)
        boxes = (boxes * scale).cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes, scores = boxes[inds], scores[inds]

        order = scores.argsort()[::-1]
        boxes, scores = boxes[order], scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        # NMS preserves the descending-score order, so dets[0] stays the
        # primary face — the property extractor.process_batch() relies on.
        return torch.from_numpy(dets)
