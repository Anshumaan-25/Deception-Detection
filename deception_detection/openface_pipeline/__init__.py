"""
openface_pipeline — production gateway package for OpenFace-3.0 inference.
===========================================================================
RECONSTRUCTED 2026-07-07. The desktop original of this package's wiring was
lost in the July-2026 transfer incident (see MASTER_REFERENCE §12); this
file re-creates it. api/extractor.py and detectors/unified_detector.py are
the recovered pre-merge working-tree files (newest surviving copies, from
OpenFace-3.0/openface_pipeline/); detectors/face_detector.py and
detectors/landmark_detector.py are clean-room reconstructions written
against extractor.py's call contract — see their module docstrings.

Why this bootstrap exists
-------------------------
main_pipeline.py imports `openface_pipeline.api.extractor` with
cwd=deception_detection/. extractor.py then does BARE imports
(`from detectors.face_detector import ...`), and the detector modules pull
in the legacy academic repo living at the sibling OpenFace-3.0/ directory,
whose vendored sub-repos use their own bare intra-repo imports:
  - Pytorch_Retinaface/data/data_augment.py: `from utils.box_utils import ...`
  - STAR/lib/utility.py:                     `from lib.dataset import ...`
So four roots must be importable. Importing this package performs that
sys.path injection exactly once, mirroring the pattern the academic repo
itself uses (OpenFace-3.0/interface.py lines 19-20).

Path shadowing note: putting Pytorch_Retinaface/ and STAR/ on sys.path
makes generic top-level names (`utils`, `data`, `layers`, `models`, `lib`)
resolvable to those repos. Nothing else in the deception pipeline imports
those names bare; keep it that way.
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DD_ROOT = os.path.dirname(_PKG_DIR)                      # deception_detection/
_OPENFACE_ROOT = os.path.join(_DD_ROOT, "OpenFace-3.0")   # legacy academic repo

for _p in (
    _PKG_DIR,                                             # bare `detectors.*`
    _OPENFACE_ROOT,                                       # `model.MLT`, `Pytorch_Retinaface.*`, `STAR.*`
    os.path.join(_OPENFACE_ROOT, "Pytorch_Retinaface"),   # its internal bare imports
    os.path.join(_OPENFACE_ROOT, "STAR"),                 # `lib.*` inside STAR
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
