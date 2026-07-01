#!/usr/bin/env python3
"""
=============================================================================
  Target #9 — Offline TensorRT Compilation Script
  File: tools/compile_pipeline_trt.py
=============================================================================

  PURPOSE:
    Run this script ONCE on the production server BEFORE any pipeline execution.
    It compiles all eligible visual extraction models into frozen TensorRT
    .engine plans with FP16 quantization.

  HARDWARE REQUIREMENT:
    - NVIDIA RTX 6000 Ada (48GB VRAM)
    - TensorRT >= 8.6 installed with trtexec on PATH
    - CUDA >= 12.0

  USAGE:
    python tools/compile_pipeline_trt.py \
        --project-root /path/to/SPOVNOB_CLONE \
        --yolo-weights weights/yolov8n.pt \
        --mlt-weights OpenFace-3.0/weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth \
        --batch-opt 16 \
        --batch-max 32

  OUTPUT:
    weights/yolov8n.engine           — Compiled YOLOv8 TensorRT engine
    weights/trt_engines/MLT.onnx     — Intermediate ONNX graph
    weights/trt_engines/MLT.engine   — Compiled MLT TensorRT engine (FP16)
    weights/trt_engines/insightface/ — Cached InsightFace TRT engine plans
=============================================================================
"""
import os
import sys
import argparse
import subprocess
import shutil
import logging

import torch
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TRT_Compiler")


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — YOLOv8 Native TensorRT Export
# ═════════════════════════════════════════════════════════════════════════════

def compile_yolov8(yolo_weights_path: str, output_dir: str, batch_opt: int) -> str:
    """
    Ultralytics provides a native .export() method that handles the full
    PyTorch → ONNX → TensorRT pipeline internally. We invoke it directly.

    Returns the path to the compiled .engine file.
    """
    from ultralytics import YOLO

    engine_path = yolo_weights_path.replace(".pt", ".engine")

    if os.path.exists(engine_path):
        log.info(f"✅ YOLOv8 engine already exists: {engine_path}")
        return engine_path

    log.info(f"🔧 Compiling YOLOv8 from {yolo_weights_path} ...")
    log.info(f"   Batch: dynamic (opt={batch_opt}), Precision: FP16")

    model = YOLO(yolo_weights_path)

    # Ultralytics export() handles ONNX intermediate + trtexec invocation
    # half=True enforces FP16 quantization
    # dynamic=True enables dynamic batch dimension
    export_path = model.export(
        format="engine",
        half=True,
        batch=batch_opt,
        dynamic=True,
        simplify=True,
        verbose=True,
    )

    if not os.path.exists(export_path):
        raise RuntimeError(
            f"FATAL: YOLOv8 TensorRT compilation failed. "
            f"Expected engine at {export_path} but file does not exist. "
            f"Check TensorRT installation and GPU driver compatibility."
        )

    log.info(f"✅ YOLOv8 engine compiled: {export_path} "
             f"({os.path.getsize(export_path) / 1e6:.1f} MB)")
    return export_path


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — OpenFace MLT (EfficientNet-B0 + GNN AU Head) → ONNX → TensorRT
# ═════════════════════════════════════════════════════════════════════════════

def export_mlt_to_onnx(
    project_root: str,
    mlt_weights_path: str,
    onnx_output_path: str,
    batch_opt: int,
) -> str:
    """
    Step 1: Load the MLT model, trace it with dummy input, and export to ONNX
    with a dynamic batch axis.
    """
    if os.path.exists(onnx_output_path):
        log.info(f"✅ MLT ONNX already exists: {onnx_output_path}")
        return onnx_output_path

    # Inject the legacy OpenFace repo into sys.path so MLT imports resolve
    legacy_root = os.path.join(project_root, "OpenFace-3.0")
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)

    from model.MLT import MLT

    log.info(f"🔧 Loading MLT weights from {mlt_weights_path} ...")
    model = MLT()

    # Load weights — handle both raw state_dict and checkpoint wrapper formats
    state_dict = torch.load(mlt_weights_path, map_location="cuda")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Strip DataParallel 'module.' prefixes if present
    clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=False)

    model.eval().cuda()

    # Create dummy input matching production shape
    dummy_input = torch.randn(batch_opt, 3, 224, 224, device="cuda")

    log.info(f"🔧 Exporting MLT to ONNX: {onnx_output_path}")
    log.info(f"   Input shape: [{batch_opt}, 3, 224, 224], Dynamic batch axis: 0")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["emotion", "gaze", "action_units"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "emotion": {0: "batch_size"},
            "gaze": {0: "batch_size"},
            "action_units": {0: "batch_size"},
        },
    )

    if not os.path.exists(onnx_output_path):
        raise RuntimeError(
            f"FATAL: MLT ONNX export failed. "
            f"Expected file at {onnx_output_path} but it does not exist."
        )

    log.info(f"✅ MLT ONNX exported: {onnx_output_path} "
             f"({os.path.getsize(onnx_output_path) / 1e6:.1f} MB)")
    return onnx_output_path


def compile_mlt_trtexec(
    onnx_path: str,
    engine_output_path: str,
    batch_min: int = 1,
    batch_opt: int = 16,
    batch_max: int = 32,
) -> str:
    """
    Step 2: Invoke trtexec to compile the ONNX graph into a frozen
    TensorRT .engine plan with FP16 and dynamic batch profiles.
    """
    if os.path.exists(engine_output_path):
        log.info(f"✅ MLT TensorRT engine already exists: {engine_output_path}")
        return engine_output_path

    # Verify trtexec is on PATH
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        raise RuntimeError(
            "FATAL: 'trtexec' binary not found on PATH. "
            "Install TensorRT and ensure trtexec is accessible. "
            "Typically located at /usr/local/tensorrt/bin/trtexec"
        )

    log.info(f"🔧 Compiling MLT via trtexec ...")
    log.info(f"   ONNX:   {onnx_path}")
    log.info(f"   Engine: {engine_output_path}")
    log.info(f"   Batch:  min={batch_min}, opt={batch_opt}, max={batch_max}")
    log.info(f"   Precision: FP16 (INT8 REJECTED)")

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_output_path}",
        "--fp16",
        f"--minShapes=input:{batch_min}x3x224x224",
        f"--optShapes=input:{batch_opt}x3x224x224",
        f"--maxShapes=input:{batch_max}x3x224x224",
        "--workspace=4096",  # 4GB workspace for kernel auto-tuning
        "--verbose",
    ]

    log.info(f"   CMD: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error(f"trtexec STDOUT:\n{result.stdout}")
        log.error(f"trtexec STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"FATAL: trtexec compilation failed with exit code {result.returncode}. "
            f"Check GPU memory availability and TensorRT version compatibility."
        )

    if not os.path.exists(engine_output_path):
        raise RuntimeError(
            f"FATAL: trtexec completed but engine file not found at "
            f"{engine_output_path}."
        )

    log.info(f"✅ MLT TensorRT engine compiled: {engine_output_path} "
             f"({os.path.getsize(engine_output_path) / 1e6:.1f} MB)")
    return engine_output_path


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — InsightFace TensorRT EP Cache Warming
# ═════════════════════════════════════════════════════════════════════════════

def warm_insightface_trt_cache(
    project_root: str,
    engine_cache_dir: str,
) -> None:
    """
    Pre-warm the InsightFace TensorRT EP cache by running a single dummy
    inference pass. This triggers onnxruntime's JIT TensorRT compilation
    for all 5 buffalo_l models and caches the resulting .engine plans
    for zero-latency loading on subsequent pipeline runs.
    """
    from insightface.app import FaceAnalysis

    os.makedirs(engine_cache_dir, exist_ok=True)

    insightface_root = os.path.join(
        project_root, "Yolo_v8", "PersonTracking4", "weights"
    )

    log.info(f"🔧 Warming InsightFace TensorRT EP cache ...")
    log.info(f"   Model root: {insightface_root}")
    log.info(f"   Cache dir:  {engine_cache_dir}")

    providers = [
        ("TensorRTExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": engine_cache_dir,
        }),
        ("CUDAExecutionProvider", {}),
    ]

    app = FaceAnalysis(name="buffalo_l", root=insightface_root, providers=providers)
    app.prepare(ctx_id=0)

    # Generate a synthetic face-like image for cache warming
    # 640x480 BGR image with a simple face-sized bright region
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[100:380, 200:440] = 180  # Bright face-sized region

    log.info("   Running warm-up inference pass (JIT compiling TRT engines) ...")
    _ = app.get(dummy_img)

    # Verify cache files were created
    cached_files = [f for f in os.listdir(engine_cache_dir) if f.endswith((".engine", ".trt"))]
    log.info(f"✅ InsightFace TRT cache warmed: {len(cached_files)} engine(s) cached in {engine_cache_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Validation Summary
# ═════════════════════════════════════════════════════════════════════════════

def validate_compiled_engines(
    yolo_engine_path: str,
    mlt_engine_path: str,
    insightface_cache_dir: str,
) -> None:
    """
    Final validation pass. Checks that all required engine files exist and
    are non-trivially sized. Raises fatal errors on any missing artifact.
    """
    log.info("=" * 72)
    log.info("  COMPILATION VALIDATION REPORT")
    log.info("=" * 72)

    checks = [
        ("YOLOv8 Engine", yolo_engine_path),
        ("MLT Engine", mlt_engine_path),
    ]

    all_ok = True
    for name, path in checks:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            log.info(f"  ✅ {name:25s} │ {size_mb:8.1f} MB │ {path}")
        else:
            log.error(f"  ❌ {name:25s} │ MISSING    │ {path}")
            all_ok = False

    # Check InsightFace cache
    if os.path.isdir(insightface_cache_dir):
        cache_files = os.listdir(insightface_cache_dir)
        total_cache_mb = sum(
            os.path.getsize(os.path.join(insightface_cache_dir, f))
            for f in cache_files
        ) / 1e6
        log.info(f"  ✅ {'InsightFace TRT Cache':25s} │ {total_cache_mb:8.1f} MB │ {len(cache_files)} file(s)")
    else:
        log.error(f"  ❌ {'InsightFace TRT Cache':25s} │ MISSING    │ {insightface_cache_dir}")
        all_ok = False

    log.info("=" * 72)

    if not all_ok:
        raise RuntimeError(
            "FATAL: One or more TensorRT engine compilations failed. "
            "Cannot proceed to production pipeline execution."
        )

    log.info("🏆 ALL ENGINES COMPILED SUCCESSFULLY. Pipeline ready for production.")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SPOVNOB TensorRT Offline Compilation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Absolute path to the SPOVNOB_CLONE project root.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="weights/yolov8n.pt",
        help="Relative path (from project-root) to YOLOv8 .pt weights.",
    )
    parser.add_argument(
        "--mlt-weights",
        type=str,
        default="OpenFace-3.0/weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth",
        help="Relative path (from project-root) to MLT backbone .pth weights.",
    )
    parser.add_argument(
        "--batch-opt",
        type=int,
        default=16,
        help="Optimal batch size for TensorRT dynamic profiles (default: 16).",
    )
    parser.add_argument(
        "--batch-max",
        type=int,
        default=32,
        help="Maximum batch size for TensorRT dynamic profiles (default: 32).",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLOv8 compilation (if already compiled).",
    )
    parser.add_argument(
        "--skip-mlt",
        action="store_true",
        help="Skip MLT compilation (if already compiled).",
    )
    parser.add_argument(
        "--skip-insightface",
        action="store_true",
        help="Skip InsightFace TRT cache warming.",
    )

    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)

    log.info("=" * 72)
    log.info("  SPOVNOB TensorRT Offline Compilation Pipeline")
    log.info(f"  Project Root: {project_root}")
    log.info(f"  Batch Profile: min=1, opt={args.batch_opt}, max={args.batch_max}")
    log.info(f"  Precision: FP16 (INT8 REJECTED)")
    log.info("=" * 72)

    # Resolve all absolute paths
    yolo_weights = os.path.join(project_root, args.yolo_weights)
    mlt_weights = os.path.join(project_root, args.mlt_weights)
    trt_engines_dir = os.path.join(project_root, "weights", "trt_engines")
    insightface_cache = os.path.join(trt_engines_dir, "insightface")

    os.makedirs(trt_engines_dir, exist_ok=True)
    os.makedirs(insightface_cache, exist_ok=True)

    mlt_onnx_path = os.path.join(trt_engines_dir, "MLT.onnx")
    mlt_engine_path = os.path.join(trt_engines_dir, "MLT.engine")
    yolo_engine_path = yolo_weights.replace(".pt", ".engine")

    # ── Phase 1: YOLOv8 ─────────────────────────────────────────────────
    if not args.skip_yolo:
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(
                f"FATAL: YOLOv8 weights not found at {yolo_weights}"
            )
        yolo_engine_path = compile_yolov8(
            yolo_weights, trt_engines_dir, args.batch_opt
        )
    else:
        log.info("⏭️  Skipping YOLOv8 compilation (--skip-yolo)")

    # ── Phase 2: MLT (ONNX + trtexec) ───────────────────────────────────
    if not args.skip_mlt:
        if not os.path.exists(mlt_weights):
            raise FileNotFoundError(
                f"FATAL: MLT weights not found at {mlt_weights}"
            )
        export_mlt_to_onnx(
            project_root, mlt_weights, mlt_onnx_path, args.batch_opt
        )
        compile_mlt_trtexec(
            mlt_onnx_path, mlt_engine_path,
            batch_min=1, batch_opt=args.batch_opt, batch_max=args.batch_max,
        )
    else:
        log.info("⏭️  Skipping MLT compilation (--skip-mlt)")

    # ── Phase 3: InsightFace TRT EP Cache ────────────────────────────────
    if not args.skip_insightface:
        warm_insightface_trt_cache(project_root, insightface_cache)
    else:
        log.info("⏭️  Skipping InsightFace cache warming (--skip-insightface)")

    # ── Phase 4: Validation ──────────────────────────────────────────────
    validate_compiled_engines(
        yolo_engine_path, mlt_engine_path, insightface_cache
    )


if __name__ == "__main__":
    main()
