"""Cascade-only (canonicalization done separately with system ffmpeg).
Per-clip cascade, calibrate=False. Run with cwd=deception_detection under
absolute spovnob_env python."""
import sys, os, time, warnings, faulthandler, traceback
warnings.filterwarnings("ignore")
faulthandler.dump_traceback_later(1800, repeat=True, exit=False)
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, "app")
from main_pipeline import MultimodalProductionOrchestrator

SCRATCH = "deception_detection/pipeline_system_outputs/GT_SUBJECTA_20260708"
CANON = f"{SCRATCH}/canonical"
OUT   = f"{SCRATCH}/outputs"

# clip list from argv (parallel shards), else default full order
CLIPS = sys.argv[1:] if len(sys.argv) > 1 else [
    "00_baseline", "06_interview", "07_interview",
    "02_interview", "03_interview", "04_interview", "05_interview", "01_interview"]

orch = MultimodalProductionOrchestrator(output_root=OUT, yolo_path="Yolo_v8/weights/yolov8n.pt")
print("[orchestrator booted]", flush=True)
done = []
for base in CLIPS:
    v = f"{CANON}/{base}_canonical.mp4"
    a = f"{CANON}/{base}_hubert.wav"
    if not (os.path.exists(v) and os.path.exists(a)):
        print(f"[skip {base}: canonical missing]", flush=True); continue
    t = time.time()
    print(f"\n=== CASCADE {base} START ===", flush=True)
    try:
        orch.process_video_session(v, a, base, pyannote_segments=None, calibrate=False)
        print(f"=== CASCADE {base} DONE {time.time()-t:.1f}s ===", flush=True)
        done.append(base)
    except Exception:
        print(f"=== CASCADE {base} FAIL {time.time()-t:.1f}s ===\n{traceback.format_exc()}", flush=True)

print(f"\n### ALL DONE: {len(done)}/{len(CLIPS)} clips → {done} ###", flush=True)
