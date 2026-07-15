"""Generic Stage-2 PRODUCTION cascade (manifest-driven). Consumes SPOVNOB
session/rec_<subject>/pipeline_output.json + canonical media → REC_SUBJECT{X}_00N.
cwd=deception_detection, absolute spovnob_env python. argv: manifest.json [clip bases...]."""
import sys, os, time, json, warnings, faulthandler, traceback
import torch  # FIRST: loads CUDA-12 libs RTLD_GLOBAL so onnxruntime's CUDA EP binds
warnings.filterwarnings("ignore")
faulthandler.dump_traceback_later(3000, repeat=True, exit=False)
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, "app")
from audio_isolation.core.diarization_bridge import DiarizationBridge
from main_pipeline import MultimodalProductionOrchestrator

man = json.load(open(sys.argv[1]))
tag, RID, CANON = man["subject"], man["recording_id"], man["canonical_dir"]
DIAR = f"/home/user1/Documents/Deception_Detection/audio_diarization/session/rec_{tag}/pipeline_output.json"
OUT = "pipeline_system_outputs"
bases = sys.argv[2:] or [c["base"] for c in man["clips"]]

bridge = DiarizationBridge.from_output_json(DIAR)
orch = MultimodalProductionOrchestrator(output_root=OUT, yolo_path="Yolo_v8/weights/yolov8n.pt")
print(f"[{tag} orchestrator booted]", flush=True)
done = []
for base in bases:
    mp4 = f"{CANON}/{base}_canonical.mp4"; wav = f"{CANON}/{base}_hubert.wav"
    fidx = bridge.index_for_clip(mp4)
    segs = bridge.segments_for(fidx, clock="local", offset_ms=0)
    sid = f"{RID}_{fidx:03d}"; t = time.time()
    print(f"\n=== {tag} {base} fidx={fidx} sid={sid} segs={len(segs)} START ===", flush=True)
    try:
        orch.process_video_session(mp4, wav, sid, pyannote_segments=segs, calibrate=False)
        print(f"=== {tag} {base} DONE {time.time()-t:.1f}s ===", flush=True); done.append(base)
    except Exception:
        print(f"=== {tag} {base} FAIL {time.time()-t:.1f}s ===\n{traceback.format_exc()}", flush=True)
print(f"\n### {tag} SHARD DONE: {done} ###", flush=True)
