"""Stage-2 PRODUCTION cascade (Pass-1 of process_recording_session, parallelized).
Real diarization segments from SPOVNOB -> is_audio_active masks non-target audio;
blink/EAR alive (seam fix). Run with cwd=deception_detection, absolute spovnob_env python.
argv = clip base names for this shard."""
import sys, os, time, warnings, faulthandler, traceback
warnings.filterwarnings("ignore")
faulthandler.dump_traceback_later(2400, repeat=True, exit=False)
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, "app")
from audio_isolation.core.diarization_bridge import DiarizationBridge
from main_pipeline import MultimodalProductionOrchestrator

DIAR = "/home/user1/Documents/Deception_Detection/audio_diarization/session/rec_subjectA/pipeline_output.json"
CANON = "pipeline_system_outputs/GT_SUBJECTA_20260708/canonical"
OUT = "pipeline_system_outputs"
CLIPS = sys.argv[1:]

bridge = DiarizationBridge.from_output_json(DIAR)
orch = MultimodalProductionOrchestrator(output_root=OUT, yolo_path="Yolo_v8/weights/yolov8n.pt")
print("[orchestrator booted]", flush=True)
done = []
for base in CLIPS:
    mp4 = f"{CANON}/{base}_canonical.mp4"
    wav = f"{CANON}/{base}_hubert.wav"
    fidx = bridge.index_for_clip(mp4)
    segs = bridge.segments_for(fidx, clock="local", offset_ms=0)
    sid = f"REC_SUBJECTA_{fidx:03d}"
    t = time.time()
    print(f"\n=== {base} fidx={fidx} sid={sid} target_segs={len(segs)} START ===", flush=True)
    try:
        orch.process_video_session(mp4, wav, sid, pyannote_segments=segs, calibrate=False)
        print(f"=== {base} DONE {time.time()-t:.1f}s ===", flush=True)
        done.append(base)
    except Exception:
        print(f"=== {base} FAIL {time.time()-t:.1f}s ===\n{traceback.format_exc()}", flush=True)
print(f"\n### SHARD DONE: {done} ###", flush=True)
