"""Generic Stage-2 Pass 2-4 (manifest-driven): fit baseline on file_index 0, apply to
all, assemble global timeline (+ analyst report Pass 5). Run after a subject's cascade
shards finish. cwd=deception_detection, absolute spovnob_env python. argv: manifest.json."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, "app")
from pathlib import Path
from analytics.baseline_calibrator import BaselineCalibrator
from analytics.recording_assembler import assemble_recording
from audio_isolation.core.diarization_bridge import DiarizationBridge

man = json.load(open(sys.argv[1]))
tag, RID = man["subject"], man["recording_id"]
BASELINE_FIDX = man["baseline_file_index"]
N_CLIPS = len(man["clips"])
DIAR = f"/home/user1/Documents/Deception_Detection/audio_diarization/session/rec_{tag}/pipeline_output.json"
OUT = Path("pipeline_system_outputs")

bridge = DiarizationBridge.from_output_json(DIAR)
recdir = OUT / RID; recdir.mkdir(parents=True, exist_ok=True)
base_sid = f"{RID}_{BASELINE_FIDX:03d}"
cal = BaselineCalibrator()
stats = cal.fit(str(OUT / base_sid / f"{base_sid}_windowed_features.csv"))
stats.to_json(str(recdir / f"{RID}_baseline_stats.json"))
print(f"[{tag} fit] {stats.baseline_window_count} baseline windows, {len(stats.feature_means)} features")

inputs = []
for fidx in range(N_CLIPS):
    sid = f"{RID}_{fidx:03d}"; wcsv = OUT / sid / f"{sid}_windowed_features.csv"
    if not wcsv.exists():
        print(f"[skip fidx={fidx}: no windowed CSV]"); continue
    ccsv = OUT / sid / f"{sid}_calibrated_features.csv"
    cal.apply(str(wcsv), stats, str(ccsv))
    inputs.append({"file_index": fidx, "csv_path": str(ccsv), "offset_ms": bridge.file_offset_ms(fidx)})

rec_csv = recdir / f"{RID}_recording_calibrated.csv"
assemble_recording(inputs, str(rec_csv))
import pandas as pd
df = pd.read_csv(rec_csv)
base = df[df.file_index == BASELINE_FIDX]
print(f"[{tag} assembled] {len(inputs)} clips → {df.shape[0]} windows x {df.shape[1]} cols; "
      f"baseline median dev {base.deviation_magnitude.median():.2f} (healthy ~√n_feat; 0=degenerate); "
      f"blink populated {df['blink_rate'].notna().mean():.0%}")
try:
    from report.analyst_report import build_report
    build_report(str(recdir))
    print(f"[{tag}] analyst report written")
except Exception as e:
    print(f"[{tag}] report skipped: {e}")
