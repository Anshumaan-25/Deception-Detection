"""Stage-2 Pass 2-4 (faithful to process_recording_session): fit baseline on
file_index 0, apply to all clips, assemble on the global timeline. Run after all
cascade shards finish. cwd=deception_detection, absolute spovnob_env python."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, "app")
from pathlib import Path
from analytics.baseline_calibrator import BaselineCalibrator
from analytics.recording_assembler import assemble_recording
from audio_isolation.core.diarization_bridge import DiarizationBridge

DIAR = "/home/user1/Documents/Deception_Detection/audio_diarization/session/rec_subjectA/pipeline_output.json"
OUT = Path("pipeline_system_outputs")
RID = "REC_SUBJECTA"
BASELINE_FIDX = 0

bridge = DiarizationBridge.from_output_json(DIAR)
recdir = OUT / RID
recdir.mkdir(parents=True, exist_ok=True)

# Pass 2: fit on the baseline clip's windowed CSV
base_sid = f"{RID}_{BASELINE_FIDX:03d}"
base_csv = OUT / base_sid / f"{base_sid}_windowed_features.csv"
cal = BaselineCalibrator()
stats = cal.fit(str(base_csv))
stats_path = recdir / f"{RID}_baseline_stats.json"
stats.to_json(str(stats_path))
print(f"[fit] {stats.baseline_window_count} baseline windows, {len(stats.feature_means)} features")

# Pass 3: apply to every clip present
inputs = []
for fidx in range(8):
    sid = f"{RID}_{fidx:03d}"
    wcsv = OUT / sid / f"{sid}_windowed_features.csv"
    if not wcsv.exists():
        print(f"[skip fidx={fidx}: no windowed CSV]"); continue
    ccsv = OUT / sid / f"{sid}_calibrated_features.csv"
    cal.apply(str(wcsv), stats, str(ccsv))
    inputs.append({"file_index": fidx, "csv_path": str(ccsv),
                   "offset_ms": bridge.file_offset_ms(fidx)})

# Pass 4: assemble on the global timeline
rec_csv = recdir / f"{RID}_recording_calibrated.csv"
assemble_recording(inputs, str(rec_csv))
print(f"[assembled] {len(inputs)} clips -> {rec_csv}")

import pandas as pd
df = pd.read_csv(rec_csv)
print(f"[deliverable] {df.shape[0]} windows x {df.shape[1]} cols")
print("  global timeline:", df.start_time_ms.min(), "->", df.start_time_ms.max(), "ms")
print("  deviation_percentile range:", round(df.deviation_percentile.min(),3), "-", round(df.deviation_percentile.max(),3))
base = df[df.file_index == BASELINE_FIDX]
print(f"  baseline clip median deviation_magnitude: {base.deviation_magnitude.median():.3f} (should be near 0)")
for c in ["blink_count","blink_rate","ear_mean","ear_var"]:
    print(f"  {c}: {df[c].notna().mean():.0%} populated" if c in df.columns else f"  {c}: MISSING")
