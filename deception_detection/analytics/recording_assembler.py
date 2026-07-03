"""
Recording Assembler — recording-level global timeline packaging (Phase A)
=========================================================================
Concatenates per-clip calibrated window CSVs into one recording-level CSV on
a single shared clock.

This is deliberately a *presentation/packaging* step with no statistical
consequences: calibration already happened per clip against the dedicated
baseline clip's stats (see analytics/baseline_calibrator.py fit/apply), so an
offset mistake here can mis-plot a timeline but can never corrupt a z-score.

Per-clip windowing means windows are hard-broken at clip boundaries by
construction — no window in the assembled CSV ever mixes frames from two
physically different files.

The one statistic computed here is deviation_percentile: a percentile rank is
only meaningful over one shared population, so it is ranked over the whole
assembled recording rather than per clip.

Offsets come from the audio-diarization pipeline's own file_offset_ms
(cumulative duration of preceding files in canonical order, via
DiarizationBridge.file_offset_ms), so audio and video share one anchoring
scheme.

Pure pandas — no GPU, no torch. Self-test: tests/verify_recording_calibration.py
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger("Recording_Assembler")


class RecordingAssemblyError(RuntimeError):
    """Raised when the assembly inputs are unusable (empty, missing files)."""


def assemble_recording(clips: list, output_csv_path: str) -> str:
    """
    Build the recording-level calibrated CSV from per-clip calibrated CSVs.

    Args:
        clips: list of dicts, one per clip:
            {"file_index": int,   # diarization file_index
             "csv_path": str,     # per-clip calibrated CSV (apply() output)
             "offset_ms": int}    # clip's offset on the recording clock
        output_csv_path: destination for the assembled CSV.

    Per clip: adds provenance columns (file_index, clip_window_id), rebases
    start_time_ms / end_time_ms by offset_ms. Then concatenates in file_index
    order, renumbers window_id sequentially across the recording, and computes
    deviation_percentile over the combined population.

    Returns the output path.
    """
    if not clips:
        raise RecordingAssemblyError("No clips to assemble.")

    frames = []
    for clip in sorted(clips, key=lambda c: int(c["file_index"])):
        csv_path = Path(clip["csv_path"])
        if not csv_path.exists():
            raise RecordingAssemblyError(f"Per-clip CSV missing: {csv_path}")
        df = pd.read_csv(csv_path)
        offset = int(clip["offset_ms"])
        df["file_index"] = int(clip["file_index"])
        df["clip_window_id"] = df["window_id"]
        df["start_time_ms"] = df["start_time_ms"] + offset
        df["end_time_ms"] = df["end_time_ms"] + offset
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["file_index", "start_time_ms"], kind="stable"
    ).reset_index(drop=True)
    combined["window_id"] = range(len(combined))

    if "deviation_magnitude" in combined.columns:
        combined["deviation_percentile"] = (
            combined["deviation_magnitude"].rank(pct=True, na_option="keep")
        )

    combined.to_csv(output_csv_path, index=False)
    logger.info(
        f"✅ Recording assembled: {len(combined)} windows across "
        f"{len(frames)} clip(s) → {output_csv_path}"
    )
    return str(output_csv_path)
