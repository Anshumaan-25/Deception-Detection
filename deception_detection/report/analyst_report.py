"""
Analyst report — data assembly + CLI.

Reads the per-recording artifacts (§9 of MASTER_REFERENCE):
    <rid>_recording_calibrated.csv      (required — the current end deliverable)
    <rid>_baseline_stats.json           (required — provenance + uncalibratable list)
    coupling_fit/fit_report.json        (optional — gates the coupling lane)
    coupling_attribution.csv            (optional — coupling-z per window)
    ../<tag>_<idx>/*_raw_features_30fps.csv  (optional — audio-active coverage)
    ELAN .eaf files                     (optional, VALIDATION MODE ONLY)

and produces a plain-JSON-able dict that render_html.py turns into a single
self-contained HTML file. Doctrine: attribution, never classification — the
report surfaces per-channel deviations and data quality; it never outputs a
verdict. ELAN ground truth is a validation-mode overlay, never a default.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# Windows are flagged for analyst review at the recording-wide 95th percentile
# of deviation_magnitude — the existing deviation_percentile column, no new
# statistics invented here.
FLAG_PERCENTILE = 0.95
# Channels validated against ELAN ground truth (RESULTS.md / RESULTS_PRODUCTION.md)
# plus their immediate family; only those present in the CSV are plotted.
TRACE_CHANNELS = [
    "AU12_velocity_tremor_band_power", "AU12_velocity_max", "AU12_mean",
    "left_hand_face_distance_min", "right_wrist_velocity_max",
    "left_wrist_velocity_max", "head_pitch_tremor_band_power",
    "gaze_x_mean", "gaze_entropy", "gaze_velocity_var",
    "blink_rate", "ear_mean",
    "macro_motion_energy_mean", "postural_stillness_mean",
    "acoustic_energy_rms", "prosodic_velocity", "acoustic_volatility",
    "mismatch_ratio",
]
# Report-local grouping of WINDOWED columns into node families (prefix rules).
# This mirrors the 11-node frame-level graph in spirit; the windowed schema has
# its own column names, so the mapping lives here, not in graph_spec.
NODE_GROUPS = [
    ("head_pose",   ("head_",)),
    ("gaze",        ("gaze_",)),
    ("blink",       ("blink_rate", "ear_")),
    ("au_upper",    ("AU1_", "AU2_", "AU4_")),
    ("au_mid",      ("AU6_", "AU9_")),
    ("au_mouth",    ("AU12", "AU25", "AU26")),
    ("hand_left",   ("left_wrist_", "left_hand_")),
    ("hand_right",  ("right_wrist_", "right_hand_")),
    ("body",        ("motion_energy", "macro_motion", "postural_", "nose_")),
    ("voice",       ("wavlm_", "acoustic_", "prosodic_", "vocal_")),
    ("congruence",  ("mismatch_", "silent_", "diarizer_")),
]
_METADATA_LIKE = {
    "window_id", "start_time_ms", "end_time_ms", "frame_count",
    "cumulative_confidence", "blink_count", "emotion_label_mode",
    "context_phase", "question_id", "phase_elapsed_ms", "file_index",
    "clip_window_id", "deviation_magnitude", "deviation_percentile",
    "target_ground_truth",
}


def _nan_to_none(seq):
    return [None if (v is None or (isinstance(v, float) and not np.isfinite(v)))
            else round(float(v), 4) for v in seq]


def _feature_cols(df):
    return [c for c in df.columns
            if c not in _METADATA_LIKE and df[c].dtype.kind in "fi"]


def _group_of(col):
    for name, prefixes in NODE_GROUPS:
        if any(col.startswith(p) for p in prefixes):
            return name
    return None


def parse_eaf(path):
    """ELAN .eaf → [(label, start_ms, end_ms)]. Validation mode only."""
    root = ET.parse(path).getroot()
    slots = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE"))
             for t in root.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text,
             slots[a.get("TIME_SLOT_REF1")], slots[a.get("TIME_SLOT_REF2")])
            for a in root.iter("ALIGNABLE_ANNOTATION")]


def _coupling_status(recording_dir):
    """(lane_df | None, human-readable status). The lane renders ONLY when the
    attribution file exists AND the fit passed the pre-registered 0.90 gate."""
    fit_json = Path(recording_dir) / "coupling_fit" / "fit_report.json"
    attr_csv = Path(recording_dir) / "coupling_attribution.csv"
    if not attr_csv.exists():
        return None, "coupling attribution not run for this recording"
    if not fit_json.exists():
        return None, "coupling_attribution.csv present but fit_report.json missing — lane withheld"
    ratio = json.loads(fit_json.read_text()).get("coupling_ratio", 1.0)
    if ratio >= 0.90:
        return None, (f"coupling fit DEGENERATE (ratio {ratio:.3f} ≥ 0.90 gate) — "
                      f"attributions are noise for this subject; lane withheld")
    return pd.read_csv(attr_csv), f"coupling fit healthy (ratio {ratio:.3f})"


def _audio_active_fraction(clips_parent, tag, fidx):
    """Best-effort per-clip audio coverage from the raw 30 fps CSV; None if absent."""
    for pat in (f"{tag}_{fidx:03d}/*_raw_features_30fps.csv",):
        hits = globmod.glob(str(Path(clips_parent) / pat))
        if hits:
            try:
                col = pd.read_csv(hits[0], usecols=["is_audio_active"])["is_audio_active"]
                return float((col == 1.0).mean())
            except Exception:
                return None
    return None


def build_report_data(recording_dir, *, recording_id=None, elan_dir=None,
                      clips_tag=None, clips_parent=None, generated_on=None):
    rec = Path(recording_dir)
    cal_hits = sorted(rec.glob("*_recording_calibrated.csv"))
    if not cal_hits:
        raise FileNotFoundError(f"no *_recording_calibrated.csv in {rec}")
    cal_csv = cal_hits[0]
    rid = recording_id or cal_csv.name.replace("_recording_calibrated.csv", "")

    df = pd.read_csv(cal_csv)
    feats = _feature_cols(df)
    stats_path = rec / f"{rid}_baseline_stats.json"
    baseline = json.loads(stats_path.read_text()) if stats_path.exists() else None
    uncalibratable = ([k for k, v in baseline["feature_stds"].items() if v is None]
                      if baseline else [])

    if "file_index" not in df.columns:
        df["file_index"] = 0
    if "deviation_percentile" not in df.columns:
        df["deviation_percentile"] = df["deviation_magnitude"].rank(pct=True, na_option="keep")

    coupling_df, coupling_status = _coupling_status(rec)
    coupling_nodes = ([c[len("coupling_z_"):] for c in coupling_df.columns
                       if c.startswith("coupling_z_") and c not in
                       ("coupling_z_global", "coupling_z_max")]
                      if coupling_df is not None else [])

    trace_cols = [c for c in TRACE_CHANNELS if c in df.columns]
    tag = clips_tag or rid
    parent = Path(clips_parent) if clips_parent else rec.parent

    # ── per-channel coverage (quality panel) ─────────────────────────────────
    coverage = {c: float(df[c].notna().mean()) for c in feats}
    dead = sorted(c for c, v in coverage.items() if v == 0.0)
    low = sorted((c for c, v in coverage.items() if 0.0 < v < 0.5))

    # baseline sanity: the baseline clip (file_index 0) should sit at z≈0/1 and
    # median deviation_magnitude ≈ sqrt(#features) (the rec_ca lesson: ~0 is the
    # DEGENERATE signature, not health)
    b = df[df.file_index == 0]
    expected_dev = float(np.sqrt(len(feats)))
    baseline_sanity = None
    if len(b):
        bz = b[feats].to_numpy(dtype=float)
        baseline_sanity = {
            "mean_abs_z": round(float(np.nanmean(np.abs(bz))), 3),
            "median_dev": round(float(b["deviation_magnitude"].median()), 2),
            "expected_dev_sqrt_f": round(expected_dev, 2),
            "degenerate": bool(b["deviation_magnitude"].median() < 0.25 * expected_dev),
        }

    # ── per-clip payloads ────────────────────────────────────────────────────
    clips, node_rows = [], []
    for fidx, cdf in df.groupby("file_index", sort=True):
        cdf = cdf.sort_values("start_time_ms")
        t = cdf["start_time_ms"].to_numpy(dtype=float)
        pct = cdf["deviation_percentile"].to_numpy(dtype=float)
        flags = []
        for _, row in cdf[cdf["deviation_percentile"] >= FLAG_PERCENTILE].iterrows():
            zrow = row[feats].astype(float)
            top = zrow.abs().nlargest(5)
            flags.append({
                "t": float(row["start_time_ms"]),
                "pct": round(float(row["deviation_percentile"]), 3),
                "top": [{"ch": ch, "z": round(float(zrow[ch]), 2)} for ch in top.index
                        if np.isfinite(zrow[ch])],
            })
        entry = {
            "file_index": int(fidx),
            "is_baseline": int(fidx) == 0,
            "n_windows": int(len(cdf)),
            "t": _nan_to_none(t),
            "pct": _nan_to_none(pct),
            "traces": {c: _nan_to_none(cdf[c].to_numpy(dtype=float)) for c in trace_cols},
            "flags": flags,
            "audio_active_frac": _audio_active_fraction(parent, tag, int(fidx)),
            "mean_confidence": (round(float(cdf["cumulative_confidence"].mean()), 3)
                                if "cumulative_confidence" in cdf else None),
            "coupling": None, "elan": None,
        }
        if coupling_df is not None and "file_index" in coupling_df.columns:
            cc = coupling_df[coupling_df.file_index == fidx].sort_values("start_time_ms")
            if len(cc):
                entry["coupling"] = {
                    "t": _nan_to_none(cc["start_time_ms"].to_numpy(dtype=float)),
                    "nodes": {n: _nan_to_none(cc[f"coupling_z_{n}"].to_numpy(dtype=float))
                              for n in coupling_nodes},
                }
        if elan_dir:   # VALIDATION MODE ONLY — never a production default
            hits = globmod.glob(str(Path(elan_dir) / f"*C{int(fidx)+1:03d}*.eaf"))
            if hits:
                entry["elan"] = [[lab, int(s), int(e)] for lab, s, e in parse_eaf(hits[0])]
        clips.append(entry)

        # direction-aware node table (median signed z per group)
        cells = []
        for gname, _ in NODE_GROUPS:
            gcols = [c for c in feats if _group_of(c) == gname]
            if not gcols:
                cells.append({"group": gname, "med_z": None, "top_ch": None, "top_z": None})
                continue
            gz = cdf[gcols].astype(float)
            med = gz.median()
            top_ch = med.abs().idxmax() if med.notna().any() else None
            cells.append({
                "group": gname,
                "med_z": (None if not med.notna().any()
                          else round(float(med.median()), 2)),
                "top_ch": top_ch,
                "top_z": (round(float(med[top_ch]), 2)
                          if top_ch is not None and np.isfinite(med[top_ch]) else None),
            })
        node_rows.append({"file_index": int(fidx), "cells": cells})

    return {
        "meta": {
            "recording_id": rid,
            "generated_on": generated_on or "",
            "source_csv": cal_csv.name,
            "n_clips": len(clips),
            "n_windows": int(len(df)),
            "n_features": len(feats),
            "flag_percentile": FLAG_PERCENTILE,
            "baseline": {
                "window_count": baseline["baseline_window_count"] if baseline else None,
                "source_csv": os.path.basename(baseline["source_csv"]) if baseline else None,
                "uncalibratable": uncalibratable,
            },
            "coupling_status": coupling_status,
            "validation_mode": bool(elan_dir),
        },
        "quality": {
            "dead_channels": dead,
            "low_coverage": [{"ch": c, "coverage": round(coverage[c], 3)} for c in low],
            "trace_coverage": {c: round(coverage[c], 3) for c in trace_cols},
            "baseline_sanity": baseline_sanity,
        },
        "node_groups": [g for g, _ in NODE_GROUPS],
        "node_table": node_rows,
        "clips": clips,
    }


def generate_report(recording_dir, output_html=None, **kw):
    from .render_html import render
    data = build_report_data(recording_dir, **kw)
    out = Path(output_html) if output_html else \
        Path(recording_dir) / f"{data['meta']['recording_id']}_analyst_report.html"
    out.write_text(render(data))
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the per-recording analyst report (self-contained HTML).")
    ap.add_argument("recording_dir")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--elan-dir", default=None,
                    help="VALIDATION MODE: directory of .eaf files (overlays ground truth)")
    ap.add_argument("--clips-tag", default=None, help="clip session dir prefix if ≠ recording id")
    ap.add_argument("--clips-parent", default=None, help="parent dir of clip session dirs")
    ap.add_argument("--recording-id", default=None)
    ap.add_argument("--generated-on", default=None, help="date stamp shown in the header")
    a = ap.parse_args()
    p = generate_report(a.recording_dir, a.out, recording_id=a.recording_id,
                        elan_dir=a.elan_dir, clips_tag=a.clips_tag,
                        clips_parent=a.clips_parent, generated_on=a.generated_on)
    print(f"[report] wrote {p}")
