"""
Cross-subject replication scorecard — does SubjectA's per-channel signal generalize?

Consumes each subject's ALREADY-PROCESSED recording (the per-recording
`*_recording_calibrated.csv` produced by `process_recording_session`) plus their
ELAN directory, scores every calibrated channel per subject (|z| AUC, Lie vs
Truth, pure windows only), and aggregates into one channel × subject scorecard
with replication verdicts. ELAN labels are used for SCORING ONLY — never
calibration, never training (doctrine unchanged).

════════════════════════════════════════════════════════════════════════════════
PRE-REGISTERED REPLICATION CRITERIA — locked 2026-07-10, BEFORE any subject
beyond SubjectA was seen. Changing these after looking at new-subject results
invalidates the exercise (same discipline that falsified the ST-GAE).
  ADEQUACY   a subject counts for a channel only with ≥ MIN_WINDOWS(30) pure
             Lie AND ≥30 pure Truth windows with non-NaN values on that channel
  R1 REPLICATES        |z| AUC ≥ 0.60 in ≥ 2/3 of adequate subjects,
                       with ≥ 2 adequate subjects
  R2 DIRECTION-STABLE  a subject expresses a direction only when
                       |median lie z − median truth z| ≥ 0.25 (noise signs do
                       NOT count — with 3 subjects, "2 of 3 share a sign" is
                       pigeonhole-guaranteed for pure noise); R2 holds when
                       ≥ 2/3 of adequate subjects express a direction AND all
                       expressed directions share one sign
  Verdict per channel:
    REPLICATES        R1 and R2
    DIRECTION-ONLY    R2 but not R1 (consistent direction, weak magnitude)
    SUBJECT-SPECIFIC  ≥2 adequate subjects, neither R1 nor R2 holds broadly
                      but some single subject clears 0.60 (a quirk, not a finding)
    NO-SIGNAL         ≥2 adequate subjects, nothing clears 0.60 anywhere
    INSUFFICIENT-DATA <2 adequate subjects
════════════════════════════════════════════════════════════════════════════════

TRACKED_CHANNELS below are SubjectA's validated findings (RESULTS.md /
RESULTS_PRODUCTION.md) — the scorecard reports them first, because "does THIS
set replicate?" is the question the new corpus exists to answer.

Usage:
    python -m multisubject.replication_scorecard manifest.json [--out DIR]
manifest.json:
    {"subjects": [
        {"name": "SubjectA", "recording_dir": "pipeline_system_outputs/REC_SUBJECTA",
         "elan_dir": "/path/to/annotated"},
        ...]}
Clip↔eaf convention: interview file_index f ↔ glob *C{f+1:03d}*.eaf (as SubjectA).
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.baseline_calibrator import NON_FEATURE_COLUMNS, parse_baseline_file_index

MIN_WINDOWS = 30          # pre-registered adequacy floor (per class, per subject)
REPLICATE_AUC = 0.60      # pre-registered magnitude bar
FRACTION = 2.0 / 3.0      # pre-registered subject-agreement fraction
DIRECTION_MIN = 0.25      # pre-registered: min |median shift| (z) to express a direction
PURE_OVERLAP = 0.6        # window counts as labeled iff ≥60% covered (as gt_score)

TRACKED_CHANNELS = [      # SubjectA's validated set — reported first
    "AU12_velocity_tremor_band_power", "AU12_velocity_max",
    "left_hand_face_distance_min", "right_wrist_velocity_max",
    "head_pitch_tremor_band_power", "gaze_x_mean", "gaze_entropy",
    "blink_rate", "ear_mean",
]


def parse_eaf(p):
    r = ET.parse(p).getroot()
    s = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE")) for t in r.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text, s[a.get("TIME_SLOT_REF1")],
             s[a.get("TIME_SLOT_REF2")]) for a in r.iter("ALIGNABLE_ANNOTATION")]


def label_window(s, e, iv):
    best, bo = "Unlabeled", 0.0
    for lab, a, b in iv:
        ov = max(0.0, min(e, b) - max(s, a))
        if ov > bo:
            bo, best = ov, lab
    return best, bo / max(1.0, e - s)


def auc(pos, neg):
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_labeled_windows(recording_dir, elan_dir):
    """One subject's calibrated windows with gtruth/ov columns (pure windows only)."""
    rec = Path(recording_dir)
    hits = sorted(rec.glob("*_recording_calibrated.csv"))
    if not hits:
        raise FileNotFoundError(f"no *_recording_calibrated.csv in {rec}")
    df = pd.read_csv(hits[0])
    # recover the baseline clip index (NOT assumed 0) — it is never labeled/scored
    stats_hits = sorted(rec.glob("*_baseline_stats.json"))
    baseline_idx = int(df["file_index"].min())
    if stats_hits:
        src = json.loads(stats_hits[0].read_text()).get("source_csv", "")
        baseline_idx = parse_baseline_file_index(src, baseline_idx)
    frames = []
    for fidx, cdf in df.groupby("file_index", sort=True):
        if int(fidx) == baseline_idx:
            continue                                   # baseline: never labeled
        eafs = globmod.glob(str(Path(elan_dir) / f"*C{int(fidx)+1:03d}*.eaf"))
        if not eafs:
            continue
        iv = parse_eaf(eafs[0])
        cdf = cdf.copy()
        # The assembled *_recording_calibrated.csv carries GLOBAL window times
        # (rebased by file_offset_ms during assembly), but each ELAN .eaf is
        # LOCAL to its clip. Rebase back to clip-local before matching, else
        # large offsets (SubjectA) yield ZERO overlap and small ones (SubjectB)
        # yield SHIFTED/mislabeled overlap. Per clip, the first window sits at
        # local 0, so the group's min start == file_offset_ms exactly.
        off = float(cdf["start_time_ms"].min())
        labs, ovs = zip(*[label_window(r.start_time_ms - off, r.end_time_ms - off, iv)
                          for r in cdf.itertuples()])
        cdf["gtruth"], cdf["ov"] = labs, ovs
        frames.append(cdf)
    if not frames:
        return pd.DataFrame()
    allw = pd.concat(frames, ignore_index=True)
    return allw[allw["ov"] >= PURE_OVERLAP].copy()


def score_subject(pure):
    """Per-channel: (auc, direction, n_lie, n_truth), scored WITHIN-CLIP.

    The magnitude signal in this project is within-clip, not pooled: pooling |z|
    across clips is confounded by clip-level offsets and collapses even SubjectA's
    validated channels to chance (positive-control failure: pooled AU12 0.50 vs
    within-clip-06 0.68). So — matching the 07-08 method and coupling_evaluate —
    AUC is computed on the **within-clip percentile of |z|** (rank each channel
    within its own clip, then pool), and direction on **within-clip-centered signed
    z** (subtract each clip's median, keeping z-units so DIRECTION_MIN still applies).
    The pre-registered thresholds (MIN_WINDOWS, REPLICATE_AUC, DIRECTION_MIN, the
    verdict logic) are UNCHANGED — only the clip-confound control is corrected."""
    feats = [c for c in pure.columns
             if c not in NON_FEATURE_COLUMNS + ["gtruth", "ov", "file_index",
                                                "clip_window_id"]
             and pure[c].dtype.kind in "fi"]
    grp = pure.groupby("file_index")
    is_lie = (pure.gtruth == "Lie").to_numpy()
    is_tru = (pure.gtruth == "Truth").to_numpy()
    out = {}
    for c in feats:
        pct = grp[c].transform(lambda s: s.abs().rank(pct=True)).to_numpy(dtype=float)
        cen = (pure[c] - grp[c].transform("median")).to_numpy(dtype=float)
        raw = pure[c].to_numpy(dtype=float)
        n_l = int(np.isfinite(raw[is_lie]).sum()); n_t = int(np.isfinite(raw[is_tru]).sum())
        a = auc(pct[is_lie], pct[is_tru])          # within-clip |z| percentile
        d = (np.nanmedian(cen[is_lie]) - np.nanmedian(cen[is_tru])
             if n_l and n_t else np.nan)           # within-clip-centered signed z
        # a direction is only EXPRESSED past the pre-registered shift floor —
        # the sign of a noise-sized median difference carries no information
        expressed = np.isfinite(d) and abs(d) >= DIRECTION_MIN
        out[c] = {"auc": a, "dir": (float(np.sign(d)) if expressed else 0.0),
                  "n_lie": n_l, "n_truth": n_t}
    return out


def verdict_for_channel(per_subject):
    """per_subject: {name: {auc, dir, n_lie, n_truth}} → (verdict, adequate_names)."""
    adequate = {n: r for n, r in per_subject.items()
                if r["n_lie"] >= MIN_WINDOWS and r["n_truth"] >= MIN_WINDOWS
                and np.isfinite(r["auc"])}
    if len(adequate) < 2:
        return "INSUFFICIENT-DATA", list(adequate)
    n = len(adequate)
    hi = sum(1 for r in adequate.values() if r["auc"] >= REPLICATE_AUC)
    r1 = hi >= np.ceil(FRACTION * n)
    dirs = [r["dir"] for r in adequate.values() if r["dir"] != 0.0]
    r2 = (len(dirs) >= np.ceil(FRACTION * n)
          and (all(d > 0 for d in dirs) or all(d < 0 for d in dirs)))
    if r1 and r2:
        return "REPLICATES", list(adequate)
    if r2 and not r1:
        return "DIRECTION-ONLY", list(adequate)
    if hi > 0:
        return "SUBJECT-SPECIFIC", list(adequate)
    return "NO-SIGNAL", list(adequate)


def build_scorecard(manifest):
    subjects = {}
    for s in manifest["subjects"]:
        pure = load_labeled_windows(s["recording_dir"], s["elan_dir"])
        if pure.empty:
            print(f"[scorecard] ⚠ {s['name']}: no labeled pure windows — skipped")
            continue
        subjects[s["name"]] = score_subject(pure)
        nl = int((pure.gtruth == "Lie").sum()); nt = int((pure.gtruth == "Truth").sum())
        print(f"[scorecard] {s['name']}: {nl} Lie / {nt} Truth pure windows, "
              f"{len(subjects[s['name']])} channels")
    channels = sorted({c for sc in subjects.values() for c in sc})
    rows = []
    for c in channels:
        per = {n: sc[c] for n, sc in subjects.items() if c in sc}
        v, adequate = verdict_for_channel(per)
        row = {"channel": c, "verdict": v, "n_adequate": len(adequate),
               "tracked": c in TRACKED_CHANNELS}
        for n in subjects:
            r = per.get(n)
            row[f"auc_{n}"] = round(r["auc"], 3) if r and np.isfinite(r["auc"]) else None
            row[f"dir_{n}"] = ({1.0: "+", -1.0: "−", 0.0: "·"}.get(r["dir"])
                               if r else None)
        rows.append(row)
    return pd.DataFrame(rows), subjects


def main(manifest_path, out_dir=None):
    manifest = json.loads(Path(manifest_path).read_text())
    card, subjects = build_scorecard(manifest)
    if card.empty:
        print("[scorecard] nothing to score.")
        return None
    out = Path(out_dir or Path(manifest_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "replication_scorecard.csv"
    card.sort_values(["tracked", "verdict", "channel"],
                     ascending=[False, True, True]).to_csv(csv_path, index=False)

    names = list(subjects)
    print(f"\n═══ TRACKED SubjectA channels (the question this corpus answers) ═══")
    print(f"{'channel':38s} {'verdict':18s} " + " ".join(f"{n[:10]:>10s}" for n in names))
    for _, r in card[card.tracked].iterrows():
        aucs = " ".join(f"{r[f'auc_{n}']:>10.3f}" if r[f"auc_{n}"] is not None
                        else f"{'–':>10s}" for n in names)
        print(f"{r.channel:38s} {r.verdict:18s} {aucs}")
    counts = card.verdict.value_counts().to_dict()
    print(f"\n═══ all {len(card)} channels ═══  " +
          "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    top = card[card.verdict == "REPLICATES"].copy()
    if len(top):
        print("REPLICATES:", ", ".join(top.channel.head(20)))
    print(f"[scorecard] wrote {csv_path}")
    return str(csv_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-subject replication scorecard (pre-registered criteria).")
    ap.add_argument("manifest", help="JSON: {subjects: [{name, recording_dir, elan_dir}]}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.manifest, a.out)
