"""ELAN ground-truth scorer (validation-only; labels never touch calibration).
Fit BaselineCalibrator on 00_baseline, apply to each interview, overlay ELAN
Truth/Lie/Neutral labels per window, measure deviation separation.
Run with cwd=deception_detection (any env with pandas/numpy)."""
import sys, os, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
sys.path.insert(0, "analytics")
from baseline_calibrator import BaselineCalibrator

SCRATCH = "deception_detection/pipeline_system_outputs/GT_SUBJECTA_20260708"
OUT = f"{SCRATCH}/outputs"
ANN = "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"

# video base -> B04C prefix (confirmed sequential rename: index N -> B04C00(N+1))
def b04c_for(base):
    idx = int(base.split("_")[0])           # 00..07
    return f"B04C{idx+1:03d}"                # 00->B04C001 ... 07->B04C008

def eaf_for(base):
    hits = glob.glob(f"{ANN}/{b04c_for(base)}_*.eaf")
    return hits[0] if hits else None

def parse_eaf(p):
    r = ET.parse(p).getroot()
    slots = {ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE")) for ts in r.iter("TIME_SLOT")}
    out = []
    for a in r.iter("ALIGNABLE_ANNOTATION"):
        out.append((a.find("ANNOTATION_VALUE").text,
                    slots[a.get("TIME_SLOT_REF1")], slots[a.get("TIME_SLOT_REF2")]))
    return out

def label_window(s, e, intervals):
    """Majority-overlap label for window [s,e); returns (label, overlap_frac)."""
    best, bestov = "Unlabeled", 0.0
    for lab, a, b in intervals:
        ov = max(0.0, min(e, b) - max(s, a))
        if ov > bestov:
            bestov, best = ov, lab
    return best, bestov / max(1.0, (e - s))

def auc(pos, neg):
    """P(pos > neg): rank-based Mann-Whitney AUC. NaNs dropped."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    r_pos = ranks[:len(pos)].sum()
    a = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return a, len(pos), len(neg)

# ---- fit baseline on 00 ----
base_csv = f"{OUT}/00_baseline/00_baseline_windowed_features.csv"
cal = BaselineCalibrator()
stats = cal.fit(base_csv)
print(f"[baseline fit] {stats.baseline_window_count} windows, {len(stats.feature_means)} features\n")

# ---- apply + label every interview clip present ----
rows = []
per_clip = {}
for base in ["02_interview","03_interview","04_interview","05_interview","06_interview","07_interview"]:
    wcsv = f"{OUT}/{base}/{base}_windowed_features.csv"
    if not os.path.exists(wcsv):
        print(f"[skip {base}: no windowed CSV yet]"); continue
    ccsv = f"{SCRATCH}/outputs/{base}_calibrated.csv"
    cal.apply(wcsv, stats, ccsv)
    df = pd.read_csv(ccsv)
    intervals = parse_eaf(eaf_for(base))
    labs, ovs = zip(*[label_window(r.start_time_ms, r.end_time_ms, intervals) for r in df.itertuples()])
    df["target_ground_truth"] = labs
    df["gt_overlap_frac"] = ovs
    df["clip"] = base
    per_clip[base] = df
    rows.append(df[["clip","start_time_ms","end_time_ms","deviation_magnitude","target_ground_truth","gt_overlap_frac"]])

if not rows:
    print("No interview clips scored yet."); sys.exit(0)

allw = pd.concat(rows, ignore_index=True)
# pooled recording-wide percentile of deviation_magnitude (matches assembler semantics)
allw["deviation_percentile"] = allw["deviation_magnitude"].rank(pct=True, na_option="keep")

PURE = allw["gt_overlap_frac"] >= 0.6   # windows dominated by a single label
def grp(df, lab): return df.loc[df.target_ground_truth == lab, "deviation_magnitude"].values

print("=== window counts per label (pooled, all clips) ===")
print(allw["target_ground_truth"].value_counts().to_string(), "\n")

print("=== median deviation_magnitude by label (pure windows, overlap>=0.6) ===")
pw = allw[PURE]
for lab in ["Truth","Lie","Neutral"]:
    v = grp(pw, lab); v = v[~np.isnan(v)]
    if len(v): print(f"  {lab:8s} n={len(v):3d}  median={np.median(v):7.2f}  mean={np.mean(v):7.2f}")
print()

print("=== POOLED AUC (deviation_magnitude discriminates ...) ===")
for name, pos, neg in [
    ("Lie vs Truth",           grp(pw,"Lie"), grp(pw,"Truth")),
    ("Lie vs Neutral",         grp(pw,"Lie"), grp(pw,"Neutral")),
    ("Lie vs Truth+Neutral",   grp(pw,"Lie"), np.concatenate([grp(pw,"Truth"), grp(pw,"Neutral")]) if len(pw) else []),
]:
    a, n1, n2 = auc(pos, neg)
    print(f"  {name:24s} AUC={a:.3f}  (n_lie={n1}, n_other={n2})")
print()

print("=== WITHIN-CLIP AUC Lie vs Truth (controls clip-level confounds) ===")
for base, df in per_clip.items():
    d = df[df.gt_overlap_frac >= 0.6]
    a, n1, n2 = auc(grp(d,"Lie"), grp(d,"Truth"))
    if not np.isnan(a):
        print(f"  {base:14s} AUC={a:.3f}  (n_lie={n1}, n_truth={n2})")

allw.to_csv(f"{SCRATCH}/gt_scored_pooled.csv", index=False)
print(f"\n[wrote {SCRATCH}/gt_scored_pooled.csv]")
