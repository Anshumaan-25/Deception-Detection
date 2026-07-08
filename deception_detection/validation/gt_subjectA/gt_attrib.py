"""Per-feature attribution + L2-domination diagnostics (validation-only).
Requires calibrated per-clip CSVs from _gt_score.py (gt_out/<clip>_calibrated.csv).
- Which features' |z| discriminate Lie vs Truth windows (per-feature rank AUC)?
- Is deviation_magnitude dominated by a few features (top-1/top-5 share of sum z^2)?
- Robust alternative aggregate (median |z|) — does it separate better than L2?
"""
import sys, os, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
sys.path.insert(0, "analytics")
from baseline_calibrator import NON_FEATURE_COLUMNS

SCRATCH = "deception_detection/pipeline_system_outputs/GT_SUBJECTA_20260708"
ANN = "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"

def parse_eaf(p):
    r = ET.parse(p).getroot()
    slots = {ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE")) for ts in r.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text,
             slots[a.get("TIME_SLOT_REF1")], slots[a.get("TIME_SLOT_REF2")])
            for a in r.iter("ALIGNABLE_ANNOTATION")]

def label_window(s, e, intervals):
    best, bestov = "Unlabeled", 0.0
    for lab, a, b in intervals:
        ov = max(0.0, min(e, b) - max(s, a))
        if ov > bestov: bestov, best = ov, lab
    return best, bestov / max(1.0, e - s)

def rank_auc(pos, neg):
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3: return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2.0) / (len(pos)*len(neg))

frames = []
for ccsv in sorted(glob.glob(f"{SCRATCH}/outputs/*_calibrated.csv")):
    base = os.path.basename(ccsv).replace("_calibrated.csv", "")
    idx = int(base.split("_")[0])
    eaf = glob.glob(f"{ANN}/B04C{idx+1:03d}_*.eaf")
    if not eaf: continue
    df = pd.read_csv(ccsv)
    intervals = parse_eaf(eaf[0])
    labs, ovs = zip(*[label_window(r.start_time_ms, r.end_time_ms, intervals) for r in df.itertuples()])
    df["gtruth"] = labs; df["ov"] = ovs; df["clip"] = base
    frames.append(df)

allw = pd.concat(frames, ignore_index=True)
pure = allw[allw["ov"] >= 0.6].copy()
feat_cols = [c for c in pure.columns
             if c not in NON_FEATURE_COLUMNS + ["gtruth", "ov", "clip"]
             and pure[c].dtype.kind in "fi"]

lie = pure[pure["gtruth"] == "Lie"]; truth = pure[pure["gtruth"] == "Truth"]
print(f"pure windows: Lie={len(lie)} Truth={len(truth)} (of {len(allw)} total)  features={len(feat_cols)}\n")

# ---- 1. per-feature |z| AUC (Lie vs Truth) ----
rows = []
for c in feat_cols:
    a = rank_auc(np.abs(lie[c].values.astype(float)), np.abs(truth[c].values.astype(float)))
    if not np.isnan(a):
        rows.append((c, a, np.nanmedian(np.abs(lie[c])), np.nanmedian(np.abs(truth[c]))))
rows.sort(key=lambda r: -r[1])
print("=== TOP 20 features by |z| AUC (Lie > Truth) ===")
print(f"{'feature':44s} {'AUC':>6s} {'med|z|Lie':>10s} {'med|z|Tru':>10s}")
for c, a, ml, mt in rows[:20]:
    print(f"{c:44s} {a:6.3f} {ml:10.2f} {mt:10.2f}")
print("\n=== BOTTOM 5 (Truth > Lie — inverse signals) ===")
for c, a, ml, mt in rows[-5:]:
    print(f"{c:44s} {a:6.3f} {ml:10.2f} {mt:10.2f}")

# ---- 2. L2 domination: share of sum z^2 from top-1 / top-5 features per window ----
Z2 = pure[feat_cols].astype(float) ** 2
tot = Z2.sum(axis=1)
srt = np.sort(Z2.fillna(0.0).values, axis=1)[:, ::-1]
top1 = srt[:, 0] / np.maximum(tot.values, 1e-12)
top5 = srt[:, :5].sum(axis=1) / np.maximum(tot.values, 1e-12)
print(f"\n=== L2 domination (pure windows) ===")
print(f"top-1 feature share of z^2:  median {np.median(top1):.1%}, p90 {np.percentile(top1,90):.1%}")
print(f"top-5 feature share of z^2:  median {np.median(top5):.1%}, p90 {np.percentile(top5,90):.1%}")
mostly = pd.Series(np.argmax(Z2.fillna(0.0).values, axis=1)).map(lambda i: feat_cols[i]).value_counts()
print("most-dominant feature (count of windows where it is top-1):")
print(mostly.head(8).to_string())

# ---- 3. aggregate comparison: L2 vs median|z| vs trimmed L2 ----
def agg_auc(series_fn, name):
    s = np.asarray(series_fn(pure[feat_cols].astype(float)), dtype=float)
    a = rank_auc(s[(pure["gtruth"] == "Lie").values], s[(pure["gtruth"] == "Truth").values])
    print(f"  {name:28s} AUC={a:.3f}")
print("\n=== aggregate Lie-vs-Truth AUC ===")
agg_auc(lambda z: np.sqrt((z**2).sum(axis=1)), "L2 (current deviation_mag)")
agg_auc(lambda z: z.abs().median(axis=1), "median |z| (robust)")
def trimmed_l2(z):
    z2 = z.fillna(0.0).values ** 2
    s = np.sort(z2, axis=1)[:, ::-1]
    return np.sqrt(s[:, 5:].sum(axis=1))   # drop top-5 per window
agg_auc(trimmed_l2, "L2 excluding top-5/window")
agg_auc(lambda z: (z.abs() > 3).sum(axis=1).astype(float), "count of |z|>3 features")

# ---- 4. CONFOUND CONTROL: per-feature AUC within clip 06 only ----
c06 = pure[pure["clip"] == "06_interview"]
l6, t6 = c06[c06["gtruth"] == "Lie"], c06[c06["gtruth"] == "Truth"]
print(f"\n=== WITHIN-CLIP-06 per-feature |z| AUC (n_lie={len(l6)}, n_truth={len(t6)}) ===")
rows6 = []
for c in feat_cols:
    a = rank_auc(np.abs(l6[c].values.astype(float)), np.abs(t6[c].values.astype(float)))
    if not np.isnan(a): rows6.append((c, a))
rows6.sort(key=lambda r: -r[1])
print("top 12 (Lie>Truth):")
for c, a in rows6[:12]: print(f"  {c:44s} {a:6.3f}")
print("bottom 5 (Truth>Lie):")
for c, a in rows6[-5:]: print(f"  {c:44s} {a:6.3f}")
# where does the pooled facial cluster land within-clip?
cluster = ["AU9_mean","AU9_var","AU4_var","AU4_max","disgust_leak","head_pitch_mean","AU4_velocity_mean"]
d6 = dict(rows6)
print("pooled-top cluster, within-06 AUC:")
for c in cluster: print(f"  {c:44s} {d6.get(c, float('nan')):6.3f}")
