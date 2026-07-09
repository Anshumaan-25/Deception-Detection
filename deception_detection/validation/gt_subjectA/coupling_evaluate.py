"""Coupling-model end-to-end evaluation on SubjectA (synced data) against the
pre-registered bars (COUPLING_MODEL_DESIGN §6). Fit on baseline only; ELAN overlaid
post-hoc (scoring only). DESKTOP script — needs pipeline_system_outputs/REC_SUBJECTA_SYNCED_*.

coupling-z per node = how much less predictable a node is from the subject's OTHER
channels vs baseline. Thesis under test: deception shows up as broken cross-modal
couplings (freeze+leakage = decoupling), which conditional prediction can see and
reconstruction error (falsified v1) cannot.

Pooled AUCs are computed on WITHIN-CLIP percentiles (any residual clip-level domain
shift cancels by construction); within-06 AUCs are rank-based and need no such step."""
import sys, os, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
import torch
sys.path.insert(0, os.path.abspath("."))
from stgae import dataset as ds, graph_spec as gs
from stgae.coupling_fit import fit_subject_coupling
from stgae.coupling_attribute import load_model, fit_normalizer, attribute_clip

OUT = "pipeline_system_outputs"
TAG = "REC_SUBJECTA_SYNCED"
ANN = "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"
FITDIR = f"{OUT}/REC_SUBJECTA/coupling_fit"
dev = "cuda" if torch.cuda.is_available() else "cpu"

def raw(fidx): return f"{OUT}/{TAG}_{fidx:03d}/{TAG}_{fidx:03d}_raw_features_30fps.csv"

def parse_eaf(p):
    r = ET.parse(p).getroot()
    s = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE")) for t in r.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text, s[a.get("TIME_SLOT_REF1")], s[a.get("TIME_SLOT_REF2")])
            for a in r.iter("ALIGNABLE_ANNOTATION")]

def label_win(s, e, iv):
    best, bo = "Unlabeled", 0.0
    for lab, a, b in iv:
        ov = max(0.0, min(e, b) - max(s, a))
        if ov > bo: bo, best = ov, lab
    return best, bo / max(1.0, e - s)

def auc(pos, neg):
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3: return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

# 1. FIT on baseline (synced)
print("=== COUPLING FIT (baseline only) ===")
bundle = fit_subject_coupling(raw(0), FITDIR, device=dev, max_epochs=400, patience=40)
print("per-node predictability (ratio vs predict-zero; <1 = neighbors carry info):")
for k, v in sorted(bundle["node_ratio"].items(), key=lambda kv: kv[1]):
    print(f"    {k:12s} {v:.3f}")
print(f"BAR 0 (fit health): coupling_ratio={bundle['coupling_ratio']:.3f} "
      f"→ {'PASS' if bundle['coupling_ratio'] < 0.90 else 'FAIL (degenerate — stop here)'}")

model = load_model(f"{FITDIR}/coupling_model.pt", device=dev)
stats = ds.FrameStats.from_json(f"{FITDIR}/frame_stats.json")
norm = fit_normalizer(raw(0), model, stats, dev)

# 2. baseline windows (Bar 4 reference) + interview attributions
base_df = attribute_clip(raw(0), model, stats, norm, dev, file_index=0)
frames = []
for fidx in range(1, 8):
    eaf = glob.glob(f"{ANN}/B04C{fidx+1:03d}_*.eaf")
    if not (os.path.exists(raw(fidx)) and eaf): continue
    df = attribute_clip(raw(fidx), model, stats, norm, dev, file_index=fidx)
    iv = parse_eaf(eaf[0])
    labs, ovs = zip(*[label_win(r.start_time_ms, r.end_time_ms, iv) for r in df.itertuples()])
    df["gtruth"] = labs; df["ov"] = ovs
    frames.append(df)
allw = pd.concat(frames, ignore_index=True)
pure = allw[allw["ov"] >= 0.6].copy()

# within-clip percentile of every z column (pooled scoring happens on THESE)
zcols = [f"coupling_z_{n}" for n in gs.NODE_NAMES] + \
        [f"feat_z_{f}" for f in gs.ALL_FEATURES] + ["coupling_z_global", "coupling_z_max"]
for c in zcols:
    pure[f"pct_{c}"] = pure.groupby("file_index")[c].rank(pct=True)

def col_aucs(sub, cols):
    lie, tru = sub[sub.gtruth == "Lie"], sub[sub.gtruth == "Truth"]
    return {c: auc(lie[c].values, tru[c].values) for c in cols}

c06 = pure[pure.file_index == 6]
a06 = col_aucs(c06, [f"coupling_z_{n}" for n in gs.NODE_NAMES])
apool = col_aucs(pure, [f"pct_coupling_z_{n}" for n in gs.NODE_NAMES])
f06 = col_aucs(c06, [f"feat_z_{f}" for f in gs.ALL_FEATURES])
print(f"\npure windows: pooled Lie={(pure.gtruth=='Lie').sum()} Truth={(pure.gtruth=='Truth').sum()}; "
      f"within-06 Lie={(c06.gtruth=='Lie').sum()} Truth={(c06.gtruth=='Truth').sum()}")

print("\n=== per-node coupling-z AUC (Lie>Truth; less predictable → lie) ===")
print(f"{'node':12s} {'within06':>9s} {'pooled(pct)':>12s}")
node_order = sorted(gs.NODE_NAMES,
                    key=lambda n: -(a06[f"coupling_z_{n}"] if not np.isnan(a06[f"coupling_z_{n}"]) else 0))
for n in node_order:
    print(f"  {n:12s} {a06[f'coupling_z_{n}']:9.3f} {apool[f'pct_coupling_z_{n}']:12.3f}")
gl06 = auc(c06[c06.gtruth=='Lie']['coupling_z_global'].values,
           c06[c06.gtruth=='Truth']['coupling_z_global'].values)
mx06 = auc(c06[c06.gtruth=='Lie']['coupling_z_max'].values,
           c06[c06.gtruth=='Truth']['coupling_z_max'].values)
print(f"  {'GLOBAL(sum)':12s} {gl06:9.3f}\n  {'MAX-node':12s} {mx06:9.3f}")

print("\n=== top-10 per-FEATURE coupling-z AUC within-06 ===")
ftop = sorted(((v, k) for k, v in f06.items() if not np.isnan(v)), reverse=True)[:10]
for v, k in ftop:
    print(f"  {k[7:]:32s} {v:.3f}")

# ── PRE-REGISTERED BARS (COUPLING_MODEL_DESIGN §6) ───────────────────────────
node_best = max(v for v in a06.values() if not np.isnan(v))
feat_best = ftop[0][0] if ftop else np.nan
best = np.nanmax([node_best, feat_best])
print("\n================ SUCCESS BARS ================")
print(f"BAR 1 (Sensitivity ≥0.69): best coupling channel within-06 AUC = {best:.3f} "
      f"(node {node_best:.3f} / feature {feat_best:.3f}) "
      f"→ {'PASS' if best >= 0.69 else 'MISS' if best >= 0.66 else 'FAIL'}")
top3 = node_order[:3]
print(f"BAR 2 (Interpretability): top nodes = {top3} "
      f"→ {'PASS' if any(t in ('au_mouth','hand_left','hand_right') for t in top3) else 'FAIL'}")
gz = a06["coupling_z_gaze"]
print(f"BAR 3 (Relational, gaze decoupling ≥0.55): gaze coupling-z within-06 AUC = {gz:.3f} "
      f"→ {'PASS' if gz >= 0.55 else 'FAIL'}")
b_med = base_df['coupling_z_global'].median()
t_med = pure[pure.gtruth=='Truth']['coupling_z_global'].median()
l_med = pure[pure.gtruth=='Lie']['coupling_z_global'].median()
thr = base_df['coupling_z_global'].quantile(0.95)
base_flag = (base_df['coupling_z_global'] > thr).mean()
truth_flag = (pure[pure.gtruth=='Truth']['coupling_z_global'] > thr).mean()
lie_flag = (pure[pure.gtruth=='Lie']['coupling_z_global'] > thr).mean()
stable = (abs(t_med - b_med) <= max(1.0, 0.5*abs(l_med - b_med))) and (truth_flag <= base_flag + 0.15)
print(f"BAR 4 (Holdout Truth Stability): global coupling-z median baseline={b_med:.2f} "
      f"truth={t_med:.2f} lie={l_med:.2f}")
print(f"        flag-rate (>p95 baseline) baseline={base_flag:.0%} truth={truth_flag:.0%} "
      f"lie={lie_flag:.0%} → {'PASS (truth stable)' if stable else 'FAIL (truth spikes → brittle)'}")

allw.to_csv(f"{OUT}/REC_SUBJECTA/coupling_attribution.csv", index=False)
print(f"\n[wrote {OUT}/REC_SUBJECTA/coupling_attribution.csv]  "
      f"coupling_ratio={bundle['coupling_ratio']:.3f}")
