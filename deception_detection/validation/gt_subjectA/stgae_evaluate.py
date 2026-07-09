"""ST-GAE end-to-end evaluation on SubjectA (synced data) against the 4 pre-registered
success bars (ST_GAE_DESIGN §6). Fit on baseline only; ELAN overlaid post-hoc (scoring only).

recon-z per node = how much worse a node reconstructs vs its baseline distribution.
The ST-GAE's thesis: BOTH hyper-activation and freeze show up as HIGH recon-z (unlike the
bipolar marginal |z|), giving a single direction-agnostic anomaly signal per node."""
import sys, os, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
import torch
sys.path.insert(0, os.path.abspath("."))
from stgae import dataset as ds, graph_spec as gs
from stgae.fit import fit_subject
from stgae.attribute import load_model, fit_normalizer, attribute_clip

OUT = "pipeline_system_outputs"
TAG = "REC_SUBJECTA_SYNCED"
ANN = "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"
FITDIR = "pipeline_system_outputs/REC_SUBJECTA/stgae_fit"
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
print("=== FIT (baseline only) ===")
bundle = fit_subject(raw(0), FITDIR, device=dev, max_epochs=400, patience=40)
model = load_model(f"{FITDIR}/stgae_model.pt", device=dev)
stats = ds.FrameStats.from_json(f"{FITDIR}/frame_stats.json")
norm = fit_normalizer(raw(0), model, stats, dev)

# 2. baseline windows (for Bar 4 reference) + interview attributions
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
zcols = [f"recon_z_{n}" for n in gs.NODE_NAMES]

def node_aucs(sub):
    lie, tru = sub[sub.gtruth=="Lie"], sub[sub.gtruth=="Truth"]
    return {n: auc(lie[f"recon_z_{n}"].values, tru[f"recon_z_{n}"].values) for n in gs.NODE_NAMES}

c06 = pure[pure.file_index==6]
a06 = node_aucs(c06)
apool = node_aucs(pure)
print(f"\npure windows: pooled Lie={ (pure.gtruth=='Lie').sum() } Truth={ (pure.gtruth=='Truth').sum() }; "
      f"within-06 Lie={ (c06.gtruth=='Lie').sum() } Truth={ (c06.gtruth=='Truth').sum() }")

print("\n=== per-node recon-z AUC (Lie>Truth; higher recon error → lie) ===")
print(f"{'node':12s} {'within06':>9s} {'pooled':>8s}")
for n in sorted(gs.NODE_NAMES, key=lambda n:-(a06[n] if not np.isnan(a06[n]) else 0)):
    print(f"  {n:12s} {a06[n]:9.3f} {apool[n]:8.3f}")
gl06 = auc(c06[c06.gtruth=='Lie']['recon_z_global'].values, c06[c06.gtruth=='Truth']['recon_z_global'].values)
mx06 = auc(c06[c06.gtruth=='Lie']['recon_z_max'].values, c06[c06.gtruth=='Truth']['recon_z_max'].values)
print(f"  {'GLOBAL(sum)':12s} {gl06:9.3f}")
print(f"  {'MAX-node':12s} {mx06:9.3f}")

# ── BARS ──────────────────────────────────────────────────────────────────
best_node = max((v for v in a06.values() if not np.isnan(v)))
print("\n================ SUCCESS BARS ================")
print(f"BAR 1 (Sensitivity ≥0.69): best within-06 node recon-z AUC = {best_node:.3f}  "
      f"→ {'PASS' if best_node>=0.69 else 'MISS' if best_node>=0.66 else 'FAIL'}")
top = sorted(gs.NODE_NAMES, key=lambda n:-(a06[n] if not np.isnan(a06[n]) else 0))[:3]
print(f"BAR 2 (Interpretability): top nodes = {top}  "
      f"→ {'PASS' if any(t in ('au_mouth','hand_left','hand_right') for t in top) else 'REVIEW'}")
print(f"BAR 3 (Relational): congruence recon-z AUC within-06 = {a06['congruence']:.3f}; "
      f"gaze (marginal-inverse) recon-z = {a06['gaze']:.3f} "
      f"→ {'gaze freeze now reads as anomaly (>0.5)' if a06['gaze']>0.5 else 'gaze still not flagged'}")
# BAR 4: Holdout Truth Stability
b_med = base_df['recon_z_global'].median()
t_med = pure[pure.gtruth=='Truth']['recon_z_global'].median()
l_med = pure[pure.gtruth=='Lie']['recon_z_global'].median()
thr = base_df['recon_z_global'].quantile(0.95)
base_flag = (base_df['recon_z_global']>thr).mean()
truth_flag = (pure[pure.gtruth=='Truth']['recon_z_global']>thr).mean()
lie_flag = (pure[pure.gtruth=='Lie']['recon_z_global']>thr).mean()
stable = (abs(t_med - b_med) <= max(1.0, 0.5*abs(l_med-b_med))) and (truth_flag <= base_flag + 0.15)
print(f"BAR 4 (Holdout Truth Stability): global recon-z median baseline={b_med:.2f} truth={t_med:.2f} lie={l_med:.2f}")
print(f"        flag-rate (>p95 baseline) baseline={base_flag:.0%} truth={truth_flag:.0%} lie={lie_flag:.0%}  "
      f"→ {'PASS (truth stable)' if stable else 'FAIL (truth spikes → brittle fit)'}")

allw.to_csv(f"{OUT}/REC_SUBJECTA/stgae_attribution.csv", index=False)
print(f"\n[wrote {OUT}/REC_SUBJECTA/stgae_attribution.csv]  recon_ratio={bundle['recon_ratio']:.3f}")
