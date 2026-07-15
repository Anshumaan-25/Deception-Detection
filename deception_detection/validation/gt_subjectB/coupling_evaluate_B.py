"""Coupling-model 4-bar evaluation on SubjectB (N>1 independent shot; runbook §6.2).
SubjectB's production cascade was canonicalized A/V-synced, so REC_SUBJECTB_00N raw CSVs
are already synced (no separate SYNCED re-cascade needed). ELAN = normalized dir
(unique C-tokens + LIe→Lie). Fit on baseline clip 0 only; labels scoring-only.
Mirror of gt_subjectA/coupling_evaluate.py."""
import sys, os, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
import torch
sys.path.insert(0, os.path.abspath("."))
from stgae import dataset as ds, graph_spec as gs
from stgae.coupling_fit import fit_subject_coupling
from stgae.coupling_attribute import load_model, fit_normalizer, attribute_clip

OUT = "pipeline_system_outputs"
TAG = "REC_SUBJECTB"
ANN = "pipeline_system_outputs/SUBJECTB_SRC/elan_normalized"
FITDIR = f"{OUT}/REC_SUBJECTB/coupling_fit"
FOCUS = 1                      # richest bilabel clip (fidx 1 = B06C002, 146 Lie / 206 Truth)
N_INTERVIEWS = range(1, 7)
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

print("=== COUPLING FIT (SubjectB baseline only) ===")
bundle = fit_subject_coupling(raw(0), FITDIR, device=dev, max_epochs=400, patience=40)
print(f"BAR 0 (fit health): coupling_ratio={bundle['coupling_ratio']:.3f} "
      f"→ {'PASS' if bundle['coupling_ratio'] < 0.90 else 'FAIL (degenerate)'}")
model = load_model(f"{FITDIR}/coupling_model.pt", device=dev)
stats = ds.FrameStats.from_json(f"{FITDIR}/frame_stats.json")
norm = fit_normalizer(raw(0), model, stats, dev)

base_df = attribute_clip(raw(0), model, stats, norm, dev, file_index=0)
frames = []
for fidx in N_INTERVIEWS:
    eaf = glob.glob(f"{ANN}/C{fidx+1:03d}*.eaf")
    if not (os.path.exists(raw(fidx)) and eaf): continue
    df = attribute_clip(raw(fidx), model, stats, norm, dev, file_index=fidx)
    iv = parse_eaf(eaf[0])
    labs, ovs = zip(*[label_win(r.start_time_ms, r.end_time_ms, iv) for r in df.itertuples()])
    df["gtruth"] = labs; df["ov"] = ovs
    frames.append(df)
allw = pd.concat(frames, ignore_index=True)
pure = allw[allw["ov"] >= 0.6].copy()
for c in [f"coupling_z_{n}" for n in gs.NODE_NAMES]:
    pure[f"pct_{c}"] = pure.groupby("file_index")[c].rank(pct=True)

cf = pure[pure.file_index == FOCUS]
a_focus = {n: auc(cf[cf.gtruth=="Lie"][f"coupling_z_{n}"].values, cf[cf.gtruth=="Truth"][f"coupling_z_{n}"].values) for n in gs.NODE_NAMES}
f_focus = {f: auc(cf[cf.gtruth=="Lie"][f"feat_z_{f}"].values, cf[cf.gtruth=="Truth"][f"feat_z_{f}"].values) for f in gs.ALL_FEATURES if f"feat_z_{f}" in cf}
print(f"\npure windows: pooled Lie={(pure.gtruth=='Lie').sum()} Truth={(pure.gtruth=='Truth').sum()}; "
      f"within-{FOCUS} Lie={(cf.gtruth=='Lie').sum()} Truth={(cf.gtruth=='Truth').sum()}")
order = sorted(gs.NODE_NAMES, key=lambda n: -(a_focus[n] if not np.isnan(a_focus[n]) else 0))
print(f"\n=== per-node coupling-z AUC within-clip-{FOCUS} ===")
for n in order: print(f"  {n:12s} {a_focus[n]:.3f}")
ftop = sorted(((v,k) for k,v in f_focus.items() if not np.isnan(v)), reverse=True)[:10]
print("=== top-10 per-feature coupling-z AUC ===")
for v,k in ftop: print(f"  {k:32s} {v:.3f}")

node_best = max(v for v in a_focus.values() if not np.isnan(v))
feat_best = ftop[0][0] if ftop else np.nan
best = np.nanmax([node_best, feat_best])
print("\n================ SUCCESS BARS (SubjectB) ================")
print(f"BAR 1 (≥0.69): best coupling channel within-{FOCUS} = {best:.3f} "
      f"→ {'PASS' if best>=0.69 else 'MISS' if best>=0.66 else 'FAIL'}")
top3 = order[:3]
print(f"BAR 2 (interpretability): top nodes {top3} "
      f"→ {'PASS' if any(t in ('au_mouth','hand_left','hand_right') for t in top3) else 'FAIL'}")
gz = a_focus["gaze"]
print(f"BAR 3 (gaze decoupling ≥0.55): gaze within-{FOCUS} = {gz:.3f} → {'PASS' if gz>=0.55 else 'FAIL'}")
b_med = base_df['coupling_z_global'].median()
t_med = pure[pure.gtruth=='Truth']['coupling_z_global'].median()
l_med = pure[pure.gtruth=='Lie']['coupling_z_global'].median()
thr = base_df['coupling_z_global'].quantile(0.95)
bf = (base_df['coupling_z_global']>thr).mean(); tf=(pure[pure.gtruth=='Truth']['coupling_z_global']>thr).mean(); lf=(pure[pure.gtruth=='Lie']['coupling_z_global']>thr).mean()
stable = (abs(t_med-b_med) <= max(1.0,0.5*abs(l_med-b_med))) and (tf <= bf+0.15)
print(f"BAR 4 (holdout truth stability): median base={b_med:.2f} truth={t_med:.2f} lie={l_med:.2f}; "
      f"flag base={bf:.0%} truth={tf:.0%} lie={lf:.0%} → {'PASS' if stable else 'FAIL (truth spikes)'}")
allw.to_csv(f"{OUT}/REC_SUBJECTB/coupling_attribution.csv", index=False)
print(f"\n[wrote {OUT}/REC_SUBJECTB/coupling_attribution.csv] coupling_ratio={bundle['coupling_ratio']:.3f}")
