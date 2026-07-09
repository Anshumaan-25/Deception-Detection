"""ELAN re-score on the PRODUCTION Stage-2 outputs (isolated target audio + blink alive).
Per-feature |z| AUC (Lie vs Truth), pooled + within-clip-06, with blink node highlighted
and compared to the 2026-07-08 whole-clip-audio run. Validation-only (labels never touch fit)."""
import sys, os, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
sys.path.insert(0, "analytics")
from baseline_calibrator import NON_FEATURE_COLUMNS

OUT = "pipeline_system_outputs"
ANN = "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"

def parse_eaf(p):
    r = ET.parse(p).getroot()
    s = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE")) for t in r.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text, s[a.get("TIME_SLOT_REF1")], s[a.get("TIME_SLOT_REF2")])
            for a in r.iter("ALIGNABLE_ANNOTATION")]

def label_window(s, e, iv):
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

frames = []
for fidx in range(1, 8):
    ccsv = f"{OUT}/REC_SUBJECTA_{fidx:03d}/REC_SUBJECTA_{fidx:03d}_calibrated_features.csv"
    eaf = glob.glob(f"{ANN}/B04C{fidx+1:03d}_*.eaf")
    if not (os.path.exists(ccsv) and eaf): continue
    df = pd.read_csv(ccsv)
    iv = parse_eaf(eaf[0])
    labs, ovs = zip(*[label_window(r.start_time_ms, r.end_time_ms, iv) for r in df.itertuples()])
    df["gtruth"] = labs; df["ov"] = ovs; df["clip"] = f"REC_SUBJECTA_{fidx:03d}"
    frames.append(df)

allw = pd.concat(frames, ignore_index=True)
pure = allw[allw["ov"] >= 0.6].copy()
feat = [c for c in pure.columns if c not in NON_FEATURE_COLUMNS + ["gtruth","ov","clip"]
        and pure[c].dtype.kind in "fi"]
lie, tru = pure[pure["gtruth"]=="Lie"], pure[pure["gtruth"]=="Truth"]
print(f"PRODUCTION re-score | pure windows pooled: Lie={len(lie)} Truth={len(tru)} | features={len(feat)}\n")

def perfeat(L, T, cols):
    rows=[]
    for c in cols:
        a = auc(np.abs(L[c].values.astype(float)), np.abs(T[c].values.astype(float)))
        if not np.isnan(a): rows.append((c, a))
    return sorted(rows, key=lambda r:-r[1])

BLINK = ["blink_rate","ear_mean","ear_var"]
ac = [c for c in feat if c.startswith("wavlm") or "acoustic" in c or "prosod" in c or "vocal" in c]

print("=== POOLED top-15 |z| AUC (Lie>Truth) ===")
for c,a in perfeat(lie,tru,feat)[:15]: print(f"  {c:40s} {a:.3f}")
print("\n=== BLINK NODE (first-ever test — was 100% NaN before today) ===")
d=dict(perfeat(lie,tru,feat))
for c in BLINK: print(f"  {c:40s} {d.get(c,float('nan')):.3f}")
print("\n=== ACOUSTIC/VOICE (now target-isolated; cf. 2026-07-08 wavlm_latent_4=0.608 whole-clip) ===")
for c,a in perfeat(lie,tru,ac)[:6]: print(f"  {c:40s} {a:.3f}")

# within-clip 06 (confound-controlled; the Truth->Lie clip)
c06 = pure[pure["clip"]=="REC_SUBJECTA_006"]
l6,t6 = c06[c06["gtruth"]=="Lie"], c06[c06["gtruth"]=="Truth"]
print(f"\n=== WITHIN-CLIP-06 |z| AUC (n_lie={len(l6)}, n_truth={len(t6)}) — cf. 2026-07-08 in [ ] ===")
prev = {"AU12_velocity_tremor_band_power":0.681,"left_hand_face_distance_min":0.679,
        "AU12_velocity_max":0.677,"right_wrist_velocity_max":0.660,"head_pitch_tremor_band_power":0.625,
        "gaze_x_mean":0.162,"gaze_entropy":0.313}
r6 = perfeat(l6,t6,feat)
d6 = dict(r6)
print("  top 12:")
for c,a in r6[:12]: print(f"    {c:40s} {a:.3f}" + (f"   [{prev[c]:.3f}]" if c in prev else ""))
print("  tracked channels (compare to 2026-07-08):")
for c in prev: print(f"    {c:40s} {d6.get(c,float('nan')):.3f}   [{prev[c]:.3f}]")
print("  blink within-06:")
for c in BLINK: print(f"    {c:40s} {d6.get(c,float('nan')):.3f}")

allw.to_csv("deception_detection/pipeline_system_outputs/REC_SUBJECTA/recA_scored.csv", index=False)
print("\n[wrote recA_scored.csv]")
