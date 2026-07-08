"""Overlap-robustness check: within-clip-06 per-channel AUC on strictly
non-overlapping windows (every 2nd window; 2s window / 1s hop -> stride 2 is
disjoint). Closes the 'adjacent windows share data' caveat in RESULTS.md.
Run with cwd=deception_detection."""
import glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET

OUT = "pipeline_system_outputs/GT_SUBJECTA_20260708/outputs"
ANN = "/home/user1/Documents/Deception_Detection/my_videos/00SubjectA_session1/annotated Videos anushree"

def parse_eaf(p):
    r = ET.parse(p).getroot()
    s = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE")) for t in r.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text, s[a.get("TIME_SLOT_REF1")], s[a.get("TIME_SLOT_REF2")])
            for a in r.iter("ALIGNABLE_ANNOTATION")]

def lab(s, e, iv):
    b, bo = "Unlabeled", 0.0
    for l, a, bb in iv:
        ov = max(0.0, min(e, bb) - max(s, a))
        if ov > bo: bo, b = ov, l
    return b, bo / max(1.0, e - s)

def auc(pos, neg):
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3: return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

df = pd.read_csv(f"{OUT}/06_interview_calibrated.csv")
iv = parse_eaf(glob.glob(f"{ANN}/B04C007_*.eaf")[0])
L, O = zip(*[lab(r.start_time_ms, r.end_time_ms, iv) for r in df.itertuples()])
df["gtruth"] = L; df["ov"] = O

nd = df.iloc[::2]                      # non-overlapping subset
nd = nd[nd["ov"] >= 0.6]
lie, tru = nd[nd.gtruth == "Lie"], nd[nd.gtruth == "Truth"]
print(f"non-overlapping pure windows: Lie={len(lie)} Truth={len(tru)}")

chans = ["AU12_velocity_tremor_band_power", "left_hand_face_distance_min", "AU12_velocity_max",
         "AU12_var", "right_wrist_velocity_max", "AU1_velocity_max", "head_pitch_tremor_band_power",
         "AU4_var", "gaze_x_mean", "gaze_z_mean", "gaze_entropy", "motion_energy_mean"]
full = df[df["ov"] >= 0.6]; fl, ft = full[full.gtruth == "Lie"], full[full.gtruth == "Truth"]
print(f"{'channel':40s} {'AUC(all)':>9s} {'AUC(non-ovl)':>12s}")
for c in chans:
    a1 = auc(np.abs(fl[c].values.astype(float)), np.abs(ft[c].values.astype(float)))
    a2 = auc(np.abs(lie[c].values.astype(float)), np.abs(tru[c].values.astype(float)))
    print(f"{c:40s} {a1:9.3f} {a2:12.3f}")
