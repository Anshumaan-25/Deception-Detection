"""Coupling-model 4-bar evaluation, generic across the N=6 corpus (runbook §6.2).

Manifest-driven mirror of gt_subjectA/coupling_evaluate.py + gt_subjectB/coupling_evaluate_B.py.
For each subject: fit the predictive cross-modal coupling model on the baseline clip ONLY (labels
never touch the fit), attribute every interview clip, auto-select the richest bilabel FOCUS clip
(the one subject-specific knob in the per-subject scripts), and run the SAME pre-registered bars:

  BAR 0  fit health           coupling_ratio < 0.90 (else degenerate)
  BAR 1  signal               best within-FOCUS coupling channel AUC ≥ 0.69 (MISS 0.66-0.69)
  BAR 2  interpretability     a top-3 node ∈ {au_mouth, hand_left, hand_right}
  BAR 3  gaze decoupling      within-FOCUS gaze-node AUC ≥ 0.55
  BAR 4  holdout truth stab.  GO/NO-GO — held-out TRUTH must not spike coupling_z vs baseline
                              (truth median shift ≤ max(1, ½·lie shift) AND truth flag-rate ≤ base+15pp)

Thresholds are pre-registered and IDENTICAL to the per-subject scripts — do not tune. Bar 4 is the
go/no-go gate that falsified the model on SubjectA (×2) and SubjectB. This extends the test to the
new subjects. Usage:
    python validation/multisubject/coupling_evaluate_generic.py <manifest.json> [SubjectC SubjectD ...]
(no name args ⇒ every subject in the manifest whose raw CSVs exist).
"""
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from analytics.baseline_calibrator import parse_baseline_file_index  # noqa: E402
from stgae import dataset as ds  # noqa: E402
from stgae import graph_spec as gs  # noqa: E402
from stgae.coupling_attribute import attribute_clip, fit_normalizer, load_model  # noqa: E402
from stgae.coupling_fit import fit_subject_coupling  # noqa: E402

MIN_FOCUS = 20   # a FOCUS clip needs ≥ this many pure Lie AND Truth windows
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_eaf(p):
    r = ET.parse(p).getroot()
    s = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE")) for t in r.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text, s[a.get("TIME_SLOT_REF1")], s[a.get("TIME_SLOT_REF2")])
            for a in r.iter("ALIGNABLE_ANNOTATION")]


def label_win(s, e, iv):
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


def baseline_index(recdir):
    h = sorted(glob.glob(f"{recdir}/*_baseline_stats.json"))
    if not h:
        return 0
    return parse_baseline_file_index(json.loads(open(h[0]).read()).get("source_csv", ""), 0)


def evaluate_subject(name, recdir, elan_dir):
    tag = os.path.basename(recdir.rstrip("/"))
    out = os.path.dirname(recdir.rstrip("/")) or "."   # clip dirs (<tag>_00N) are SIBLINGS of recdir
    raw = lambda i: f"{out}/{tag}_{i:03d}/{tag}_{i:03d}_raw_features_30fps.csv"
    b_idx = baseline_index(recdir)
    n_clips = len(glob.glob(f"{out}/{tag}_0*/{tag}_0*_raw_features_30fps.csv"))
    if not os.path.exists(raw(b_idx)):
        print(f"[{name}] baseline raw CSV missing — skip"); return None

    print(f"\n{'='*66}\n=== COUPLING FIT ({name}, baseline clip {b_idx} only) ===")
    fitdir = f"{recdir}/coupling_fit"
    bundle = fit_subject_coupling(raw(b_idx), fitdir, device=DEV, max_epochs=400, patience=40)
    ratio = bundle["coupling_ratio"]
    print(f"BAR 0 (fit health): coupling_ratio={ratio:.3f} "
          f"→ {'PASS' if ratio < 0.90 else 'FAIL (degenerate)'}")
    model = load_model(f"{fitdir}/coupling_model.pt", device=DEV)
    stats = ds.FrameStats.from_json(f"{fitdir}/frame_stats.json")
    norm = fit_normalizer(raw(b_idx), model, stats, DEV)

    base_df = attribute_clip(raw(b_idx), model, stats, norm, DEV, file_index=b_idx)
    frames = []
    for fidx in range(n_clips):
        if fidx == b_idx:
            continue
        eaf = glob.glob(f"{elan_dir}/C{fidx+1:03d}*.eaf")
        if not (os.path.exists(raw(fidx)) and eaf):
            continue
        df = attribute_clip(raw(fidx), model, stats, norm, DEV, file_index=fidx)
        iv = parse_eaf(eaf[0])
        labs, ovs = zip(*[label_win(r.start_time_ms, r.end_time_ms, iv) for r in df.itertuples()])
        df["gtruth"] = labs; df["ov"] = ovs
        frames.append(df)
    if not frames:
        print(f"[{name}] no labeled interview clips — skip"); return None
    allw = pd.concat(frames, ignore_index=True)
    pure = allw[allw["ov"] >= 0.6].copy()

    # auto-select FOCUS = interview clip with the most balanced bilabel support
    bal = {}
    for fidx, g in pure.groupby("file_index"):
        nl = int((g.gtruth == "Lie").sum()); nt = int((g.gtruth == "Truth").sum())
        if nl >= MIN_FOCUS and nt >= MIN_FOCUS:
            bal[int(fidx)] = min(nl, nt)
    if not bal:
        print(f"[{name}] no clip has ≥{MIN_FOCUS} pure Lie AND Truth windows — cannot run FOCUS bars")
        return {"name": name, "coupling_ratio": ratio, "focus": None, "best": np.nan,
                "gaze": np.nan, "bar4_pass": None, "base_flag": np.nan, "truth_flag": np.nan,
                "lie_flag": np.nan, "n_pure_lie": int((pure.gtruth == "Lie").sum()),
                "n_pure_truth": int((pure.gtruth == "Truth").sum())}
    FOCUS = max(bal, key=bal.get)
    cf = pure[pure.file_index == FOCUS]

    a_node = {n: auc(cf[cf.gtruth == "Lie"][f"coupling_z_{n}"].values,
                     cf[cf.gtruth == "Truth"][f"coupling_z_{n}"].values) for n in gs.NODE_NAMES}
    f_feat = {f: auc(cf[cf.gtruth == "Lie"][f"feat_z_{f}"].values,
                     cf[cf.gtruth == "Truth"][f"feat_z_{f}"].values)
              for f in gs.ALL_FEATURES if f"feat_z_{f}" in cf}
    order = sorted(gs.NODE_NAMES, key=lambda n: -(a_node[n] if not np.isnan(a_node[n]) else 0))
    ftop = sorted(((v, k) for k, v in f_feat.items() if not np.isnan(v)), reverse=True)[:10]
    print(f"FOCUS clip = {FOCUS} (Lie={int((cf.gtruth=='Lie').sum())} "
          f"Truth={int((cf.gtruth=='Truth').sum())}); pooled pure "
          f"Lie={int((pure.gtruth=='Lie').sum())} Truth={int((pure.gtruth=='Truth').sum())}")
    print("  per-node coupling-z AUC:", "  ".join(f"{n}:{a_node[n]:.2f}" for n in order[:5]))

    node_best = max((v for v in a_node.values() if not np.isnan(v)), default=np.nan)
    feat_best = ftop[0][0] if ftop else np.nan
    best = np.nanmax([node_best, feat_best])
    top3 = order[:3]
    gz = a_node["gaze"]
    print(f"BAR 1 (≥0.69): best coupling channel = {best:.3f} "
          f"→ {'PASS' if best>=0.69 else 'MISS' if best>=0.66 else 'FAIL'}")
    print(f"BAR 2 (interpretability): top nodes {top3} "
          f"→ {'PASS' if any(t in ('au_mouth','hand_left','hand_right') for t in top3) else 'FAIL'}")
    print(f"BAR 3 (gaze decoupling ≥0.55): gaze = {gz:.3f} → {'PASS' if gz>=0.55 else 'FAIL'}")

    b_med = base_df["coupling_z_global"].median()
    t_med = pure[pure.gtruth == "Truth"]["coupling_z_global"].median()
    l_med = pure[pure.gtruth == "Lie"]["coupling_z_global"].median()
    thr = base_df["coupling_z_global"].quantile(0.95)
    bf = (base_df["coupling_z_global"] > thr).mean()
    tf = (pure[pure.gtruth == "Truth"]["coupling_z_global"] > thr).mean()
    lf = (pure[pure.gtruth == "Lie"]["coupling_z_global"] > thr).mean()
    stable = (abs(t_med - b_med) <= max(1.0, 0.5 * abs(l_med - b_med))) and (tf <= bf + 0.15)
    print(f"BAR 4 (holdout truth stability): median base={b_med:.2f} truth={t_med:.2f} lie={l_med:.2f}; "
          f"flag base={bf:.0%} truth={tf:.0%} lie={lf:.0%} → "
          f"{'PASS' if stable else 'FAIL (truth spikes)'}")
    allw.to_csv(f"{recdir}/coupling_attribution.csv", index=False)
    return {"name": name, "coupling_ratio": ratio, "focus": FOCUS, "best": float(best),
            "gaze": float(gz) if not np.isnan(gz) else np.nan, "bar4_pass": bool(stable),
            "base_flag": float(bf), "truth_flag": float(tf), "lie_flag": float(lf),
            "n_pure_lie": int((pure.gtruth == "Lie").sum()),
            "n_pure_truth": int((pure.gtruth == "Truth").sum())}


def main(manifest_path, only=None):
    subjects = json.loads(open(manifest_path).read())["subjects"]
    if only:
        subjects = [s for s in subjects if s["name"] in only]
    rows = []
    for s in subjects:
        r = evaluate_subject(s["name"], s["recording_dir"], s["elan_dir"])
        if r:
            rows.append(r)
    print(f"\n{'='*66}\n=== COUPLING 4-BAR SUMMARY ===")
    print(f"{'subject':<10}{'ratio':>7}{'focus':>6}{'best':>7}{'gaze':>7}"
          f"{'BAR4':>7}{'baseFP':>8}{'truthFP':>8}{'lieFP':>7}")
    for r in rows:
        b4 = "PASS" if r["bar4_pass"] else ("—" if r["bar4_pass"] is None else "FAIL")
        fp = lambda k: (f"{r[k]:.0%}" if r[k] == r[k] else "—")
        best = f"{r['best']:.3f}" if r["best"] == r["best"] else "—"
        gaze = f"{r['gaze']:.3f}" if r["gaze"] == r["gaze"] else "—"
        foc = r["focus"] if r["focus"] is not None else "—"
        print(f"{r['name']:<10}{r['coupling_ratio']:>7.3f}{str(foc):>6}{best:>7}{gaze:>7}"
              f"{b4:>7}{fp('base_flag'):>8}{fp('truth_flag'):>8}{fp('lie_flag'):>7}")
    return rows


if __name__ == "__main__":
    manifest = sys.argv[1]
    names = sys.argv[2:] or None
    main(manifest, names)
