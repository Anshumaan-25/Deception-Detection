"""§6.1 — coarser decision units: does aggregating windows into answer-level SEGMENTS lift accuracy?

Window-level AUCs are modest (per-subject ~0.6–0.7; cross-subject LOSO ~0.47). If a real decision is
made per *answer* rather than per *second*, averaging a channel/model over the windows of one
contiguous labeled answer should raise separation when window errors are even weakly independent.
Decision unit = a **contiguous run of one label within a clip** (an answer). We report, window-level
vs segment-level:

  A. CROSS-SUBJECT (the general-detector question): LOSO logreg out-of-fold probabilities, averaged
     per segment → per-held-out-subject segment AUC. Does aggregation rescue the universal model?
  B. CROSS-SUBJECT, weak-universal PANEL: same, but the model uses ONLY the meta-analysis candidate
     channels (the directionally-consistent ones) — the best honest shot at a general detector.
  C. PER-SUBJECT (accuracy from what we already ship): each subject's strongest within-clip channel,
     window-level |z| AUC vs segment-averaged — the cheap within-doctrine accuracy win.

No new data, no doctrine change (labels only score; LOSO holds the subject out). At N=6 read as
directional, not definitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.baseline_calibrator import NON_FEATURE_COLUMNS
from multisubject.loso_head import _model, auc
from multisubject.replication_scorecard import load_labeled_windows

_META = set(NON_FEATURE_COLUMNS) | {"gtruth", "ov", "file_index", "clip_window_id"}


def load_all(manifest):
    """Per-window frame across subjects: subject, file_index, start_time_ms, seg_id, y, features
    (within-clip centered). seg_id = contiguous same-label run within a clip (an 'answer')."""
    parts = []
    for s in manifest["subjects"]:
        pure = load_labeled_windows(s["recording_dir"], s["elan_dir"])
        if pure.empty:
            continue
        pure = pure[pure["gtruth"].isin(["Lie", "Truth"])].copy()
        if pure.empty:
            continue
        feats = [c for c in pure.columns if c not in _META and pure[c].dtype.kind in "fi"]
        cen = pure[feats] - pure.groupby("file_index")[feats].transform("median")
        cen = cen.reset_index(drop=True)
        meta = pure[["file_index", "start_time_ms", "gtruth"]].reset_index(drop=True)
        meta = meta.sort_values(["file_index", "start_time_ms"]).reset_index(drop=False)  # keep orig idx
        # contiguous-run segmentation within each clip
        seg_ids = np.empty(len(meta), dtype=object)
        seg = 0
        for fi, g in meta.groupby("file_index", sort=False):
            prev = None
            for pos, r in zip(g.index, g.itertuples()):
                if prev is None or r.gtruth != prev:
                    seg += 1
                seg_ids[pos] = f"{s['name']}_c{fi}_s{seg}"
                prev = r.gtruth
        meta["seg_id"] = seg_ids
        meta = meta.sort_values("index").set_index("index")
        cen["__subject__"] = s["name"]
        cen["__y__"] = (pure["gtruth"].to_numpy() == "Lie").astype(int)
        cen["__seg__"] = meta["seg_id"].to_numpy()
        parts.append(cen)
    allw = pd.concat(parts, ignore_index=True)
    feats = [c for c in allw.columns if not c.startswith("__")]
    return allw, feats


def loso_oof(allw, feats):
    """Leave-one-subject-out out-of-fold probabilities for every window."""
    subj = allw["__subject__"].to_numpy(); y = allw["__y__"].to_numpy()
    X = allw[feats]; p = np.full(len(allw), np.nan)
    for held in sorted(set(subj)):
        tr, te = subj != held, subj == held
        if len(set(y[tr])) < 2:
            continue
        p[te] = _model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def segment_auc(sub_df, prob_col):
    """AUC of Lie-vs-Truth segments after averaging prob within each segment (segment label = its
    single contiguous label). Returns (win_auc, seg_auc, n_seg_lie, n_seg_truth)."""
    d = sub_df.dropna(subset=[prob_col])
    win = auc(d[d.__y__ == 1][prob_col].to_numpy(), d[d.__y__ == 0][prob_col].to_numpy())
    g = d.groupby("__seg__").agg(prob=(prob_col, "mean"), y=("__y__", "first")).reset_index()
    seg = auc(g[g.y == 1]["prob"].to_numpy(), g[g.y == 0]["prob"].to_numpy())
    return win, seg, int((g.y == 1).sum()), int((g.y == 0).sum())


def main(manifest_path, out_dir=None):
    manifest = json.loads(Path(manifest_path).read_text())
    allw, feats = load_all(manifest)
    subjects = sorted(set(allw["__subject__"]))
    print(f"[agg] {len(subjects)} subjects, {len(allw)} windows, "
          f"{allw['__seg__'].nunique()} answer-segments\n")

    # ── A. cross-subject LOSO: window vs segment ──
    allw["p_full"] = loso_oof(allw, feats)
    print("═══ A. CROSS-SUBJECT (LOSO, all 134 channels): window → answer-segment AUC ═══")
    print(f"{'held-out':<11}{'window':>9}{'segment':>9}{'segLie':>8}{'segTruth':>9}")
    rowsA = []
    for s in subjects:
        w, sg, nl, nt = segment_auc(allw[allw.__subject__ == s], "p_full")
        rowsA.append((s, w, sg, nl, nt))
        print(f"{s:<11}{w:>9.3f}{sg:>9.3f}{nl:>8}{nt:>9}")
    mw = np.nanmean([r[1] for r in rowsA]); ms = np.nanmean([r[2] for r in rowsA])
    print(f"{'MEAN':<11}{mw:>9.3f}{ms:>9.3f}   (window→segment)")

    # ── B. weak-universal PANEL (meta candidates only) ──
    meta_csv = Path(out_dir or ".") / "meta_analysis.csv" if out_dir else None
    panel = []
    if meta_csv and meta_csv.exists():
        md = pd.read_csv(meta_csv)
        panel = [c for c in md[md["candidate"]]["channel"].tolist() if c in feats]
    print(f"\n═══ B. CROSS-SUBJECT weak-universal PANEL ({len(panel)} meta-candidate channels) ═══")
    if len(panel) >= 2:
        allw["p_panel"] = loso_oof(allw, panel)
        print(f"    panel: {', '.join(panel)}")
        print(f"{'held-out':<11}{'window':>9}{'segment':>9}")
        rowsB = []
        for s in subjects:
            w, sg, _, _ = segment_auc(allw[allw.__subject__ == s], "p_panel")
            rowsB.append((s, w, sg)); print(f"{s:<11}{w:>9.3f}{sg:>9.3f}")
        print(f"{'MEAN':<11}{np.nanmean([r[1] for r in rowsB]):>9.3f}"
              f"{np.nanmean([r[2] for r in rowsB]):>9.3f}")
    else:
        rowsB = []
        print("    (need ≥2 meta-candidate channels + meta_analysis.csv in --out; run meta_analysis.py first)")

    # ── C. per-subject strongest channel: window vs segment ──
    print(f"\n═══ C. PER-SUBJECT strongest within-clip channel: window → segment |z|-percentile AUC ═══")
    print(f"{'subject':<11}{'channel':<34}{'window':>9}{'segment':>9}")
    rowsC = []
    for s in subjects:
        sd = allw[allw.__subject__ == s].copy()
        # within-clip |z| percentile per channel, pick the strongest at window level
        best_c, best_w = None, -1
        pct_cache = {}
        for c in feats:
            pct = sd[c].abs().rank(pct=True)  # already within-clip centered; |z| percentile
            a = auc(pct[sd.__y__ == 1].to_numpy(), pct[sd.__y__ == 0].to_numpy())
            if np.isfinite(a) and a > best_w:
                best_w, best_c, pct_cache["p"] = a, c, pct
        sd["p_chan"] = pct_cache["p"].to_numpy()
        w, sg, _, _ = segment_auc(sd, "p_chan")
        rowsC.append((s, best_c, w, sg))
        print(f"{s:<11}{best_c:<34}{w:>9.3f}{sg:>9.3f}")
    print(f"{'MEAN':<11}{'':<34}{np.nanmean([r[2] for r in rowsC]):>9.3f}"
          f"{np.nanmean([r[3] for r in rowsC]):>9.3f}")

    if out_dir:
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rowsA, columns=["subject", "win_auc", "seg_auc", "seg_lie", "seg_truth"]).to_csv(
            p / "aggregate_loso.csv", index=False)
        pd.DataFrame(rowsC, columns=["subject", "channel", "win_auc", "seg_auc"]).to_csv(
            p / "aggregate_persubject.csv", index=False)
        print(f"\n[agg] wrote {p/'aggregate_loso.csv'} + {p/'aggregate_persubject.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.manifest, a.out)
