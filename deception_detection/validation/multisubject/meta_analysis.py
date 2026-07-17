"""§6.2 — weak-but-broad channel meta-analysis (the general-signal hunt on small N).

The strict replication scorecard asks "does a channel clear |z| AUC 0.60 in ≥2/3 subjects, same
direction?" — a coarse, high-bar test that a *weak* universal signal would fail. This asks the
complementary question with a random-effects meta-analysis: **pooling all subjects, is a channel's
DIRECTIONAL Lie-vs-Truth separation reliably above chance, and is it consistent across people?**

Two design points that make this honest:
1. **Directional, not magnitude.** We score within-clip-centered *signed* z (higher = Lie), so a
   channel that inverts across subjects (blink: SubjectB lie→more, SubjectD lie→less) correctly
   pools toward 0.5 with HIGH heterogeneity — it is NOT a universal detector even though its |z|
   looked strong. A genuine universal channel must point the same way in most people.
2. **Sampling variance per subject** (Hanley–McNeil SE from n_lie/n_truth) feeds a DerSimonian–Laird
   random-effects pool → pooled AUC, 95% CI, heterogeneity (Q, I², τ²), one-sided p(pooled>0.5).

A channel is a **universal-signal CANDIDATE** iff its pooled-AUC 95% CI excludes 0.5 AND I² is not
high (directionally consistent). At N=6 treat everything as hypothesis-generating (wide CIs); the
value is a ranked shortlist to re-test as the corpus grows toward N≈12.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # deception_detection/ on path

from analytics.baseline_calibrator import NON_FEATURE_COLUMNS
from multisubject.replication_scorecard import load_labeled_windows

_META = set(NON_FEATURE_COLUMNS) | {"gtruth", "ov", "file_index", "clip_window_id"}
MIN_PER_CLASS = 15   # a subject contributes a channel estimate only with ≥ this many Lie & Truth


def directional_auc(pos, neg):
    """AUC of pos>neg (signed): 0.5 = no separation, >0.5 = Lie has higher signed value."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan, 0, 0
    r = stats.rankdata(np.concatenate([pos, neg]))
    a = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return a, len(pos), len(neg)


def hanley_mcneil_se(a, n_pos, n_neg):
    """Standard error of an AUC estimate (Hanley & McNeil 1982)."""
    if not np.isfinite(a) or n_pos < 2 or n_neg < 2:
        return np.nan
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 1e-9)))


def per_subject_channel_aucs(manifest):
    """{channel: [(subject, dir_auc, se, n_lie, n_truth), ...]} using within-clip-centered signed z."""
    out = {}
    for s in manifest["subjects"]:
        pure = load_labeled_windows(s["recording_dir"], s["elan_dir"])
        if pure.empty:
            continue
        pure = pure[pure["gtruth"].isin(["Lie", "Truth"])].copy()
        feats = [c for c in pure.columns if c not in _META and pure[c].dtype.kind in "fi"]
        cen = pure[feats] - pure.groupby("file_index")[feats].transform("median")  # within-clip signed
        is_l = (pure["gtruth"].to_numpy() == "Lie"); is_t = ~is_l
        for c in feats:
            v = cen[c].to_numpy(float)
            a, npos, nneg = directional_auc(v[is_l], v[is_t])
            if not np.isfinite(a) or npos < MIN_PER_CLASS or nneg < MIN_PER_CLASS:
                continue
            se = hanley_mcneil_se(a, npos, nneg)
            out.setdefault(c, []).append((s["name"], a, se, npos, nneg))
    return out


def dersimonian_laird(aucs, ses):
    """Random-effects pool of AUC estimates. Returns pooled, se_pool, tau2, Q, I2, k."""
    a = np.asarray(aucs, float); s = np.asarray(ses, float)
    m = np.isfinite(a) & np.isfinite(s) & (s > 0)
    a, s = a[m], s[m]; k = len(a)
    if k == 0:
        return (np.nan,) * 5 + (0,)
    w = 1.0 / s**2
    fixed = np.sum(w * a) / np.sum(w)
    Q = float(np.sum(w * (a - fixed) ** 2))
    if k > 1:
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    else:
        tau2 = 0.0
    wr = 1.0 / (s**2 + tau2)
    pooled = float(np.sum(wr * a) / np.sum(wr))
    se_pool = float(np.sqrt(1.0 / np.sum(wr)))
    I2 = float(max(0.0, (Q - (k - 1)) / Q) * 100) if (k > 1 and Q > 0) else 0.0
    return pooled, se_pool, tau2, Q, I2, k


def run(manifest, min_subjects=4):
    chans = per_subject_channel_aucs(manifest)
    rows = []
    for c, recs in chans.items():
        subs = [r[0] for r in recs]; aucs = [r[1] for r in recs]; ses = [r[2] for r in recs]
        pooled, se_pool, tau2, Q, I2, k = dersimonian_laird(aucs, ses)
        if k < min_subjects:
            continue
        lo, hi = pooled - 1.96 * se_pool, pooled + 1.96 * se_pool
        z = (pooled - 0.5) / se_pool if se_pool > 0 else np.nan
        p_one = float(stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan  # two-tailed/2 below
        # consistency: fraction of subjects on the pooled side of 0.5
        side = np.sign(pooled - 0.5)
        consist = float(np.mean([np.sign(a - 0.5) == side for a in aucs]))
        rows.append({"channel": c, "k": k, "pooled_auc": pooled, "ci_lo": lo, "ci_hi": hi,
                     "abs_effect": abs(pooled - 0.5), "z": z, "p": p_one, "I2": I2,
                     "consistency": consist, "subjects": ",".join(subs),
                     "per_subject_auc": ";".join(f"{s}:{a:.2f}" for s, a in zip(subs, aucs))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # candidate = CI excludes 0.5 AND directionally consistent (I² not high, ≥ ~2/3 agree)
    df["candidate"] = ((df.ci_lo > 0.5) | (df.ci_hi < 0.5)) & (df.I2 < 60) & (df.consistency >= 0.66)
    return df.sort_values(["candidate", "abs_effect"], ascending=[False, False]).reset_index(drop=True)


def main(manifest_path, out_dir=None, min_subjects=4):
    manifest = json.loads(Path(manifest_path).read_text())
    print(f"[meta] {len(manifest['subjects'])} subjects; directional within-clip signed-z AUC, "
          f"DerSimonian–Laird random-effects pool (min {min_subjects} subjects/channel)\n")
    df = run(manifest, min_subjects=min_subjects)
    if df.empty:
        print("[meta] no channel had enough subjects — nothing to pool."); return

    cand = df[df.candidate]
    print(f"═══ UNIVERSAL-SIGNAL CANDIDATES (pooled-AUC 95% CI excludes 0.5, I²<60, ≥2/3 same dir) ═══")
    if cand.empty:
        print("  NONE. No channel shows a directionally-consistent above-chance pooled effect.")
        print("  → consistent with the strict scorecard + LOSO: no weak universal signal at N=6 either.")
    else:
        for _, r in cand.iterrows():
            print(f"  {r.channel:<40} pooled AUC {r.pooled_auc:.3f} "
                  f"[{r.ci_lo:.3f},{r.ci_hi:.3f}]  I²={r.I2:.0f}%  k={r.k}  "
                  f"consist={r.consistency:.0%}  p={r.p:.3f}")

    print(f"\n═══ top 12 channels by |pooled effect| (candidate or not) ═══")
    print(f"{'channel':<40}{'pooledAUC':>10}{'95% CI':>18}{'I2':>6}{'k':>3}{'consist':>8}  per-subject")
    for _, r in df.head(12).iterrows():
        flag = "★" if r.candidate else " "
        print(f"{flag}{r.channel:<39}{r.pooled_auc:>10.3f}"
              f"{'['+format(r.ci_lo,'.2f')+','+format(r.ci_hi,'.2f')+']':>18}"
              f"{r.I2:>5.0f}%{r.k:>3}{r.consistency:>7.0%}  {r.per_subject_auc}")

    if out_dir:
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        df.to_csv(p / "meta_analysis.csv", index=False)
        print(f"\n[meta] wrote {p/'meta_analysis.csv'} ({len(df)} channels, {int(df.candidate.sum())} candidates)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-subjects", type=int, default=4)
    a = ap.parse_args()
    main(a.manifest, a.out, a.min_subjects)
