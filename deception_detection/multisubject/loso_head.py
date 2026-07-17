"""Supervised leave-one-subject-out (LOSO) head — the pre-registered *generalizability test*.

This is NOT a deployable classifier and NOT a performance play. It answers exactly one question:
*"is there a UNIVERSAL deception signature that transfers across people, or is the signal
subject-specific?"* — by training on N-1 subjects and testing on the held-out subject.
Design + pre-registered bars: ``Documentation/SUPERVISED_LOSO_DESIGN.md``.

Honest prior (written before scoring): given N=6 replication found 0/134 channels replicate, this
head is expected to land near chance. LOSO ≈ chance is a SUCCESS of the test — it quantifies the
subject-specificity and vindicates per-subject calibration + attribution.

Features reuse the shipped within-clip-normalized per-channel z-features (the same the replication
scorecard scores): each subject's *_recording_calibrated.csv values are already per-subject
baseline-normalized z; we additionally CENTER each channel within its own clip (subtract the clip
median) to control clip-level confounds by construction — identical to the scorecard's direction
control, and leak-free (no ELAN labels touch the features). ELAN labels are ONLY the supervised
target, on TRAIN folds, and only ever evaluated on held-out test folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analytics.baseline_calibrator import NON_FEATURE_COLUMNS
from multisubject.replication_scorecard import load_labeled_windows

# ── pre-registered bars (fixed BEFORE scoring; see SUPERVISED_LOSO_DESIGN.md §6) ──
GENERALIZES_MEAN = 0.62      # mean LOSO test AUC bar
GENERALIZES_PERSUBJ = 0.58   # per-held-out-subject bar
GENERALIZES_NSUBJ = 4        # ≥ this many of 6 held-out subjects must clear GENERALIZES_PERSUBJ
WEAK_MEAN_LO = 0.55          # weak/partial band lower edge
MIN_PER_CLASS = 30           # a subject needs ≥ this many Lie AND Truth pure windows to be a fold
_META = set(NON_FEATURE_COLUMNS) | {"gtruth", "ov", "file_index", "clip_window_id"}


def auc(pos, neg):
    """Mann-Whitney AUC of pos vs neg (same estimator the scorecard uses)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def build_feature_table(manifest):
    """Concatenate all subjects' pure Lie/Truth windows into one within-clip-centered table.

    Returns (X: DataFrame[n_windows × n_channels], y: 0/1 Lie array, subj: str array,
             feats: channel list). Subjects with < MIN_PER_CLASS in either class are dropped
             from folds (reported by the caller)."""
    parts = []
    for s in manifest["subjects"]:
        pure = load_labeled_windows(s["recording_dir"], s["elan_dir"])
        if pure.empty:
            continue
        pure = pure[pure["gtruth"].isin(["Lie", "Truth"])].copy()
        if pure.empty:
            continue
        feats = [c for c in pure.columns if c not in _META and pure[c].dtype.kind in "fi"]
        # within-clip centering: subtract each clip's per-channel median (leak-free, no labels)
        cen = pure[feats] - pure.groupby("file_index")[feats].transform("median")
        cen = cen.reset_index(drop=True)
        cen["__subject__"] = s["name"]
        cen["__y__"] = (pure["gtruth"].to_numpy() == "Lie").astype(int)
        parts.append(cen)
    if not parts:
        return pd.DataFrame(), np.array([]), np.array([]), []
    allw = pd.concat(parts, ignore_index=True)
    feats = [c for c in allw.columns if c not in ("__subject__", "__y__")]
    return (allw[feats], allw["__y__"].to_numpy(), allw["__subject__"].to_numpy(), feats)


def _model():
    """Primary head: L2-logreg over the per-channel z-features. Impute clip-centered NaNs to 0
    (= clip median = neutral), then per-fold StandardScaler (fit on TRAIN only — no leakage)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=0.5, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")),
    ])


def run_loso(X, y, subj, feats, model_fn=_model):
    """Strict leave-one-subject-out. Train on 5 subjects, test on the held-out 6th — NO subject
    in both. Returns per-subject rows + aggregate + the pre-registered bar verdict."""
    subjects = sorted(set(subj))
    rows, coefs = [], []
    for held in subjects:
        tr = subj != held; te = subj == held
        n_l = int((y[te] == 1).sum()); n_t = int((y[te] == 0).sum())
        if n_l < MIN_PER_CLASS or n_t < MIN_PER_CLASS:
            rows.append({"subject": held, "test_auc": np.nan, "n_lie": n_l, "n_truth": n_t,
                         "truth_fp_rate": np.nan, "note": "insufficient pure windows"})
            continue
        if len(set(y[tr])) < 2:
            rows.append({"subject": held, "test_auc": np.nan, "n_lie": n_l, "n_truth": n_t,
                         "truth_fp_rate": np.nan, "note": "train single-class"})
            continue
        mdl = model_fn().fit(X[tr], y[tr])
        p = mdl.predict_proba(X[te])[:, 1]
        a = auc(p[y[te] == 1], p[y[te] == 0])
        # Bar-4 analog: held-out TRUTH windows flagged lie (prob > 0.5) — must not rival lie recall
        truth_fp = float(np.mean(p[y[te] == 0] > 0.5)) if n_t else np.nan
        rows.append({"subject": held, "test_auc": float(a), "n_lie": n_l, "n_truth": n_t,
                     "truth_fp_rate": truth_fp, "note": ""})
        clf = mdl.named_steps["clf"]
        coefs.append(pd.Series(clf.coef_[0], index=feats, name=held))
    scored = [r for r in rows if np.isfinite(r["test_auc"])]
    aucs = np.array([r["test_auc"] for r in scored], float)
    mean_auc = float(np.mean(aucs)) if len(aucs) else np.nan
    sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan
    n_clear = int(np.sum(aucs >= GENERALIZES_PERSUBJ))
    verdict = _verdict(mean_auc, n_clear, len(aucs))
    coef_tbl = pd.concat(coefs, axis=1) if coefs else pd.DataFrame()
    return {"rows": rows, "mean_auc": mean_auc, "sd": sd, "n_folds": len(aucs),
            "n_clear": n_clear, "verdict": verdict, "coef_table": coef_tbl}


def _verdict(mean_auc, n_clear, n_folds):
    if not np.isfinite(mean_auc):
        return "INSUFFICIENT"
    if mean_auc >= GENERALIZES_MEAN and n_clear >= GENERALIZES_NSUBJ:
        return "GENERALIZES"
    if mean_auc >= WEAK_MEAN_LO:
        return "WEAK/PARTIAL"
    return "SUBJECT-SPECIFIC"
