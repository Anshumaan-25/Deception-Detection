"""Run the pre-registered supervised LOSO head on the N=6 corpus and print the bar verdict.

Usage:
    python validation/multisubject/loso_evaluate.py \
        validation/multisubject/replication_manifest_N6.json [--out validation/multisubject]

Prints per-held-out-subject test AUC (all folds, not just the mean), aggregate mean ± SD, the
held-out truth-window false-positive rate (Bar-4 analog), the top cross-subject logreg channel
weights, and the pre-registered verdict (GENERALIZES / WEAK-PARTIAL / SUBJECT-SPECIFIC). Writes
``loso_results.csv``. This is a generalizability TEST, not a deployable model — see
``Documentation/SUPERVISED_LOSO_DESIGN.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # deception_detection/ on path

from multisubject.loso_head import (GENERALIZES_MEAN, GENERALIZES_NSUBJ,  # noqa: E402
                                    GENERALIZES_PERSUBJ, build_feature_table, run_loso)


def main(manifest_path, out_dir=None):
    manifest = json.loads(Path(manifest_path).read_text())
    names = [s["name"] for s in manifest["subjects"]]
    print(f"[loso] corpus: {', '.join(names)} (N={len(names)})")
    X, y, subj, feats = build_feature_table(manifest)
    if X.empty:
        print("[loso] no labeled windows — nothing to score."); return
    for s in sorted(set(subj)):
        m = subj == s
        print(f"[loso] {s}: {int((y[m]==1).sum())} Lie / {int((y[m]==0).sum())} Truth pure windows")
    print(f"[loso] {len(feats)} channels, {len(y)} windows total\n")

    res = run_loso(X, y, subj, feats)

    print("═══ LEAVE-ONE-SUBJECT-OUT test AUC (held-out subject never in train) ═══")
    print(f"{'held-out':<12}{'test_AUC':>10}{'truth_FP':>11}{'n_Lie':>8}{'n_Truth':>9}  note")
    for r in res["rows"]:
        a = f"{r['test_auc']:.3f}" if np.isfinite(r["test_auc"]) else "   —"
        fp = f"{r['truth_fp_rate']:.3f}" if np.isfinite(r["truth_fp_rate"]) else "   —"
        print(f"{r['subject']:<12}{a:>10}{fp:>11}{r['n_lie']:>8}{r['n_truth']:>9}  {r['note']}")

    m, sd = res["mean_auc"], res["sd"]
    print(f"\nmean LOSO test AUC = {m:.3f}"
          + (f" ± {sd:.3f} (SD over {res['n_folds']} folds)" if np.isfinite(sd) else "")
          + f"  |  {res['n_clear']}/{res['n_folds']} held-out subjects ≥ {GENERALIZES_PERSUBJ}")
    print(f"pre-registered bars: GENERALIZES = mean ≥ {GENERALIZES_MEAN} AND "
          f"≥ {GENERALIZES_NSUBJ}/6 subjects ≥ {GENERALIZES_PERSUBJ}\n")

    print(f"╣ VERDICT: {res['verdict']} ╠")
    if res["verdict"] == "SUBJECT-SPECIFIC":
        print("  → confirms channels don't transfer across people; VINDICATES per-subject calibration")
        print("    + per-channel |z| attribution. No universal/supervised model pursued at this N.")
    elif res["verdict"] == "WEAK/PARTIAL":
        print("  → some transferable signal, unreliable; NOT deployable. Flag the recurring channels.")
    elif res["verdict"] == "GENERALIZES":
        print("  → DOCTRINE-REVISING: a universal signature exists. Report which channels it uses;")
        print("    the head becomes a research complement (channel weighting), NEVER a per-window verdict.")

    # top cross-subject channel weights (|mean coef| across folds) — interpretability, not deployment
    if not res["coef_table"].empty:
        imp = res["coef_table"].abs().mean(axis=1).sort_values(ascending=False)
        print("\ntop cross-subject logreg channels (|mean coef| over folds — research signal only):")
        for c, v in imp.head(10).items():
            print(f"   {c:<40}{v:.3f}")

    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res["rows"]).to_csv(out / "loso_results.csv", index=False)
        if not res["coef_table"].empty:
            res["coef_table"].to_csv(out / "loso_coefficients.csv")
        print(f"\n[loso] wrote {out/'loso_results.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.manifest, a.out)
