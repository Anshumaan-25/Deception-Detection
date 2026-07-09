"""
Coupling attribution — apply a fitted CouplingPredictor to interview clips and emit
per-node AND per-feature coupling-z on the 2 s / 1 s window grid (aligned with the
marginal z path and the ELAN scorer).

coupling-z per node = how much worse the node is predicted from its neighbors vs the
baseline clip's own prediction-error distribution: +2 means "this channel is 2σ less
predictable from the subject's other channels than at baseline" — a decoupling reading.

Two v1 lessons are baked in (COUPLING_MODEL_DESIGN §5):
  • per-feature residuals are first-class outputs (feat_z_* columns), so the
    discriminative needle is never lost to node aggregation;
  • all frame→window aggregation is TARGET-VALIDITY-weighted — a non-speaking voice
    frame contributes nothing instead of a fake zero error, and a window with no valid
    target frames is NaN, not 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import graph_spec as gs
from . import dataset as ds
from .coupling_model import CouplingPredictor, expand_all_masks

WIN_MS, HOP_MS, FPS = 2000, 1000, 30.0


def load_model(model_pt: str, embed=16, blocks=2, device="cpu"):
    m = CouplingPredictor(embed=embed, blocks=blocks)
    m.load_state_dict(torch.load(model_pt, map_location=device))
    m.eval().to(device)
    return m


@torch.no_grad()
def _frame_coupling_error(csv, model, stats, dev, win=90, stride=15, chunk=64):
    """Per-frame target-node prediction error, averaged over all covering model
    windows with validity weights.

    Returns (acc_n, wgt_n, acc_f, wgt_f, ts):
      acc_n/wgt_n [T,N]  Σ node error / Σ has-valid-frame counts
      acc_f/wgt_f [T,F]  Σ per-feature squared error / Σ valid counts
    """
    X, V, W, ts = ds.load_clip_matrix(csv, stats)
    T, F = X.shape[0], len(gs.ALL_FEATURES)
    acc_n = np.zeros((T, gs.N_NODES)); wgt_n = np.zeros((T, gs.N_NODES))
    acc_f = np.zeros((T, F)); wgt_f = np.zeros((T, F))

    xb, vb, starts = [], [], []
    for s in range(0, max(1, T - win + 1), stride):
        e = s + win
        if e > T:
            break
        xb.append(X[s:e]); vb.append(V[s:e]); starts.append(s)
    if not xb:                                   # clip shorter than one window
        xb = [np.pad(X, ((0, win - T), (0, 0), (0, 0)))]
        vb = [np.pad(V, ((0, win - T), (0, 0), (0, 0)))]
        starts = [0]

    for i in range(0, len(xb), chunk):
        Xt = torch.from_numpy(np.stack(xb[i:i+chunk])).to(dev)
        Vt = torch.from_numpy(np.stack(vb[i:i+chunk])).to(dev)
        b = Xt.shape[0]
        # vectorized 11-target pass: one batched forward, not 11 sequential ones
        Xe, Ve, _, mi = expand_all_masks(Xt, Vt, torch.ones(b, Xt.shape[1], device=dev))
        xhat = model(Xe, mi)
        rows = torch.arange(Xe.shape[0], device=dev)
        se_t = (Ve * (xhat - Xe) ** 2)[rows, :, mi]              # [b*N, win, D]
        v_t = Ve[rows, :, mi]                                    # [b*N, win, D]
        ne_t = (se_t.sum(-1) / v_t.sum(-1).clamp_min(1.0))       # [b*N, win]
        has_t = (v_t.sum(-1) > 0).float()                        # [b*N, win]
        # sample order from repeat_interleave: (window-major, node-minor)
        ne_t = ne_t.reshape(b, gs.N_NODES, win).cpu().numpy()
        has_t = has_t.reshape(b, gs.N_NODES, win).cpu().numpy()
        se_t = se_t.reshape(b, gs.N_NODES, win, -1).cpu().numpy()
        v_t = v_t.reshape(b, gs.N_NODES, win, -1).cpu().numpy()
        for k, s in enumerate(starts[i:i+chunk]):
            span = min(win, T - s)
            acc_n[s:s+span] += (ne_t[k, :, :span] * has_t[k, :, :span]).T
            wgt_n[s:s+span] += has_t[k, :, :span].T
            for name, (lo, hi) in gs.NODE_SLICES.items():
                n, hn = gs.NODE_INDEX[name], hi - lo
                acc_f[s:s+span, lo:hi] += se_t[k, n, :span, :hn]
                wgt_f[s:s+span, lo:hi] += v_t[k, n, :span, :hn]
    return acc_n, wgt_n, acc_f, wgt_f, ts


def _windowize(acc, wgt, ts):
    """Validity-weighted 2 s / 1 s aggregation → (starts_ms, [W,C]); a window with no
    valid target frames is NaN (never a fake 0)."""
    T = acc.shape[0]
    win_f, hop_f = int(WIN_MS / 1000 * FPS), int(HOP_MS / 1000 * FPS)
    starts_ms, rows = [], []
    for s in range(0, max(1, T - win_f + 1), hop_f):
        e = s + win_f
        if e > T:
            break
        num = acc[s:e].sum(0); den = wgt[s:e].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
        starts_ms.append(float(ts[s]))
    return np.array(starts_ms), np.array(rows)


def fit_normalizer(baseline_csv, model, stats, dev):
    """Per-node and per-feature coupling-error mean/std over the baseline clip's
    2 s windows — the reference distribution for coupling-z."""
    acc_n, wgt_n, acc_f, wgt_f, ts = _frame_coupling_error(baseline_csv, model, stats, dev)
    _, rows_n = _windowize(acc_n, wgt_n, ts)
    _, rows_f = _windowize(acc_f, wgt_f, ts)
    return (np.nanmean(rows_n, 0), np.nanstd(rows_n, 0) + 1e-6,
            np.nanmean(rows_f, 0), np.nanstd(rows_f, 0) + 1e-6)


def attribute_clip(csv, model, stats, norm, dev, file_index=None):
    """DataFrame: one row per 2 s window with per-node coupling_z_*, per-feature
    feat_z_*, and global/max summaries."""
    n_mean, n_std, f_mean, f_std = norm
    acc_n, wgt_n, acc_f, wgt_f, ts = _frame_coupling_error(csv, model, stats, dev)
    starts, rows_n = _windowize(acc_n, wgt_n, ts)
    _, rows_f = _windowize(acc_f, wgt_f, ts)
    zn = (rows_n - n_mean) / n_std
    zf = (rows_f - f_mean) / f_std
    df = pd.DataFrame({"start_time_ms": starts, "end_time_ms": starts + WIN_MS})
    for n, name in enumerate(gs.NODE_NAMES):
        df[f"coupling_z_{name}"] = zn[:, n]
    df["coupling_z_global"] = np.nansum(zn, 1)
    mx = np.full(len(starts), np.nan)
    anyv = ~np.isnan(zn).all(1)
    if anyv.any():
        mx[anyv] = np.nanmax(zn[anyv], 1)
    df["coupling_z_max"] = mx
    for j, feat in enumerate(gs.ALL_FEATURES):
        df[f"feat_z_{feat}"] = zf[:, j]
    if file_index is not None:
        df["file_index"] = file_index
    return df


if __name__ == "__main__":
    import sys
    from .dataset import FrameStats
    model_pt, frame_stats_json, baseline_csv, clip_csv = sys.argv[1:5]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_pt, device=dev)
    stats = FrameStats.from_json(frame_stats_json)
    norm = fit_normalizer(baseline_csv, model, stats, dev)
    out = attribute_clip(clip_csv, model, stats, norm, dev)
    print(out.filter(regex="start_time_ms|coupling_z_(global|au_mouth|gaze|voice)").head(10).to_string())
