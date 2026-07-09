"""
ST-GAE attribution — apply a fitted subject model to interview clips and emit the
per-node reconstruction-z on the 2 s / 1 s window grid (aligned with the z-score
deviation path and the ELAN scorer).

Reconstruction-z is fit/apply just like BaselineCalibrator: normalize each node's
window reconstruction error by that node's mean/std over the BASELINE clip's windows,
so a value of +2 means "this node reconstructs 2σ worse than it does at baseline."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import graph_spec as gs
from . import dataset as ds
from .model import STGAE, masked_node_error

WIN_MS, HOP_MS, FPS = 2000, 1000, 30.0


def load_model(model_pt: str, embed=16, blocks=3, device="cpu"):
    m = STGAE(embed=embed, blocks=blocks)
    m.load_state_dict(torch.load(model_pt, map_location=device))
    m.eval().to(device)
    return m


def _frame_node_error(csv, model, stats, dev, win=90, stride=15):
    """Per-frame per-node reconstruction error [T,N], averaged over all overlapping
    model windows that cover each frame."""
    X, V, W, ts = ds.load_clip_matrix(csv, stats)
    T = X.shape[0]
    acc = np.zeros((T, gs.N_NODES), dtype=np.float64)
    cnt = np.zeros((T, 1), dtype=np.float64)
    xb, vb, starts = [], [], []
    for s in range(0, max(1, T - win + 1), stride):
        e = s + win
        if e > T:
            break
        xb.append(X[s:e]); vb.append(V[s:e]); starts.append(s)
    if not xb:                                   # clip shorter than one window
        xb, vb, starts = [np.pad(X, ((0, win - T), (0, 0), (0, 0)))], \
                         [np.pad(V, ((0, win - T), (0, 0), (0, 0)))], [0]
    with torch.no_grad():
        for i in range(0, len(xb), 64):
            Xt = torch.from_numpy(np.stack(xb[i:i+64])).to(dev)
            Vt = torch.from_numpy(np.stack(vb[i:i+64])).to(dev)
            ne = masked_node_error(model(Xt), Xt, Vt).cpu().numpy()   # [b,win,N]
            for b, s in enumerate(starts[i:i+64]):
                span = min(win, T - s)
                acc[s:s+span] += ne[b, :span]
                cnt[s:s+span, 0] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt, ts                          # [T,N], [T]


def _windowize(frame_err, ts):
    """Aggregate per-frame node error into 2 s / 1 s windows → (starts_ms, [W,N])."""
    T = frame_err.shape[0]
    win_f, hop_f = int(WIN_MS / 1000 * FPS), int(HOP_MS / 1000 * FPS)
    starts_ms, rows = [], []
    for s in range(0, max(1, T - win_f + 1), hop_f):
        e = s + win_f
        if e > T:
            break
        starts_ms.append(float(ts[s]))
        rows.append(frame_err[s:e].mean(0))
    return np.array(starts_ms), np.array(rows)     # [W], [W,N]


def fit_normalizer(baseline_csv, model, stats, dev):
    """Per-node reconstruction-error mean/std over the baseline clip's windows."""
    fe, ts = _frame_node_error(baseline_csv, model, stats, dev)
    _, rows = _windowize(fe, ts)
    return rows.mean(0), rows.std(0) + 1e-6        # [N], [N]


def attribute_clip(csv, model, stats, norm, dev, file_index=None):
    """DataFrame: one row per 2 s window with per-node reconstruction-z + global."""
    mean, std = norm
    fe, ts = _frame_node_error(csv, model, stats, dev)
    starts, rows = _windowize(fe, ts)              # [W], [W,N]
    z = (rows - mean) / std                        # reconstruction-z per node
    df = pd.DataFrame({"start_time_ms": starts,
                       "end_time_ms": starts + WIN_MS})
    for n, name in enumerate(gs.NODE_NAMES):
        df[f"recon_z_{name}"] = z[:, n]
    df["recon_z_global"] = z.sum(1)                # summed over 11 nodes
    df["recon_z_max"] = z.max(1)
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
    print(out.filter(regex="start_time_ms|recon_z_(global|au_mouth|gaze|hand_left)").head(10).to_string())
