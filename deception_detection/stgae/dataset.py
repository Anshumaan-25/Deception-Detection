"""
ST-GAE dataset — frame-level feature CSV → masked, baseline-normalized clip tensors.

Mirrors the calibration doctrine at the FRAME level: fit per-feature mean/std on the
baseline clip's 30 fps CSV (analogous to BaselineStats, which is windowed-only), z-score
every clip against it, and carry an explicit validity mask so NaN gaps, non-speaking
frames (voice node) and low-confidence face frames contribute exactly zero to the loss.

Tensors per clip window (numpy):
    X      [T, N, D]  baseline-z features, padded to D=MAX_NODE_DIM, invalid→0
    valid  [T, N, D]  1.0 where the feature is real (not NaN/masked/padding)
    weight [T]        per-frame joint_confidence (loss weight)
with T = clip length (default 90 = 3 s @ 30 fps), stride 15.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import graph_spec as gs


@dataclass
class FrameStats:
    """Per-feature baseline mean/std over the baseline clip's frame CSV."""
    means: dict[str, float]
    stds: dict[str, float]
    baseline_frames: int
    source_csv: str
    spec_version: str = gs.SPEC_VERSION

    def to_json(self, path: str) -> str:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        return str(path)

    @classmethod
    def from_json(cls, path: str) -> "FrameStats":
        return cls(**json.loads(Path(path).read_text()))


def fit_frame_stats(baseline_csv: str) -> FrameStats:
    """Fit per-feature mean/std on the baseline clip. Zero-variance features get
    std=1 (constant → contributes ~0 to loss, never NaN/inf)."""
    df = pd.read_csv(baseline_csv)
    means, stds = {}, {}
    for c in gs.ALL_FEATURES:
        col = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(dtype=float)
        m = float(col.mean()) if col.notna().any() else 0.0
        s = float(col.std(ddof=1)) if col.notna().sum() > 1 else 0.0
        means[c] = 0.0 if np.isnan(m) else m
        stds[c] = 1.0 if (np.isnan(s) or s < 1e-8) else s
    return FrameStats(means=means, stds=stds, baseline_frames=len(df),
                      source_csv=str(baseline_csv))


def _normalized_matrix(csv: str, stats: FrameStats):
    """Return (X_full [T,N,D], valid_full [T,N,D], weight [T], timestamps [T])."""
    df = pd.read_csv(csv)
    T = len(df)
    N, D = gs.N_NODES, gs.MAX_NODE_DIM
    X = np.zeros((T, N, D), dtype=np.float32)
    valid = np.zeros((T, N, D), dtype=np.float32)

    # node-level extra masks
    if gs.VOICE_MASK_COL in df:
        voice_ok = (pd.to_numeric(df[gs.VOICE_MASK_COL], errors="coerce").to_numpy() == 1.0)
    else:
        voice_ok = np.ones(T, dtype=bool)
    if gs.FACE_MASK_COL in df:
        face_ok = (pd.to_numeric(df[gs.FACE_MASK_COL], errors="coerce").fillna(0).to_numpy()
                   >= gs.FACE_CONF_MIN)
    else:
        face_ok = np.ones(T, dtype=bool)

    for n, (name, feats) in enumerate(gs.NODES):
        node_ok = np.ones(T, dtype=bool)
        if name == gs.VOICE_NODE:
            node_ok = voice_ok
        elif name in gs.FACE_NODES:
            node_ok = face_ok
        for k, c in enumerate(feats):
            raw = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) if c in df \
                else np.full(T, np.nan)
            z = (raw - stats.means[c]) / stats.stds[c]
            ok = np.isfinite(z) & node_ok
            X[:, n, k] = np.where(ok, z, 0.0)
            valid[:, n, k] = ok.astype(np.float32)

    if gs.WEIGHT_COL in df:
        w = pd.to_numeric(df[gs.WEIGHT_COL], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        w = np.ones(T, dtype=np.float32)
    ts = (pd.to_numeric(df["timestamp"], errors="coerce").to_numpy()
          if "timestamp" in df else np.arange(T) * (1000.0 / 30.0))
    return X, valid, w, ts


def load_clip_matrix(csv: str, stats: FrameStats):
    """Full-clip tensors (no windowing) — used at attribution time."""
    return _normalized_matrix(csv, stats)


def make_clips(csv: str, stats: FrameStats, clip_len: int = 90, stride: int = 15):
    """List of (X, valid, weight) windows of length clip_len. Drops windows whose
    total valid fraction is < 0.2 (a gap that long carries no reconstructable signal)."""
    X, valid, w, _ = _normalized_matrix(csv, stats)
    T = X.shape[0]
    out = []
    for s in range(0, max(1, T - clip_len + 1), stride):
        e = s + clip_len
        if e > T:
            break
        vx, vv, vw = X[s:e], valid[s:e], w[s:e]
        if vv.mean() < 0.2:
            continue
        out.append((vx, vv, vw))
    return out


if __name__ == "__main__":
    import sys
    base = sys.argv[1]  # baseline frame CSV
    st = fit_frame_stats(base)
    print(f"fitted frame stats on {st.baseline_frames} frames, {len(st.means)} features")
    clips = make_clips(base, st)
    print(f"{len(clips)} clips of shape {clips[0][0].shape}; "
          f"mean valid frac {np.mean([c[1].mean() for c in clips]):.2%}")
