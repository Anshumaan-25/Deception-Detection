"""
ST-GAE per-subject fit — learn "normal" from the baseline clip only.

Anti-overfit stack (ST_GAE_DESIGN §4), because the fit budget is ~2 min of data:
  • <100k params (model.py)                • temporal 80/20 split, not random
  • early-stop on val reconstruction        • denoising augmentation (noise + feature dropout)
  • fixed seed + deterministic              • loud failure if the fit can't beat "predict baseline"

Writes a FittedSTGAE bundle: model weights, FrameStats, per-node baseline residual
mean/std (for the attribution reconstruction-z), and the val trajectory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import graph_spec as gs
from . import dataset as ds
from .model import STGAE, stgae_loss, masked_node_error


def _seed(s: int = 0):
    torch.manual_seed(s); np.random.seed(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batches(clips, bs, shuffle):
    idx = np.arange(len(clips))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        X = torch.from_numpy(np.stack([clips[k][0] for k in j]))
        V = torch.from_numpy(np.stack([clips[k][1] for k in j]))
        W = torch.from_numpy(np.stack([clips[k][2] for k in j]))
        yield X, V, W


def fit_subject(baseline_csv: str, out_dir: str, *, device: str = "cuda",
                clip_len: int = 90, stride: int = 15, embed: int = 16, blocks: int = 3,
                max_epochs: int = 400, patience: int = 40, lr: float = 2e-3,
                batch: int = 32, noise: float = 0.1, drop: float = 0.25, seed: int = 0):
    _seed(seed)
    dev = device if torch.cuda.is_available() else "cpu"
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    stats = ds.fit_frame_stats(baseline_csv)
    clips = ds.make_clips(baseline_csv, stats, clip_len, stride)
    if len(clips) < 20:
        raise RuntimeError(f"baseline too short: {len(clips)} clips (<20). "
                           "ST-GAE cannot fit a subject on this little data.")
    # temporal split: first 80% train, last 20% val (random would leak via overlap)
    cut = int(len(clips) * 0.8)
    train, val = clips[:cut], clips[cut:]

    model = STGAE(embed=embed, blocks=blocks).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def eval_loss(split):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for X, V, W in _batches(split, batch, shuffle=False):
                X, V, W = X.to(dev), V.to(dev), W.to(dev)
                tot += float(stgae_loss(model(X), X, V, W)) * len(X); n += len(X)
        return tot / max(1, n)

    def predict_zero_loss(split):
        # reference: reconstruct the baseline mean (x̂=0 in z-space). If the model
        # can't beat this, it learned nothing (Bar-4 small-baseline failure).
        tot, n = 0.0, 0
        with torch.no_grad():
            for X, V, W in _batches(split, batch, shuffle=False):
                z = torch.zeros_like(X)
                tot += float(stgae_loss(z, X, V, W)) * len(X); n += len(X)
        return tot / max(1, n)

    ref = predict_zero_loss(val)
    best, best_state, best_ep, hist = float("inf"), None, -1, []
    for ep in range(max_epochs):
        model.train()
        for X, V, W in _batches(train, batch, shuffle=True):
            X, V, W = X.to(dev), V.to(dev), W.to(dev)
            xin = X + noise * torch.randn_like(X) * V           # denoising noise on valid feats
            if drop > 0:                                        # feature dropout (denoising)
                keep = (torch.rand_like(X) > drop).float()
                xin = xin * keep
            opt.zero_grad()
            loss = stgae_loss(model(xin), X, V, W)              # reconstruct CLEAN target
            loss.backward(); opt.step()
        vl = eval_loss(val)
        hist.append(vl)
        if vl < best - 1e-4:
            best, best_state, best_ep = vl, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, ep
        if ep - best_ep >= patience:
            break

    model.load_state_dict(best_state)
    ratio = best / ref if ref > 1e-9 else 1.0
    # per-node baseline residual stats on VAL (held-out) → reconstruction-z reference
    resid = _node_residual_stats(model, val, dev, batch)

    bundle = {
        "spec_version": gs.SPEC_VERSION,
        "config": {"clip_len": clip_len, "stride": stride, "embed": embed,
                   "blocks": blocks, "seed": seed},
        "n_params": model.n_params(),
        "baseline_csv": str(baseline_csv),
        "n_clips": len(clips), "n_train": len(train), "n_val": len(val),
        "val_loss": best, "predict_zero_loss": ref, "recon_ratio": ratio,
        "best_epoch": best_ep, "val_hist": hist,
        "node_resid_mean": resid["mean"], "node_resid_std": resid["std"],
        "adjacency": model.adjacency().detach().cpu().tolist(),
    }
    torch.save(model.state_dict(), out / "stgae_model.pt")
    stats.to_json(str(out / "frame_stats.json"))
    (out / "fit_report.json").write_text(json.dumps(bundle, indent=2))

    # A compressive AE hits a ~0.5 "lossy-compression floor" on structureless data
    # (it captures per-window variance without learning structure); a real subject
    # baseline reconstructs meaningfully better (~0.30–0.40 observed). Gate between them.
    healthy = ratio < 0.48
    print(f"[fit] {model.n_params()} params | clips {len(clips)} (train {len(train)}/val {len(val)}) "
          f"| val {best:.3f} vs predict-zero {ref:.3f} → ratio {ratio:.3f} "
          f"({'HEALTHY' if healthy else 'DEGENERATE — baseline too brittle'}) | best epoch {best_ep}")
    if not healthy:
        print("  ⚠️  ST-GAE did not meaningfully beat the baseline-mean predictor; "
              "per ST_GAE_DESIGN §4 the fit is too brittle — treat attributions with suspicion.")
    return bundle


def _node_residual_stats(model, split, dev, batch):
    """Per-node reconstruction-error mean/std over held-out baseline windows,
    aggregated to the 2 s / 1 s deviation grid (45-frame window, 30-frame hop)."""
    model.eval()
    per_node = [[] for _ in range(gs.N_NODES)]
    with torch.no_grad():
        for X, V, W in _batches(split, batch, shuffle=False):
            X, V, W = X.to(dev), V.to(dev), W.to(dev)
            ne = masked_node_error(model(X), X, V)   # [B,T,N]
            # frame-level → collapse over T (a clip ≈ its own window here)
            per = ne.mean(1).cpu().numpy()            # [B,N]
            for n in range(gs.N_NODES):
                per_node[n].extend(per[:, n].tolist())
    mean = [float(np.mean(v)) if v else 0.0 for v in per_node]
    std = [float(np.std(v) + 1e-6) if v else 1.0 for v in per_node]
    return {"mean": mean, "std": std}


if __name__ == "__main__":
    import sys
    fit_subject(sys.argv[1], sys.argv[2])
