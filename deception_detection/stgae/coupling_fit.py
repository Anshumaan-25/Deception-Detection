"""
Coupling-model per-subject fit — learn the subject's cross-modal couplings from the
baseline clip only (COUPLING_MODEL_DESIGN §4).

Anti-overfit stack inherited from the v1 fit (temporal 80/20 split, early stop on val,
denoising noise + feature dropout on the VISIBLE features, fixed seed, loud degenerate
gate) with one change of meaning: the reference is predict-zero (ẑ=0 = the subject's
baseline mean for the hidden node), so

    ratio = val_prediction_error / val_predict_zero_error

directly answers "do this subject's channels carry information about each other?"
On structureless noise the neighbors carry nothing → ratio ≈ 1.0.
Gate (pre-registered): HEALTHY iff ratio < 0.90.

RTX 6000 Ada profile (COUPLING_MODEL_DESIGN §7): the whole baseline tensor set is moved
to the device ONCE (~10 MB — zero per-step host↔device transfer), the 11-target
validation pass is folded into the batch dimension, and TF32 matmuls are enabled on
CUDA. Identical code path runs the CPU test suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import graph_spec as gs
from . import dataset as ds
from .coupling_model import (COUPLING_VERSION, CouplingPredictor, coupling_loss,
                             expand_all_masks)
from .model import masked_node_error


def _seed(s: int = 0):
    torch.manual_seed(s); np.random.seed(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _setup_device(device: str) -> str:
    dev = device if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        # Ada tensor cores; the pre-registered bars are rank-based → immune to the
        # tiny numeric differences vs CPU. fp16 AMP deliberately NOT used (§7).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return dev


def _stack_to(clips, dev):
    X = torch.from_numpy(np.stack([c[0] for c in clips])).to(dev)
    V = torch.from_numpy(np.stack([c[1] for c in clips])).to(dev)
    W = torch.from_numpy(np.stack([c[2] for c in clips])).to(dev)
    return X, V, W


def _target_errors(xhat, x, valid, weight, mask_idx):
    """Per-sample weighted target-node error: (num [B], den [B]) with
    num/den = the sample's mean prediction error over target-valid frames."""
    ne = masked_node_error(xhat, x, valid)                        # [B,T,N]
    b = torch.arange(x.shape[0], device=x.device)
    err = ne[b, :, mask_idx]                                      # [B,T]
    w = weight * (valid[b, :, mask_idx].sum(-1) > 0).to(weight.dtype)
    return (err * w).sum(-1), w.sum(-1)                           # [B], [B]


@torch.no_grad()
def _eval_split(model, X, V, W, batch):
    """All-11-target validation pass. Returns (pred_num, pred_den, zero_num, zero_den)
    each [M*N] ordered sample-major (window b, target n at b*N+n)."""
    pn, pd, zn, zd = [], [], [], []
    for i in range(0, X.shape[0], batch):
        Xe, Ve, We, mi = expand_all_masks(X[i:i+batch], V[i:i+batch], W[i:i+batch])
        n1, d1 = _target_errors(model(Xe, mi), Xe, Ve, We, mi)
        n0, d0 = _target_errors(torch.zeros_like(Xe), Xe, Ve, We, mi)
        pn.append(n1); pd.append(d1); zn.append(n0); zd.append(d0)
    return (torch.cat(pn), torch.cat(pd), torch.cat(zn), torch.cat(zd))


def fit_subject_coupling(baseline_csv: str, out_dir: str, *, device: str = "cuda",
                         clip_len: int = 90, stride: int = 15, embed: int = 16,
                         blocks: int = 2, max_epochs: int = 400, patience: int = 40,
                         lr: float = 2e-3, batch: int = 64, noise: float = 0.1,
                         drop: float = 0.25, seed: int = 0):
    _seed(seed)
    dev = _setup_device(device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    stats = ds.fit_frame_stats(baseline_csv)
    clips = ds.make_clips(baseline_csv, stats, clip_len, stride)
    if len(clips) < 20:
        raise RuntimeError(f"baseline too short: {len(clips)} clips (<20). "
                           "Coupling model cannot fit a subject on this little data.")
    # temporal split: first 80% train, last 20% val (random would leak via overlap)
    cut = int(len(clips) * 0.8)
    # device-resident tensors — the entire fit is transfer-free after this point
    Xtr, Vtr, Wtr = _stack_to(clips[:cut], dev)
    Xva, Vva, Wva = _stack_to(clips[cut:], dev)
    M_tr = Xtr.shape[0]

    model = CouplingPredictor(embed=embed, blocks=blocks).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def val_losses():
        model.eval()
        pn, pd, zn, zd = _eval_split(model, Xva, Vva, Wva, batch)
        pred = float(pn.sum() / pd.sum().clamp_min(1e-6))
        zero = float(zn.sum() / zd.sum().clamp_min(1e-6))
        return pred, zero

    _, ref = val_losses()   # predict-zero reference is model-independent
    best, best_state, best_ep, hist = float("inf"), None, -1, []
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(M_tr, device=dev)
        for i in range(0, M_tr, batch):
            j = perm[i:i+batch]
            X, V, W = Xtr[j], Vtr[j], Wtr[j]
            mask_idx = torch.randint(0, gs.N_NODES, (X.shape[0],), device=dev)
            # denoising on the VISIBLE features (the target's input is replaced by
            # the mask token anyway); target stays CLEAN in the loss
            xin = X + noise * torch.randn_like(X) * V
            if drop > 0:
                xin = xin * (torch.rand_like(X) > drop).float()
            opt.zero_grad()
            loss = coupling_loss(model(xin, mask_idx), X, V, W, mask_idx)
            loss.backward(); opt.step()
        vl, _ = val_losses()
        hist.append(vl)
        if vl < best - 1e-4:
            best, best_state, best_ep = vl, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, ep
        if ep - best_ep >= patience:
            break

    model.load_state_dict(best_state); model.to(dev).eval()
    ratio = best / ref if ref > 1e-9 else 1.0

    # per-node predictability ratios + residual stats on held-out val windows
    pn, pd, zn, zd = _eval_split(model, Xva, Vva, Wva, batch)
    Mv = Xva.shape[0]
    pn = pn.reshape(Mv, gs.N_NODES); pd = pd.reshape(Mv, gs.N_NODES)
    zn = zn.reshape(Mv, gs.N_NODES); zd = zd.reshape(Mv, gs.N_NODES)
    node_ratio, resid_mean, resid_std = [], [], []
    for n in range(gs.N_NODES):
        p_tot = float(pn[:, n].sum() / pd[:, n].sum().clamp_min(1e-6))
        z_tot = float(zn[:, n].sum() / zd[:, n].sum().clamp_min(1e-6))
        node_ratio.append(p_tot / z_tot if z_tot > 1e-9 else 1.0)
        per_win = (pn[:, n] / pd[:, n].clamp_min(1e-6)).cpu().numpy()
        has = pd[:, n].cpu().numpy() > 0
        resid_mean.append(float(per_win[has].mean()) if has.any() else 0.0)
        resid_std.append(float(per_win[has].std() + 1e-6) if has.any() else 1.0)

    bundle = {
        "coupling_version": COUPLING_VERSION,
        "spec_version": gs.SPEC_VERSION,
        "config": {"clip_len": clip_len, "stride": stride, "embed": embed,
                   "blocks": blocks, "seed": seed},
        "n_params": model.n_params(),
        "baseline_csv": str(baseline_csv),
        "n_clips": len(clips), "n_train": M_tr, "n_val": Mv,
        "val_loss": best, "predict_zero_loss": ref, "coupling_ratio": ratio,
        "node_ratio": {gs.NODE_NAMES[n]: node_ratio[n] for n in range(gs.N_NODES)},
        "best_epoch": best_ep, "val_hist": hist,
        "node_resid_mean": resid_mean, "node_resid_std": resid_std,
        "adjacency": model.adjacency().detach().cpu().tolist(),
    }
    torch.save(model.state_dict(), out / "coupling_model.pt")
    stats.to_json(str(out / "frame_stats.json"))
    (out / "fit_report.json").write_text(json.dumps(bundle, indent=2))

    healthy = ratio < 0.90   # pre-registered gate (COUPLING_MODEL_DESIGN §4)
    top = sorted(bundle["node_ratio"].items(), key=lambda kv: kv[1])[:3]
    print(f"[coupling-fit] {model.n_params()} params | clips {len(clips)} "
          f"(train {M_tr}/val {Mv}) | val {best:.3f} vs predict-zero {ref:.3f} "
          f"→ ratio {ratio:.3f} ({'HEALTHY' if healthy else 'DEGENERATE'}) "
          f"| best epoch {best_ep} | most-predictable: "
          + ", ".join(f"{k} {v:.2f}" for k, v in top))
    if not healthy:
        print("  ⚠️  the neighbors do not predict any node better than the baseline "
              "mean — no usable coupling structure; treat attributions as noise "
              "(COUPLING_MODEL_DESIGN §4 gate).")
    return bundle


if __name__ == "__main__":
    import sys
    fit_subject_coupling(sys.argv[1], sys.argv[2])
