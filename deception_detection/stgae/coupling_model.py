"""
Predictive cross-modal coupling model — masked-node prediction over the 11-node graph.

v2 of the graph line (Documentation/COUPLING_MODEL_DESIGN.md). The reconstruction
ST-GAE (model.py) was falsified 2026-07-09; this model changes the QUESTION: hide one
node entirely and predict it from the other 10. The per-node prediction residual reads
"this channel stopped moving the way this subject's other channels say it should" — a
decoupling detector, not a distance-from-baseline meter.

Leakage guarantee: the target node's encoder stream is REPLACED by a learned per-node
mask token before any message passing, so the whole forward output is invariant to the
target's input at any network depth — bit-for-bit (verify_coupling.py test 1). This is
the only safe construction: with message passing, information flows n→m→n in two hops,
so merely down-weighting a self-edge would leak.

No bottleneck: v1's temporal/channel squeeze existed solely to stop an autoencoder from
copying its input. A masked node cannot be copied, so it is removed (~15k params).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import graph_spec as gs
from .model import _STBlock, masked_node_error

N, D = gs.N_NODES, gs.MAX_NODE_DIM

COUPLING_VERSION = "coupling-v1"


class CouplingPredictor(nn.Module):
    def __init__(self, embed: int = 16, blocks: int = 2):
        super().__init__()
        self.embed = embed
        # per-node encoder / decoder (same layout as v1)
        self.enc_w = nn.Parameter(torch.empty(N, D, embed))
        self.enc_b = nn.Parameter(torch.zeros(N, embed))
        self.dec_w = nn.Parameter(torch.empty(N, embed, D))
        self.dec_b = nn.Parameter(torch.zeros(N, D))
        nn.init.xavier_uniform_(self.enc_w); nn.init.xavier_uniform_(self.dec_w)
        # the leakage barrier: one learned token per node, substituted for the
        # target node's encoded stream
        self.mask_token = nn.Parameter(torch.zeros(N, embed))
        A0 = torch.tensor(gs.prior_adjacency(), dtype=torch.float32)
        self.A_logit = nn.Parameter(torch.log(A0 + 1e-3))
        self.blocks = nn.ModuleList([_STBlock(embed) for _ in range(blocks)])

    def adjacency(self):
        return torch.softmax(self.A_logit, dim=1)

    def forward(self, x: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        """x [B,T,N,D]; mask_idx int64 [B] — the target node hidden in each sample.
        Returns xhat [B,T,N,D]; only the mask_idx slice is a prediction-from-neighbors,
        the rest is incidental decode."""
        A = self.adjacency()
        h = torch.einsum("btnd,nde->btne", x, self.enc_w) + self.enc_b   # [B,T,N,E]
        # substitute the target stream with its mask token (out-of-place: autograd-clean
        # and kills every path from x[target] into the network exactly)
        m = torch.nn.functional.one_hot(mask_idx, N).to(h.dtype)[:, None, :, None]  # [B,1,N,1]
        h = h * (1.0 - m) + self.mask_token[None, None, :, :] * m
        for blk in self.blocks:
            h = blk(h, A)
        return torch.einsum("btne,ned->btnd", h, self.dec_w) + self.dec_b

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def coupling_loss(xhat, x, valid, weight, mask_idx):
    """Prediction loss on the TARGET node only: feature-count-normalized (÷ valid
    feature count — masked_node_error), frames where the target has zero valid
    features carry zero weight (a non-speaking voice frame is not a fake 0-error).
    xhat/x/valid [B,T,N,D]; weight [B,T]; mask_idx [B]. Scalar."""
    ne = masked_node_error(xhat, x, valid)                       # [B,T,N]
    b = torch.arange(x.shape[0], device=x.device)
    err = ne[b, :, mask_idx]                                     # [B,T]
    tgt_has_valid = (valid[b, :, mask_idx].sum(-1) > 0).to(weight.dtype)
    w = weight * tgt_has_valid
    return (err * w).sum() / w.sum().clamp_min(1e-6)


def expand_all_masks(X, V, W):
    """Fold the 11 target choices into the batch dim: one big forward instead of 11
    sequential ones (the RTX-profile fast path; equivalence unit-tested).
    X/V [B,T,N,D], W [B,T] → (Xe [B*N,...], Ve, We, mask_idx [B*N]) where sample
    b*N + n targets node n of window b."""
    B = X.shape[0]
    Xe = X.repeat_interleave(N, dim=0)
    Ve = V.repeat_interleave(N, dim=0)
    We = W.repeat_interleave(N, dim=0)
    mask_idx = torch.arange(N, device=X.device).repeat(B)
    return Xe, Ve, We, mask_idx


if __name__ == "__main__":
    torch.manual_seed(0)
    m = CouplingPredictor()
    print(f"params: {m.n_params()}")
    x = torch.randn(4, 90, N, D)
    valid = (torch.rand(4, 90, N, D) > 0.3).float()
    w = torch.rand(4, 90)
    idx = torch.tensor([0, 3, 9, 10])
    xhat = m(x, idx)
    print("pred shape:", tuple(xhat.shape))
    print("loss:", float(coupling_loss(xhat, x, valid, w, idx)))
