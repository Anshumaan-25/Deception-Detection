"""
ST-GAE model — a tiny masked spatio-temporal graph autoencoder (<100k params).

Plain torch + einsum (no torch-geometric — an 11-node dense adjacency needs none,
and we add zero deps to spovnob_env). Architecture per ST_GAE_DESIGN §3:

  input [B,T,N,D] → per-node encoder → 3× ST-block → temporal ↓2 (latent)
                  → temporal ↑2 → 3× ST-block → per-node head → recon [B,T,N,D]

Loss is the mandated **feature-count-normalized**, masked, confidence-weighted MSE:
each node's squared error is divided by its number of VALID features at that frame
before summing across nodes, so the 2-dim blink node is never drowned by the 18-dim
voice node (ST_GAE_DESIGN §3, the condition of the 11-node lock).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import graph_spec as gs

N, D = gs.N_NODES, gs.MAX_NODE_DIM


class _STBlock(nn.Module):
    """Spatial graph-conv (learned adjacency, shared across the batch) + temporal
    conv + GeLU, residual."""
    def __init__(self, ch: int, k: int = 9):
        super().__init__()
        self.spatial = nn.Linear(ch, ch)
        self.temporal = nn.Conv1d(ch, ch, kernel_size=k, padding=k // 2, groups=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(ch)

    def forward(self, h, A):                       # h [B,T,N,C], A [N,N]
        B, T, Nn, C = h.shape
        hs = torch.einsum("nm,btmc->btnc", A, h)   # message passing
        hs = self.spatial(hs)
        # temporal conv per node: fold N into batch
        ht = hs.permute(0, 2, 3, 1).reshape(B * Nn, C, T)
        ht = self.temporal(ht).reshape(B, Nn, C, T).permute(0, 3, 1, 2)
        return self.norm(h + self.act(ht))


class STGAE(nn.Module):
    def __init__(self, embed: int = 16, blocks: int = 3, latent_ch: int = 4):
        super().__init__()
        self.embed = embed
        self.latent_ch = latent_ch
        # per-node encoder / decoder: separate weights per node (N,D,embed)
        self.enc_w = nn.Parameter(torch.empty(N, D, embed))
        self.enc_b = nn.Parameter(torch.zeros(N, embed))
        self.dec_w = nn.Parameter(torch.empty(N, embed, D))
        self.dec_b = nn.Parameter(torch.zeros(N, D))
        nn.init.xavier_uniform_(self.enc_w); nn.init.xavier_uniform_(self.dec_w)
        # learned adjacency, seeded from the functional prior (log-space → softmax)
        A0 = torch.tensor(gs.prior_adjacency(), dtype=torch.float32)
        self.A_logit = nn.Parameter(torch.log(A0 + 1e-3))
        self.enc_blocks = nn.ModuleList([_STBlock(embed) for _ in range(blocks)])
        self.dec_blocks = nn.ModuleList([_STBlock(embed) for _ in range(blocks)])
        # Bottleneck: compress BOTH temporally (↓2) and channel-wise (embed→latent_ch).
        # This must be genuinely compressive — latent = N·latent_ch·(T/2) < real inputs —
        # or the autoencoder learns identity (copies) and reconstructs anomalies just as
        # well as baseline, giving no anomaly signal (verify_stgae test 6).
        self.down = nn.Conv1d(embed, latent_ch, 4, stride=2, padding=1)
        self.up = nn.ConvTranspose1d(latent_ch, embed, 4, stride=2, padding=1)

    def adjacency(self):
        return torch.softmax(self.A_logit, dim=1)   # row-normalized message passing

    def forward(self, x):                            # x [B,T,N,D]
        B, T, Nn, Dd = x.shape
        A = self.adjacency()
        h = torch.einsum("btnd,nde->btne", x, self.enc_w) + self.enc_b   # [B,T,N,embed]
        for blk in self.enc_blocks:
            h = blk(h, A)
        # temporal downsample→upsample (fold N into batch)
        C = self.embed
        hd = h.permute(0, 2, 3, 1).reshape(B * Nn, C, T)
        z = self.down(hd)
        hu = self.up(z)
        if hu.shape[-1] != T:                        # guard odd T
            hu = torch.nn.functional.interpolate(hu, size=T, mode="linear", align_corners=False)
        h = hu.reshape(B, Nn, C, T).permute(0, 3, 1, 2)
        for blk in self.dec_blocks:
            h = blk(h, A)
        xhat = torch.einsum("btne,ned->btnd", h, self.dec_w) + self.dec_b  # [B,T,N,D]
        return xhat

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Feature-count-normalized masked loss ─────────────────────────────────────
_NODE_DIM = torch.tensor(gs.NODE_DIMS, dtype=torch.float32)   # real F_n per node


def masked_node_error(xhat, x, valid):
    """Per-(clip,frame,node) mean squared error over VALID features only —
    i.e. Σ_f valid·(x̂−x)² / max(1, Σ_f valid). Shape [B,T,N]. This is the ÷F_n
    normalization (active-feature count), so every node is on equal footing."""
    se = valid * (xhat - x) ** 2                     # [B,T,N,D]
    num = se.sum(-1)                                 # [B,T,N]
    den = valid.sum(-1).clamp_min(1.0)               # [B,T,N]
    return num / den


def stgae_loss(xhat, x, valid, weight):
    """Confidence-weighted sum over nodes of the feature-count-normalized error.
    weight [B,T] (joint_confidence). Returns scalar."""
    node_err = masked_node_error(xhat, x, valid)     # [B,T,N]
    per_frame = node_err.sum(-1)                      # [B,T] — Σ over 11 nodes
    w = weight                                        # [B,T]
    return (per_frame * w).sum() / w.sum().clamp_min(1e-6)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = STGAE()
    print(f"params: {m.n_params()}")
    x = torch.randn(4, 90, N, D)
    valid = (torch.rand(4, 90, N, D) > 0.3).float()
    w = torch.rand(4, 90)
    xhat = m(x)
    print("recon shape:", tuple(xhat.shape))
    print("loss:", float(stgae_loss(xhat, x, valid, w)))
    print("adjacency row sums:", [round(float(r), 3) for r in m.adjacency().sum(1)])
