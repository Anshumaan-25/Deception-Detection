"""
verify_stgae.py — ST-GAE unit checks (CPU torch + synthetic data; no GPU, no real footage).

Covers the invariants ST_GAE_DESIGN calls out:
  1. graph_spec integrity (11 nodes, no duplicate feature, dims consistent)
  2. dataset masking (NaN gaps, non-speaking voice frames, low-conf face frames → invalid)
  3. feature-count-normalized loss (2-dim blink node counts equally with 18-dim voice)
  4. mask ⇒ exactly-zero gradient (a fully-masked node/feature never moves the weights)
  5. model determinism (same seed → identical forward)
  6. overfit-on-noise failure (fitting pure noise must NOT beat predict-baseline → loud fail)
Run: python tests/verify_stgae.py
"""
import os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # force CPU, deterministic
import torch

from stgae import graph_spec as gs
from stgae import dataset as ds
from stgae.model import STGAE, masked_node_error, stgae_loss

ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("  ✓", msg)


def _synth_csv(path, T=300, speaking_from=100, speaking_to=200, gap_at=None):
    """A synthetic frame CSV with all ST-GAE columns + aux columns."""
    rng = np.random.default_rng(0)
    data = {"timestamp": np.arange(T) * (1000.0 / 30.0), "frame_id": np.arange(T)}
    for c in gs.ALL_FEATURES:
        data[c] = rng.normal(size=T)
    data["is_audio_active"] = np.where((np.arange(T) >= speaking_from) &
                                       (np.arange(T) < speaking_to), 1.0, 0.0)
    data["face_confidence"] = np.full(T, 0.9)
    data["joint_confidence"] = np.full(T, 0.8)
    df = pd.DataFrame(data)
    if gap_at is not None:
        for c in gs.NODE_FEATURES["head_pose"]:
            df.loc[gap_at, c] = np.nan
    df.to_csv(path, index=False)
    return path


print("1. graph_spec integrity")
check(gs.N_NODES == 11, "11 nodes")
check(len(gs.ALL_FEATURES) == sum(gs.NODE_DIMS), "flat feature count == Σ node dims")
check(len(set(gs.ALL_FEATURES)) == len(gs.ALL_FEATURES), "no duplicate feature column")
check(all(len(r) == 11 for r in gs.prior_adjacency()), "adjacency is 11×11")
check(gs.NODE_DIMS[gs.NODE_INDEX["voice"]] == 18 and gs.NODE_DIMS[gs.NODE_INDEX["blink"]] == 2,
      "voice=18, blink=2 (the imbalance the loss must normalize)")

print("2. dataset masking")
with tempfile.TemporaryDirectory() as tmp:
    csv = _synth_csv(os.path.join(tmp, "s.csv"), gap_at=150)
    st = ds.fit_frame_stats(csv)
    X, valid, w, ts = ds.load_clip_matrix(csv, st)
    vi = gs.NODE_INDEX["voice"]
    # voice valid only where is_audio_active==1 (frames 100..199)
    check(valid[100:200, vi, :18].mean() == 1.0 and valid[:100, vi, :18].mean() == 0.0,
          "voice node valid iff is_audio_active==1")
    hp = gs.NODE_INDEX["head_pose"]
    check(valid[150, hp, 0] == 0.0, "NaN gap frame → that node/feature invalid")
    check(valid[151, hp, 0] == 1.0, "non-gap frame stays valid")
    # padding columns beyond a node's real dim are always invalid
    check(valid[:, gs.NODE_INDEX["blink"], 2:].sum() == 0.0, "blink padding (dims 2..17) all invalid")

print("3. feature-count-normalized loss (2-dim node == 18-dim node)")
torch.manual_seed(0)
B, T, N, D = 2, 10, gs.N_NODES, gs.MAX_NODE_DIM
x = torch.zeros(B, T, N, D); xhat = torch.zeros(B, T, N, D); valid = torch.zeros(B, T, N, D)
bi, vi = gs.NODE_INDEX["blink"], gs.NODE_INDEX["voice"]
# give blink (2 valid) and voice (18 valid) the SAME per-feature error of 1.0
for k in range(2):
    valid[:, :, bi, k] = 1.0; xhat[:, :, bi, k] = 1.0
for k in range(18):
    valid[:, :, vi, k] = 1.0; xhat[:, :, vi, k] = 1.0
ne = masked_node_error(xhat, x, valid)   # [B,T,N]
check(abs(float(ne[0, 0, bi]) - float(ne[0, 0, vi])) < 1e-6,
      f"blink ({float(ne[0,0,bi]):.3f}) == voice ({float(ne[0,0,vi]):.3f}) despite 2 vs 18 features")
check(abs(float(ne[0, 0, bi]) - 1.0) < 1e-6, "per-feature error 1.0 → node error 1.0 (÷ valid count)")

print("4. mask ⇒ exactly-zero gradient")
torch.manual_seed(1)
m = STGAE()
x = torch.randn(1, 90, N, D)
valid = torch.zeros(1, 90, N, D)
valid[:, :, gs.NODE_INDEX["au_mouth"], :6] = 1.0    # only au_mouth valid
w = torch.ones(1, 90)
loss = stgae_loss(m(x), x, valid, w)
loss.backward()
# the per-node DECODER head for a fully-masked node must get zero gradient
gb = m.dec_w.grad[gs.NODE_INDEX["blink"]]
check(torch.count_nonzero(gb) == 0, "fully-masked blink decoder head has zero gradient")
gm = m.dec_w.grad[gs.NODE_INDEX["au_mouth"]]
check(torch.count_nonzero(gm) > 0, "the one valid node (au_mouth) does receive gradient")

print("5. model determinism")
def fwd(seed):
    torch.manual_seed(seed); mm = STGAE()
    torch.manual_seed(123); xx = torch.randn(1, 90, N, D)
    with torch.no_grad():
        return mm(xx)
check(torch.allclose(fwd(7), fwd(7)), "same seed → identical forward")
check(not torch.allclose(fwd(7), fwd(8)), "different seed → different init")

print("6. overfit-on-noise failure fires")
with tempfile.TemporaryDirectory() as tmp:
    # pure-noise 'baseline' with NO temporal structure — the model cannot learn to
    # reconstruct it, so the fit must NOT beat predict-baseline (recon_ratio ≈ 1).
    csv = _synth_csv(os.path.join(tmp, "noise.csv"), T=2000, speaking_from=0, speaking_to=2000)
    from stgae.fit import fit_subject
    b = fit_subject(csv, os.path.join(tmp, "fit"), device="cpu",
                    max_epochs=40, patience=15, seed=0)
    # Structureless noise can't beat the ~0.5 lossy-compression floor; a real subject
    # baseline reconstructs to ~0.30–0.40. The gate (0.48) separates them → on noise
    # the loud-failure path must trigger.
    check(b["recon_ratio"] > 0.48, f"pure noise → recon_ratio {b['recon_ratio']:.3f} > 0.48 gate "
                                   "(degenerate-fit loud-failure path triggers)")

print(f"\nverify_stgae: {ok} checks passed — no GPU, no real footage.")
