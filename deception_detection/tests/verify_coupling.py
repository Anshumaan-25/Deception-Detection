"""
verify_coupling.py — coupling-model checks (CPU torch + synthetic data; no GPU, no
real footage). Covers the invariants COUPLING_MODEL_DESIGN pre-registers as Bar 0:

  1. mask isolation: the forward output is invariant to the target node's input,
     bit-for-bit, at full network depth (the leakage guarantee)
  2. vectorized 11-target expansion ≡ sequential per-node passes
  3. determinism (same seed → identical forward)
  4. feature-count-normalized target loss (2-dim blink == 18-dim voice)
  5. target-validity semantics (all-invalid target → zero loss AND zero gradient;
     an invalid target feature's value never reaches the loss)
  6. coupling recovery: a planted cross-node coupling is learned (per-node ratios
     separate coupled from independent nodes); fit gate reads HEALTHY
  7. attribution: baseline scores itself ≈0; BREAKING the planted coupling spikes
     exactly the broken node; a pure marginal scale shift (couplings intact — the
     simulated v1 domain-gap failure) stays far below the broken spike
  8. overfit-on-noise: no coupling structure → ratio ≈ 1 → degenerate gate fires

Run: python tests/verify_coupling.py
"""
import os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # force CPU, deterministic
import torch

from stgae import graph_spec as gs
from stgae import dataset as ds
from stgae.coupling_model import CouplingPredictor, coupling_loss, expand_all_masks
from stgae.coupling_fit import fit_subject_coupling
from stgae.coupling_attribute import load_model, fit_normalizer, attribute_clip

N, D = gs.N_NODES, gs.MAX_NODE_DIM
ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("  ✓", msg)


# ── synthetic coupled world ──────────────────────────────────────────────────
# Three smooth drivers feed every node EXCEPT the hands (independent noise).
# Mixing weights are drawn from a FIXED generator so baseline / broken / shifted
# CSVs share the exact same coupling structure where it is not deliberately broken.
INDEP_NODES = {"hand_left", "hand_right"}

def _coupled_csv(path, T=2400, seed=1, scale=1.0, break_au_mouth=False):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    U = np.stack([np.sin(2*np.pi*t/97 + 0.3), np.sin(2*np.pi*t/41 + 1.1),
                  np.sin(2*np.pi*t/149 + 2.0)], 1)
    U2 = np.stack([np.sin(2*np.pi*t/61 + 0.9), np.sin(2*np.pi*t/113 + 2.4),
                   np.sin(2*np.pi*t/37 + 0.2)], 1)   # independent drivers (the break)
    wrng = np.random.default_rng(7)                  # fixed mixes across all CSVs
    data = {"timestamp": t * (1000.0/30.0), "frame_id": t}
    for name, feats in gs.NODES:
        for c in feats:
            w = wrng.normal(size=3)
            if name in INDEP_NODES:
                col = rng.normal(size=T)
            else:
                drv = U2 if (break_au_mouth and name == "au_mouth") else U
                col = drv @ w + 0.3 * rng.normal(size=T)
            data[c] = col * scale
    data["is_audio_active"] = np.ones(T)
    data["face_confidence"] = np.full(T, 0.9)
    data["joint_confidence"] = np.ones(T)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


print("1. mask isolation — output invariant to the target node's input (bitwise)")
torch.manual_seed(0)
m = CouplingPredictor()
x = torch.randn(2, 90, N, D)
tgt = gs.NODE_INDEX["au_mouth"]
idx = torch.full((2,), tgt, dtype=torch.long)
with torch.no_grad():
    y1 = m(x, idx)
    x2 = x.clone(); x2[:, :, tgt, :] = 1e6 * torch.randn(2, 90, D)   # arbitrary garbage
    y2 = m(x2, idx)
check(torch.equal(y1, y2), "entire forward output bit-identical under target-input garbage")
with torch.no_grad():
    x3 = x.clone(); x3[:, :, gs.NODE_INDEX["voice"], :] += 1.0        # a VISIBLE node
    y3 = m(x3, idx)
check(not torch.equal(y1, y3), "changing a visible neighbor DOES change the prediction")

print("2. vectorized 11-target expansion ≡ sequential per-node passes")
torch.manual_seed(1)
xb = torch.randn(3, 90, N, D)
vb = (torch.rand(3, 90, N, D) > 0.3).float()
wb = torch.ones(3, 90)
Xe, Ve, We, mi = expand_all_masks(xb, vb, wb)
check(Xe.shape[0] == 3 * N and torch.equal(mi[:N], torch.arange(N)),
      "expansion is window-major/node-minor with per-replica mask index")
with torch.no_grad():
    ye = m(Xe, mi)
    for n in (0, 5, 10):
        ys = m(xb, torch.full((3,), n, dtype=torch.long))
        rows = torch.arange(3) * N + n
        check(torch.allclose(ye[rows], ys, atol=1e-5),
              f"vectorized pass for target node {n} matches the sequential pass")

print("3. determinism")
def fwd(seed):
    torch.manual_seed(seed); mm = CouplingPredictor()
    torch.manual_seed(123); xx = torch.randn(1, 90, N, D)
    with torch.no_grad():
        return mm(xx, torch.zeros(1, dtype=torch.long))
check(torch.equal(fwd(7), fwd(7)), "same seed → identical forward")
check(not torch.equal(fwd(7), fwd(8)), "different seed → different init")

print("4. feature-count-normalized target loss (blink 2-dim == voice 18-dim)")
xx = torch.zeros(1, 10, N, D); w1 = torch.ones(1, 10)
bi, vi = gs.NODE_INDEX["blink"], gs.NODE_INDEX["voice"]
losses = {}
for node, dim in ((bi, 2), (vi, 18)):
    xhat = torch.zeros(1, 10, N, D); valid = torch.zeros(1, 10, N, D)
    xhat[:, :, node, :dim] = 1.0; valid[:, :, node, :dim] = 1.0
    losses[node] = float(coupling_loss(xhat, xx, valid, w1,
                                       torch.tensor([node])))
check(abs(losses[bi] - losses[vi]) < 1e-6,
      f"blink ({losses[bi]:.3f}) == voice ({losses[vi]:.3f}) despite 2 vs 18 features")
check(abs(losses[bi] - 1.0) < 1e-6, "per-feature error 1.0 → target loss 1.0 (÷ valid count)")

print("5. target-validity semantics")
torch.manual_seed(2)
m5 = CouplingPredictor()
x5 = torch.randn(1, 90, N, D)
v5 = torch.ones(1, 90, N, D)
v5[:, :, vi, :] = 0.0                                 # target voice: NO valid features
loss = coupling_loss(m5(x5, torch.tensor([vi])), x5, v5, torch.ones(1, 90),
                     torch.tensor([vi]))
check(float(loss.detach()) == 0.0, "all-invalid target → loss exactly 0")
loss.backward()
check(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in m5.parameters()),
      "all-invalid target → exactly-zero gradient everywhere")
v6 = torch.ones(1, 90, N, D); v6[:, :, tgt, 0] = 0.0  # one invalid target feature
xa = torch.randn(1, 90, N, D); xb2 = xa.clone(); xb2[:, :, tgt, 0] = 999.0
with torch.no_grad():
    la = coupling_loss(m5(xa, torch.tensor([tgt])), xa, v6, torch.ones(1, 90), torch.tensor([tgt]))
    lb = coupling_loss(m5(xb2, torch.tensor([tgt])), xb2, v6, torch.ones(1, 90), torch.tensor([tgt]))
check(torch.equal(la, lb), "an invalid target feature's value never reaches the loss")

print("6. coupling recovery on a planted cross-node structure (small CPU fit)")
tmp = tempfile.mkdtemp()
base_csv = _coupled_csv(os.path.join(tmp, "base.csv"), seed=1)
bundle = fit_subject_coupling(base_csv, os.path.join(tmp, "fit"), device="cpu",
                              max_epochs=60, patience=20, seed=0)
check(bundle["coupling_ratio"] < 0.90,
      f"coupled world → ratio {bundle['coupling_ratio']:.3f} < 0.90 (gate reads HEALTHY)")
nr = bundle["node_ratio"]
check(nr["au_mouth"] < 0.80,
      f"driven node au_mouth is predictable from neighbors (ratio {nr['au_mouth']:.2f})")
check(nr["hand_left"] > 0.85,
      f"independent node hand_left is NOT predictable (ratio {nr['hand_left']:.2f})")

print("7. attribution — break vs domain-shift (the v1 failure, simulated)")
model = load_model(os.path.join(tmp, "fit", "coupling_model.pt"))
stats = ds.FrameStats.from_json(os.path.join(tmp, "fit", "frame_stats.json"))
norm = fit_normalizer(base_csv, model, stats, "cpu")
self_df = attribute_clip(base_csv, model, stats, norm, "cpu")
check(abs(float(self_df["coupling_z_au_mouth"].median())) < 0.7,
      "baseline scores itself ≈ 0 coupling-z")
break_csv = _coupled_csv(os.path.join(tmp, "break.csv"), seed=2, break_au_mouth=True)
brk = attribute_clip(break_csv, model, stats, norm, "cpu")
b_mouth = float(brk["coupling_z_au_mouth"].median())
b_hand = float(brk["coupling_z_hand_left"].median())
check(b_mouth > 3.0, f"broken coupling spikes au_mouth coupling-z (median {b_mouth:.1f} > 3)")
check(b_hand < 2.0, f"specificity: hand_left stays quiet (median {b_hand:.1f} < 2)")
shift_csv = _coupled_csv(os.path.join(tmp, "shift.csv"), seed=3, scale=1.3)
shf = attribute_clip(shift_csv, model, stats, norm, "cpu")
s_mouth = float(shf["coupling_z_au_mouth"].median())
check(s_mouth < b_mouth / 3.0,
      f"marginal scale-shift with couplings intact stays far below the break "
      f"({s_mouth:.1f} < {b_mouth:.1f}/3) — the v1 domain-gap failure does not recur")

print("8. overfit-on-noise → degenerate gate fires")
noise_csv = os.path.join(tmp, "noise.csv")
rng = np.random.default_rng(0)
Tn = 1500
dn = {"timestamp": np.arange(Tn) * (1000.0/30.0), "frame_id": np.arange(Tn)}
for c in gs.ALL_FEATURES:
    dn[c] = rng.normal(size=Tn)
dn["is_audio_active"] = np.ones(Tn); dn["face_confidence"] = np.full(Tn, 0.9)
dn["joint_confidence"] = np.ones(Tn)
pd.DataFrame(dn).to_csv(noise_csv, index=False)
bn = fit_subject_coupling(noise_csv, os.path.join(tmp, "fit_noise"), device="cpu",
                          max_epochs=30, patience=12, seed=0)
check(bn["coupling_ratio"] > 0.90,
      f"pure noise → ratio {bn['coupling_ratio']:.3f} > 0.90 (degenerate gate fires)")

print(f"\nverify_coupling: {ok} checks passed — no GPU, no real footage.")
