"""
ST-GAE graph specification — the single source of truth for nodes, edges and masks.

Pure-Python (no torch/pandas), so it can be imported anywhere and unit-tested
without the ML stack. Encodes every §2 decision of Documentation/ST_GAE_DESIGN.md:
11 semantic nodes over the 30 fps frame-level feature CSV, a functional-prior
adjacency (learned but seeded here), and per-node mask rules.

The node set is LOCKED at 11 (2026-07-09 review). If you change a node's feature
list, bump SPEC_VERSION — fitted models are keyed on it.
"""
from __future__ import annotations

SPEC_VERSION = "stgae-graph-v1"

# ── Nodes: (name, [frame-CSV feature columns]) in fixed order ────────────────
# Order is the node index used everywhere downstream; do not reorder.
NODES: list[tuple[str, list[str]]] = [
    ("head_pose",  ["head_yaw", "head_pitch", "head_roll"]),
    ("gaze",       ["gaze_x", "gaze_y", "gaze_z", "gaze_velocity"]),
    ("blink",      ["ear", "is_blinking"]),
    ("au_upper",   ["AU1", "AU2", "AU4", "AU1_velocity", "AU2_velocity", "AU4_velocity"]),
    ("au_mid",     ["AU6", "AU9", "AU6_velocity", "AU9_velocity"]),
    ("au_mouth",   ["AU12", "AU25", "AU26", "AU12_velocity", "AU25_velocity", "AU26_velocity"]),
    ("hand_left",  ["left_wrist_x", "left_wrist_y", "left_wrist_z",
                    "left_wrist_velocity", "left_hand_face_distance"]),
    ("hand_right", ["right_wrist_x", "right_wrist_y", "right_wrist_z",
                    "right_wrist_velocity", "right_hand_face_distance"]),
    ("body",       ["macro_motion_energy", "postural_stillness", "nose_x", "nose_y", "nose_z"]),
    ("voice",      [f"frame_wavlm_latent_{i}" for i in range(16)]
                   + ["frame_prosodic_velocity", "frame_acoustic_energy_rms"]),
    ("congruence", ["is_audio_active", "mismatch_incongruence", "silent_incongruence"]),
]

NODE_NAMES: list[str] = [n for n, _ in NODES]
NODE_INDEX: dict[str, int] = {n: i for i, n in enumerate(NODE_NAMES)}
NODE_FEATURES: dict[str, list[str]] = {n: f for n, f in NODES}
NODE_DIMS: list[int] = [len(f) for _, f in NODES]          # F_n per node (2..18)
N_NODES = len(NODES)                                        # 11
MAX_NODE_DIM = max(NODE_DIMS)                               # 18 (voice)

# Flat ordered feature list + per-node column slices into it.
ALL_FEATURES: list[str] = [c for _, feats in NODES for c in feats]
_pos = 0
NODE_SLICES: dict[str, tuple[int, int]] = {}
for _n, _f in NODES:
    NODE_SLICES[_n] = (_pos, _pos + len(_f))
    _pos += len(_f)

# ── Mask policy ──────────────────────────────────────────────────────────────
# Beyond per-value NaN (any node, any feature → that value contributes 0 loss):
FACE_NODES = {"head_pose", "gaze", "blink", "au_upper", "au_mid", "au_mouth"}
VOICE_NODE = "voice"
# voice node is additionally masked wherever the target is not verifiably
# speaking; face nodes wherever the face isn't confidently tracked.
VOICE_MASK_COL = "is_audio_active"        # keep frames where == 1
FACE_MASK_COL = "face_confidence"         # keep frames where >= FACE_CONF_MIN
FACE_CONF_MIN = 0.30
# Per-frame loss weight (multiplies every node's loss on that frame).
WEIGHT_COL = "joint_confidence"
# Columns the dataset must read from the CSV but never reconstructs.
AUX_COLS = [VOICE_MASK_COL, FACE_MASK_COL, WEIGHT_COL, "timestamp"]

# ── Functional-prior adjacency (undirected edges; symmetric) ─────────────────
# The learned adjacency initializes from this prior and adapts per subject.
# Groupings from ST_GAE_DESIGN §2.2.
FUNCTIONAL_EDGES: list[tuple[str, str]] = [
    # face cluster — densely interconnected
    ("head_pose", "gaze"), ("head_pose", "blink"), ("head_pose", "au_upper"),
    ("head_pose", "au_mid"), ("head_pose", "au_mouth"),
    ("gaze", "blink"), ("gaze", "au_upper"), ("gaze", "au_mid"),
    ("blink", "au_upper"), ("blink", "au_mid"),
    ("au_upper", "au_mid"), ("au_upper", "au_mouth"), ("au_mid", "au_mouth"),
    # hands ↔ body ↔ each other
    ("hand_left", "hand_right"), ("hand_left", "body"), ("hand_right", "body"),
    # hand-to-face gesture couples hands to the head
    ("hand_left", "head_pose"), ("hand_right", "head_pose"),
    # the audio-visual speech loop
    ("voice", "au_mouth"), ("voice", "body"),
    # congruence is a derived cross-modal signal → coupled to speech + motion
    ("congruence", "voice"), ("congruence", "au_mouth"),
    ("congruence", "head_pose"), ("congruence", "body"),
]


def prior_adjacency() -> list[list[float]]:
    """Symmetric N×N prior with self-loops (1.0 diagonal, 1.0 on prior edges,
    a small 0.05 floor elsewhere so every pair can still learn a weak link)."""
    A = [[0.05] * N_NODES for _ in range(N_NODES)]
    for i in range(N_NODES):
        A[i][i] = 1.0
    for a, b in FUNCTIONAL_EDGES:
        i, j = NODE_INDEX[a], NODE_INDEX[b]
        A[i][j] = A[j][i] = 1.0
    return A


def summary() -> str:
    lines = [f"{SPEC_VERSION}: {N_NODES} nodes, {len(ALL_FEATURES)} features, "
             f"{len(FUNCTIONAL_EDGES)} prior edges"]
    for n, f in NODES:
        lines.append(f"  {NODE_INDEX[n]:2d} {n:11s} dim={len(f):2d}  {', '.join(f[:4])}"
                     + (" …" if len(f) > 4 else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    assert len(ALL_FEATURES) == sum(NODE_DIMS)
    assert len(set(ALL_FEATURES)) == len(ALL_FEATURES), "duplicate feature column"
    print("\nprior adjacency row sums:",
          [round(sum(r), 2) for r in prior_adjacency()])
