# ST-GAE End-Stage — Design Document

*Status: DESIGN (2026-07-09). Nothing here is implemented. Companion to MASTER_REFERENCE §14.2;
supersedes the open design questions listed there when this doc and §14 disagree.*

## 0. Why this design is now empirically constrained

The 2026-07-08 ground-truth validation (`deception_detection/validation/gt_subjectA/RESULTS.md`)
measured three facts that dictate the architecture:

1. **No scalar aggregate carries signal** (L2 `deviation_magnitude` AUC 0.384 — *inverted*;
   median |z| 0.505; trimmed L2 0.435). Signal exists only per-channel.
2. **Channel directions oppose each other and are subject-specific**: for SubjectA, lies show
   AU12 velocity dynamics ↑ (0.69), hand↔face ↑ (0.69), wrist velocity ↑ (0.65), head-pitch
   tremor ↑ (0.63) — while gaze variability ↓ (0.14, i.e. *freezes*) and gross body motion ↓.
   Any end-stage must preserve per-channel identity and direction; summing destroys the signal.
3. **The per-subject fit budget is small**: one baseline clip ≈ 105 s ≈ 3,200 frames.
   The model must be fittable, per subject, on minutes of data — this kills any large architecture.

The mandate: a model whose *output* is a per-channel, per-time deviation map (attribution),
not any single score — and whose *advantage* over the existing z-score path is capturing
**relational and temporal structure** the marginal z-scores cannot see.

## 1. What the ST-GAE adds over the z-score path (and why keep both)

The z-score path measures **first-order marginal deviations**: "AU12 variance in this window is
2.3σ above this subject's baseline." It is blind to:

- **Cross-channel structure**: baseline establishes how channels co-move (e.g. when this subject
  speaks, AU25/26 and voice energy and gaze saccades co-activate in a characteristic pattern).
  A lie segment where each marginal stays in-range but the *coupling* breaks (voice active, face
  uncharacteristically still — the freeze pattern) is invisible to marginals and is exactly what
  `mismatch_incongruence` crudely approximates today with a hand-written rule.
- **Temporal dynamics**: onset/decay shapes, lead–lag between channels (gesture precedes speech
  in baseline; lags during high cognitive load per the literature).

The ST-GAE learns this joint structure from the baseline clip and reports **where reconstruction
fails, per node, per time** — deviations *from the subject's own behavioral grammar*, not just
from their marginal feature ranges.

**Decision: complement, not replace.** The Temporal Anomaly Attribution Report shows both layers:
marginal z (existing path, §8) and relational reconstruction error (ST-GAE), per channel, on one
timeline. They answer different questions and cross-validate each other; agreement between the
two is itself forensic evidence of robustness.

## 2. Graph definition

### 2.1 Nodes = semantic channel groups (11 nodes)

Not one-node-per-feature (60+ nodes, ignores structure; attribution too granular to read), and
not skeleton landmarks (classic ST-GCN — unavailable: the 30 fps CSV carries derived features,
not raw landmark grids). Nodes are the units an analyst reasons about — the same units the
validation found signal in:

| # | node | frame features (from the 70-col CSV) | dim |
|---|---|---|---|
| 0 | `head_pose` | head_yaw, head_pitch, head_roll | 3 |
| 1 | `gaze` | gaze_x, gaze_y, gaze_z, gaze_velocity | 4 |
| 2 | `blink` | ear, is_blinking *(alive since the 2026-07-08 seam fix)* | 2 |
| 3 | `au_upper` | AU1, AU2, AU4 + velocities | 6 |
| 4 | `au_mid` | AU6, AU9 + velocities | 4 |
| 5 | `au_mouth` | AU12, AU25, AU26 + velocities | 6 |
| 6 | `hand_left` | left_wrist_{x,y,z}, left_wrist_velocity, left_hand_face_distance | 5 |
| 7 | `hand_right` | (mirror of 6) | 5 |
| 8 | `body` | macro_motion_energy, postural_stillness, nose_{x,y,z} (trunk drift) | 5 |
| 9 | `voice` | frame_wavlm_latent_0..15, frame_prosodic_velocity, frame_acoustic_energy_rms | 18 |
| 10 | `congruence` | is_audio_active, mismatch_incongruence, silent_incongruence | 3 |

Excluded from reconstruction: confidences (`joint_confidence` etc. → **loss weights**, not
features), `emotion_label`/`emotion_confidence` (categorical, MLT-derived, redundant with AUs),
bookkeeping (frame_id, timestamp).

Each node carries its own small feature vector; nodes are projected to a **shared node-embedding
dim** (D=16) by per-node linear encoders, so the graph operates on an 11×16 tensor per frame.

### 2.2 Edges

- **Spatial (within-frame): learned adjacency over 11 nodes** (a single 11×11 matrix, softmax-
  normalized), initialized with functional priors and free to adapt per subject during fit:
  face nodes densely connected; hands↔body; voice↔au_mouth and voice↔congruence (the
  audio-visual speech loop); congruence↔everything (it is a derived cross-modal signal).
  11×11 = 121 parameters — learning it is cheaper than arguing about it, and the *fitted
  adjacency itself is forensic output* (how strongly this subject's channels couple at baseline).
- **Temporal: 1-D temporal convolutions per node** (kernel 9 ≈ 300 ms at 30 fps — micro-
  expression onset scale), standard ST-GCN factorization: alternate spatial graph-conv and
  temporal conv. No temporal attention — parameter budget (§4) forbids it.

## 3. Architecture & losses

**Model: masked spatio-temporal graph autoencoder**, deliberately tiny (<100k params).

```
input  [T=90, 11 nodes, F_node]           (3 s clips, stride 15 frames)
  → per-node linear → [90, 11, 16]
  → 3 × { spatial GCN (learned A) → temporal conv (k=9) → GeLU, residual }
  → bottleneck: temporal stride-2 ×2 → [~23, 11, 8]      (the latent)
  → mirrored decoder (upsample + st-blocks)
  → per-node linear heads → reconstruction [90, 11, F_node]
```

**Loss** = confidence-weighted, masked, **feature-dimension-normalized** MSE:

- **MANDATORY — feature-count normalization (per-node MSE ÷ F_node, then sum).** The 11 nodes
  have wildly unequal dimensionalities (`voice`=18, `blink`=2). A plain summed MSE lets the
  18-dim voice node dominate the gradient and the 2-dim blink node get ignored — the model
  would optimize audio reconstruction and never learn blinks. So the global loss is

  ```
  L = Σ_n  w_n · ( 1 / F_n ) · Σ_{f∈n} mask · (x̂ - x)²        ← divide EACH node's MSE by its
                                                                  own feature count F_n before
                                                                  summing across the 11 nodes
  ```

  Every node contributes on equal footing regardless of width. (`w_n` is an optional per-node
  importance weight, default 1; leave uniform unless a later ablation motivates otherwise.)
- **Features pre-normalized with the existing `BaselineStats`** (fit on the same baseline clip)
  so every channel is in z-units — reconstruction error is then scale-free and directly
  comparable across nodes. Reuses, not duplicates, the §8 calibration machinery. (This handles
  *within-node* per-feature scale; the ÷F_n above handles *across-node* dimensionality — both
  are required, they fix different imbalances.)
- **Masks** (loss = 0, no gradient): NaN tracking gaps (any node); `voice` node wherever
  `is_audio_active != 1` (the mask column the frame CSV already carries for exactly this
  purpose); `blink`/AU nodes wherever face_confidence < threshold. A fully-masked node
  contributes 0 and is excluded from that frame's node count (no divide-by-zero).
- **Weights**: per-frame `joint_confidence` scales each frame's loss contribution.
- 10 % feature dropout on inputs (masked-denoising) — regularization + forces cross-channel
  inference, which is the point of the graph.

## 4. Per-subject fit protocol (the hard constraint)

Budget: ~3,200 baseline frames → ~205 clips of 90 frames at stride 15. Tiny. Anti-overfit stack:

1. **<100k parameters** (D=16, 3 blocks — the sizing above lands ~60–80k).
2. **Temporal split, not random**: first 80 % of baseline = train, last 20 % = validation
   (random splits leak through overlapping clips). Early-stop on val reconstruction.
3. **Augmentation**: temporal jitter (±2 frames per node group), Gaussian noise at 0.1σ,
   random 10 % feature dropout (doubles as the denoising objective).
4. **Determinism**: fixed seed, deterministic CUDA — same doctrine as SPOVNOB; a re-fit on the
   same baseline must reproduce the same attribution (forensic requirement).
5. **Failure mode**: if val error does not plateau above the noise floor (baseline too short /
   too corrupted), the fit **fails loudly** — mirroring `BaselineCalibrationError` doctrine.
   Minimum-baseline-duration question (§14) becomes an *empirical output* of this failure test,
   not a design guess.

Fit cost estimate: 205 clips × ~300 epochs × tiny model — minutes on the RTX 6000. Per-subject
fit is not a bottleneck.

## 5. Attribution output (the deliverable)

For every interview window (reusing the 2 s / 1 s grid so ST-GAE and z-path rows align):

1. **Node error** `E[n, t]` = masked mean squared residual per node per window, **normalized to
   the baseline's residual distribution**: fit residual mean/std per node on the *validation*
   slice of baseline → report `(E − μ_n) / σ_n` (a "reconstruction-z"). Same fit/apply shape as
   the calibrator — deliberately.
2. **Directional detail**: top-k per-feature signed residuals within each flagged node (the model
   *expected* AU12 velocity X, *observed* Y) — this is where direction-awareness lives, answering
   fact 2 of §0.
3. **Coupling anomalies**: per spatial-edge contribution shifts on flagged windows (which
   couplings broke) — from the same forward pass, no extra machinery.

**Temporal Anomaly Attribution Report** = the `clip06_timeline.html` layout (validated as the
right visual form), with per-node rows showing *both* marginal z and reconstruction-z, plus
flagged-segment summaries. Output artifact per recording:
`<rid>_stgae_attribution.csv` (windows × [node errors, top residuals]) + the HTML report.

## 6. Validation protocol (unchanged doctrine)

Fit on `00_baseline` **only**; attribute `01..07`; ELAN labels overlay *post hoc* for scoring
only (never touch fit). Metrics, all within-clip-06 + non-overlapping-window robustness, mirror
2026-07-08: per-node reconstruction-z AUC (Lie vs Truth), pooled + per clip.

**Success bars** (pre-registered so we can't move the goalposts):
- **Bar 1 — Sensitivity.** Beat or match the best single-channel marginal AUC (0.69) with *at
  least node-level* attribution intact.
- **Bar 2 — Interpretability.** Rediscover the known channel structure blind (au_mouth/hands ↑,
  gaze ↓) — a sanity check that reconstruction error tracks real behavior, not artifacts.
- **Bar 3 — Relational value.** The congruence/coupling layer should flag the freeze pattern
  (voice active + face still) that motivated it; if it can't beat the hand-written
  `mismatch_incongruence` rule, the relational claim of §1 is falsified and we keep the z-path only.
- **Bar 4 — Holdout Truth Stability (false-positive check; protects against Small-Baseline Panic).**
  The single biggest risk here is *not* missing a lie — it is that a fit on only ~2 min of
  baseline fails to generalize the subject's normal behavior, so reconstruction error spikes on
  *everything* past the baseline, including plain truthful speech. **Rule:** applied to the
  ELAN-verified **Truth** segments of the interview clips (01–06), the global reconstruction-z
  must stay *stable* — it must not raise widespread anomaly flags (operationalized: median
  reconstruction-z on held-out Truth windows within ~1 of its value on held-out *baseline*
  windows, and the Truth-window flag rate must not exceed the baseline-window flag rate by more
  than a small pre-set margin). **Falsification:** if the model clears Bar 1 on lies *but* also
  fires massive, erratic errors on truth, the 2-minute fit was too brittle to survive real-world
  temporal drift — the graph is discarded and we revert to the marginal z-score path. This bar
  is the go/no-go gate; a model can be sensitive (Bar 1) and still be useless if it cries wolf on
  every truthful moment.

  *Bars 1–3 test whether the ST-GAE sees more than marginals; Bar 4 tests whether it can be
  trusted not to hallucinate on this data budget. All four are required to ship it.*

## 7. VideoMAE v2 verdict

**Deferred with a concrete re-entry criterion** (previously: vague "presumed 4th branch").
The graph consumes engineered channels; VideoMAE latents would add a 12th node
(`appearance`, D≈16 projected) carrying facial texture/micro-movement information the 8 AUs may
miss. Re-open **only if** the fitted ST-GAE shows a measurable blind spot: ELAN-labeled lie
segments where *no* node deviates (misses concentrated where AU coverage is known-thin —
e.g. AU5/7/23 territory). Until that evidence exists, VideoMAE is dead weight: +1 GPU-heavy
extractor, no measured need.

## 8. Implementation plan (when green-lit — NOT part of this design task)

- `deception_detection/stgae/`: `graph_spec.py` (node/edge tables above, single source of
  truth), `dataset.py` (frame-CSV → masked clip tensors; reuses `BaselineStats`),
  `model.py` (~150 lines, plain torch — an 11-node dense adjacency needs einsum, **not**
  torch-geometric; zero new dependencies in `spovnob_env`), `fit.py` / `attribute.py` (CLI
  mirroring calibrator fit/apply), `report.py` (attribution CSV + HTML).
- Tests in the house style: synthetic-data determinism, mask-correctness (a NaN gap or
  inactive-voice frame must contribute exactly zero gradient), overfit-on-noise failure test.
- Order of work: graph_spec + dataset + masks first (they encode every §2–§3 decision);
  model is the trivial part.

## 9. Open questions this design *closes* vs *leaves open*

Closed: **node granularity — LOCKED at 11 nodes (2026-07-09 review): neither finer nor coarser,
with mandatory feature-count-normalized loss (§3) as the condition of that lock**; edge
definition (§2); replace-vs-complement (§1 — complement); VideoMAE (§7 — deferred w/ criterion);
per-subject fit feasibility (§4 — yes, with the anti-overfit stack); attribution form (§5);
false-positive gate (§6 Bar 4 — Holdout Truth Stability, the go/no-go).

Open (empirical, answered by the first fit): minimum baseline duration (§4.5 failure test);
whether D=16 / 3 blocks is the right size (val-error plateau will say); whether coupling
anomalies (§5.3) add analyst value beyond node errors.
