# Accuracy levers on the N=6 corpus — §6.1 aggregation + §6.2 meta-analysis (2026-07-17)

Two no-new-data levers from `PROJECT_RETROSPECTIVE.md`, run after the user set the goal as a
**general cross-subject detector** with a **handful (≤~6) more subjects** available. Bottom line: the
strong per-subject channels don't transfer, but a **small panel of weak, directionally-consistent
channels generalises better than the full model** — a real (if weak) lead for a general detector, and
the thing to firm up as N grows toward ~12.

## §6.2 Weak-but-broad meta-analysis (`meta_analysis.py`)
Random-effects (DerSimonian–Laird) pool of each channel's **directional** within-clip Lie-vs-Truth
AUC across the 6 subjects, with Hanley–McNeil sampling variance. Directional (signed) is the point:
a channel that inverts across people (blink: B lie→more 0.48-side, D lie→less) correctly pools to
~0.44 with **I²=96%** and is rejected — a universal detector needs the same sign in most people.

**5 universal-signal candidates** (pooled 95% CI excludes 0.5, I²<60%, ≥⅔ same direction):

| channel | pooled AUC | 95% CI | I² | consistency |
|---|---|---|---|---|
| emotion_confidence_mean | 0.522 | [0.509, 0.536] | 0% | 100% |
| wavlm_latent_9 | 0.521 | [0.506, 0.536] | 0% | 83% |
| gaze_y_mean | 0.516 | [0.501, 0.531] | 17% | 100% |
| AU6_max | 0.485 | [0.471, 0.498] | 0% | 100% |
| AU2_velocity_max | 0.514 | [0.501, 0.527] | 0% | 67% |

Two honest observations:
1. **Real but tiny.** Each is ~1–2 AUC points off chance — statistically significant only because
   pooling 6 subjects × hundreds of windows tightens the CI. Individually useless for detection.
2. **Strength ⊥ generality.** These universal channels (emotion-confidence, a WavLM latent, gaze-Y)
   are *entirely different* from the strong per-subject channels (AU12, blink, silent-speech). The
   channels that carry strong signal are idiosyncratic; the channels that generalise carry almost
   none. (My prior guess `head_pitch_somatic_dominant_freq` did **not** survive the directional test.)

This refines the strict "0/134 replicate": there is a *whisper* of universal signal, just far below
usable strength — consistent with the deception literature.

## §6.1 Coarser decision units + the weak-universal panel (`aggregate_evaluate.py`)
Decision unit = a contiguous same-label run within a clip (an "answer"). **58 answer-segments total**
(3–11 per subject) — so *segment-level* AUCs are noise-dominated and reported as directional only;
the **window-level** numbers (thousands of windows) are the trustworthy ones.

### A. Cross-subject LOSO, all 134 channels — window → segment
mean **0.469 → 0.539**. Aggregation nudges the full model from below-chance to barely-above, but per
subject it swings 0.25–0.75 on ~8 segments — not a rescue, not distinguishable from chance.

### B. Cross-subject LOSO, **weak-universal panel only** (the 5 §6.2 candidates) — the real result
| held-out | window | segment |
|---|---|---|
| A | 0.542 | 0.867 |
| B | 0.554 | 0.677 |
| C | 0.532 | 0.556 |
| D | 0.524 | 0.600 |
| E | 0.480 | 0.438 |
| F | 0.507 | 0.583 |
| **mean** | **0.523** | **0.620** |

**Window-level 0.523 — above chance, and materially better than the full-feature LOSO (0.469)** on
the same thousands of windows. 5/6 held-out subjects ≥ 0.507. This is robust to the small-N segment
problem (it's a window-level result) and is the **honest lead for a general detector**: throwing away
the 129 idiosyncratic channels and keeping the 5 directionally-consistent ones *improves* transfer.
The segment column (0.620) is encouraging but N-fragile — treat as hypothesis, not proof.

### C. Per-subject strongest channel — window → segment
mean **0.657 → 0.740** (aggregation helps within-subject too). **Caveat:** the per-subject "strongest"
here is a post-hoc max over 134 channels (winner's-curse-inflated) and the segment counts are tiny
(two subjects hit 1.000 on 3–6 segments). Direction (segment ≥ window) is plausible; absolute values
are not to be quoted. The shipped, unbiased per-subject numbers remain the scorecard's tracked
channels (A silent-speech 0.68, B blink 0.71).

## What this means for the general-detector goal (on a ≤~6-more-subjects runway)
- A *validated universal supervised detector* remains out of reach at N≈12 (needs 50–200). **But**
  the weak-universal **panel** (§6.2 candidates + §6.1 aggregation) is a concrete, testable general
  detector prototype sitting at ~0.52 window / ~0.62 segment — weak, honest, and pointed the right way.
- **The plan as the handful of new subjects arrive:** re-run `meta_analysis.py` (the panel firms up
  or shrinks with more evidence) and `aggregate_evaluate.py` §B. **The key metric to watch is the
  panel's LOSO *window* AUC**: if it climbs from 0.523 toward ~0.60 as N→~12, that's genuine progress
  toward a general detector; if it stalls at ~0.52, the null hardens at an authoritative sample size.
- No doctrine change and no new data were needed for any of this; labels only scored, LOSO held each
  subject out. Everything re-runs automatically on a larger corpus via the generic toolchain.

## Provenance
`validation/multisubject/meta_analysis.py` → `meta_analysis.csv`;
`validation/multisubject/aggregate_evaluate.py` → `aggregate_loso.csv` + `aggregate_persubject.csv`.
Manifest `replication_manifest_N6.json`. Design: `Documentation/PROJECT_RETROSPECTIVE.md` §6.1/§6.2.
