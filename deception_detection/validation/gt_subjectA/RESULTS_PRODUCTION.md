# Production Stage-2 re-score — SubjectA_session1 (2026-07-09)

Second, independent validation of the deviation pipeline — this time through the **full production
path**: operator-clicked SPOVNOB Stage-1 diarization → real target-only audio segments →
recording-level Stage-2 with blink/EAR alive. Where the 2026-07-08 run used whole-clip audio and
had a dead blink channel, this run isolates the target's voice and populates blink 100%.
Labels remain scoring-only (never touch calibration).

## Run provenance
- SPOVNOB batch `rec_subjectA`: hash chain intact (1145 entries), all 8 WAVs SHA-256-verified,
  operator speaking-click at 32880 ms on the baseline (NO_ANTI_PROFILE). `pipeline_output.json`:
  26 clean target segments, global timeline 0→765140 ms.
- Stage-2 `REC_SUBJECTA`: 8 clips, real diarization segments applied (is_audio_active masks
  non-target audio). **First genuinely-calibrated recording** — baseline z: |mean| 0.000, std
  1.000, all 134 features non-NaN; baseline `deviation_magnitude` median 9.85 ≈ √134 (the correct
  healthy value — a baseline of 0.0 is degenerate, which is what rec_ca was). Interview clips
  deviate above baseline (medians 11–20). Deliverable: 952 windows × 148 cols.

## Headline: the signal replicates, and it's directional per node

Within `06_interview` (the Truth→Lie clip, confound-controlled; n_lie=189, n_truth=51),
per-feature |z| rank-AUC (Lie vs Truth) vs the 2026-07-08 whole-clip run in brackets:

| channel | production AUC | 2026-07-08 |
|---|---|---|
| AU12 velocity tremor band power | **0.696** | 0.681 |
| AU12 velocity max | 0.684 | 0.677 |
| left hand↔face distance (min) | 0.680 | 0.679 |
| right wrist velocity max | 0.667 | 0.660 |
| AU1 velocity max | 0.632 | — |
| AU4 var | 0.624 | 0.624 |
| acoustic_energy_rms | 0.622 | (audio contaminated before) |
| head-pitch tremor | 0.615 | 0.625 |
| gaze_x mean (inverse) | 0.162 | 0.162 |
| gaze entropy (inverse) | 0.310 | 0.313 |

The visual channels reproduce to ±0.01 — expected, since they were never audio-dependent; the
gaze inversion is **bit-identical** (0.162). This robustness across two independent audio pipelines
(whole-clip → target-isolated) is strong evidence the signal is real, not a pipeline artifact.

## Per-node attribution table (within-06) — the ST-GAE preview

Collapsing to the 11 ST-GAE nodes (peak-|deviation| channel per node):

| node | peak AUC | direction | driver |
|---|---|---|---|
| au_mouth | 0.696 | **lie ↑** | AU12 velocity tremor |
| hand_left | 0.680 | **lie ↑** | hand↔face distance |
| hand_right | 0.667 | **lie ↑** | wrist velocity |
| au_upper | 0.632 | lie ↑ | AU1 velocity |
| au_mid | 0.594 | lie ↑ | AU6 velocity |
| gaze | 0.162 | **lie ↓ (freeze)** | gaze-x |
| head_pose | 0.354 | lie ↓ (freeze) | head-yaw |
| blink | 0.365 | lie ↓ (freeze) | blink-count |
| voice | 0.368 | lie ↓ (freeze) | wavlm latent |
| body | 0.378 | lie ↓ (freeze) | postural stillness |

**Bipolar "freeze + facial-leakage" signature:** during lies this subject's gross motor systems
(gaze, head, blink, body, voice prosody) *stabilize toward baseline* while facial micro-dynamics
(AU12/AU1 velocity) and the hands *activate*. This is exactly why any scalar L2 aggregate is null
(opposing directions cancel) and why the ST-GAE must be direction-aware per node (§ST_GAE_DESIGN).

## New channels, first measurement
- **Blink node (was 100% NaN before the seam fix):** weak/inverse — `blink_count` 0.365,
  `ear_var` 0.434 within-06 (pooled 0.45–0.47). Blink joins the **freeze cluster** (blink
  suppression / reduced deviation during lies — a documented cue), not a strong standalone
  channel for this subject, but now real and available to the graph.
- **Isolated voice:** `wavlm_latent_4` 0.574 (was 0.608 on whole-clip audio) — the drop is
  consistent with removing interviewer contamination that had spuriously helped;
  `acoustic_energy_rms` is now a clean moderate channel (0.622 within-06).

## Known caveat carried forward
An ~80 ms A/V desync exists in the canonical files (video leads audio; open-GOP MPEG-2 start
offset dropped by ffmpeg — see MASTER_REFERENCE §12.1). Negligible here (2.4 frames vs 2 s
windows; visual AUCs unaffected, gaze inversion bit-identical) but must be corrected in the
canonicalizer before the frame-level ST-GAE congruence node is trained.

## Reproduce
Outputs: `pipeline_system_outputs/REC_SUBJECTA*/` (gitignored). Diarization:
`audio_diarization/session/rec_subjectA/pipeline_output.json`. Scripts (this dir):
`recA_cascade.py` (Pass-1, real segments), `recA_assemble.py` (Pass 2–4), `recA_score.py`
(ELAN re-score). Env: absolute `~/anaconda3/envs/spovnob_env/bin/python`; HF offline is now
hardcoded (acoustic_extractor.py) so WavLM never phones home.
