Got it — pure analysis, no code. Let me think deeply through **Part 1** right now.

---

# Part 1 — Speaker Diarization & Voice Isolation

*The full code for this stage lives in two files: `diarizer_engine.py` and `acoustic_extractor.py`. I'll go through every function, every assumption, and every hidden dependency.*

---

## What this stage is actually trying to do

The job of this stage, at a high level, is:

> "Given a mixed audio recording of an interview with multiple speakers, figure out which voice belongs to the target (interviewee) and produce an isolated audio file of only their voice."

This sounds straightforward but it's one of the hardest problems in the entire pipeline because it requires **fusing two fundamentally different streams of information** — acoustic speaker clustering from Pyannote, and visual lip tracking from MediaPipe — and getting them to agree on the same person.

The system's bet is: *"Whoever is speaking will also have moving lips at the same time. So the speaker whose audio timestamps most overlap with lip-moving video frames must be the target."*

That's a reasonable heuristic. But heuristics fail in edge cases, and in interview recordings, edge cases are extremely common. Let's go through each one.

---

## Loophole 1 — The Composite Score Formula Is Fundamentally Ambiguous

Look at this:
```python
composite_score = binary_overlap_ratio * mean_visual_velocity
```

There are **two completely different quantities** being multiplied here:
- `binary_overlap_ratio` — how often the lips are moving during that speaker's audio segments (a ratio, 0–1)
- `mean_visual_velocity` — the average *speed* of lip movement during those segments

**Why this is a problem:**

Multiplying these two together creates a score that you can't meaningfully interpret. Here's why:

A speaker who talks for a short burst with *very fast* lip movement could score the same as a speaker who talks for a long time with *moderate* lip movement. These two scenarios are physically very different, but the formula can't tell them apart.

More critically: the score depends on the **unit of `mor_velocity`**. What units are `mor_velocity` measured in? Pixels per frame? Normalized landmark distance per frame? The magnitude of this number completely dominates the composite score, making `binary_overlap_ratio` almost irrelevant if `mor_velocity` values are large (e.g. in pixel units, they could be in the dozens).

So essentially, the current formula is weighting `mean_visual_velocity` much more heavily than `binary_overlap_ratio` without explicitly choosing to do so. The two components are never normalized relative to each other.

**The hidden assumption that can break this:**

The formula only works if the target is the *most visually active* speaker. What if the interview setup has two people on camera and the target is a reserved, soft-spoken person — maybe a nervous interviewee who speaks slowly with minimal lip movement? They will systematically score lower than an animated interviewer, and the wrong person gets locked as "target" — with full confidence.

---

## Loophole 2 — There Is No Sanity Check on the Winning Score

Look at the selection:
```python
if composite_score > highest_composite_score:
    highest_composite_score = composite_score
    target_speaker_id = speaker_id
```

And then in `execute_isolation_pipeline`:
```python
target_id, confidence = self.anchor_target_identity(...)
if target_id is None:
    print("⚠️ WARNING: ...")
    target_id = "TARGET_SILENT"
    confidence = 0.0
    
self.isolate_voice_channel(...)
```

There is only **one failure condition** that stops execution: `target_id is None`, which only happens if *all* speaker segments have zero frames in the visual logs. In every other case — including a `composite_score` of `0.00001` — the system proceeds to isolation without any hesitation.

**The real danger:** Imagine a session where the target barely speaks during the first few minutes (they're listening), and the interviewer has consistently high lip motion + audio overlap. The system picks the interviewer with a score of, say, `0.38`. Then it isolates the interviewer's voice. Every single acoustic feature — `acoustic_volatility`, `prosodic_velocity`, `vocal_entropy`, all 16 HuBERT latent channels — now describe the *interviewer*.

The manifest writes:
```
"assigned_target_speaker_id": "SPEAKER_01"
"cross_modal_correlation_score": 0.38
```

And nothing downstream ever questions this. The `cross_modal_correlation_score` sits in the manifest but is never used as a gate anywhere. It's purely informational — which means it could be screaming "wrong speaker" and nobody would hear it.

---

## Loophole 3 — The Visual Logs Have No Face Identity

This is probably the most fundamental architectural gap in this entire stage.

The `visual_speech_logs` look like this:
```python
[
  {'timestamp_ms': 1033.3, 'is_moving': 1, 'mor_velocity': 0.024},
  {'timestamp_ms': 1066.6, 'is_moving': 0, 'mor_velocity': 0.003},
  ...
]
```

Notice what's missing: **which face these observations belong to.**

The lip tracking pipeline locks onto a face and tracks it. But in a real two-person interview with both people visible on camera, the tracker could be following either face at any given moment — especially if the target looks away, turns, or is partially occluded.

So when `anchor_target_identity` computes "how much does this speaker's audio overlap with moving lips," it's looking at lip motion from *whichever face the tracker happened to be on*, not necessarily the face that belongs to the speaker being evaluated.

**The concrete failure scenario:**

- Pyannote correctly says `SPEAKER_00` is the target (they speak more and louder)
- The face tracker, for the first 2 minutes, is actually locked onto the interviewer's face (larger, more centered, better lit)
- The `visual_speech_logs` for those 2 minutes reflect the interviewer's lip activity
- During `SPEAKER_00`'s audio segments, the tracker shows the *interviewer* nodding and talking back-channel — so `SPEAKER_00`'s audio overlaps with *interviewer lip activity*
- The system concludes: `SPEAKER_00`'s audio correlates with high lip motion → target confirmed
- But it was actually tracking the wrong face. The correlation was a coincidence or a false match.

Or the reverse failure: tracker is on the target's face, but the target is intently listening and barely moving their lips while the interviewer talks. Lip motion is low → score is low → wrong speaker is selected.

This is a fundamental flaw because **you can't do cross-modal anchoring correctly without knowing which face produced the lip motion data.**

---

## Loophole 4 — The "Silence Tripwire" in HuBERT Has a Hidden Gap

In `acoustic_extractor.py`:
```python
rms_energy = float(np.sqrt(np.mean(raw_chunk ** 2)))
if rms_energy < SILENCE_RMS_FLOOR:
    return null_result
```

`SILENCE_RMS_FLOOR = 0.005`

The intent is: if this audio chunk was attenuated (non-target speaker), it will be near-silent and we'll skip it rather than running HuBERT on noise.

**The gap:** The attenuation factor is `0.05` (5% amplitude). This means a non-target speaker's chunk has 5% of its original amplitude. If the original RMS of that chunk was, say, `0.12` (a normal speaking voice), after attenuation it becomes `0.006` — which is **above the silence floor of 0.005**.

So HuBERT will *still process the interviewer's attenuated voice* on windows where their original RMS is even slightly above `0.1`. The "silence tripwire" doesn't actually silence them — it just attenuates them. And attenuated speech is still speech. HuBERT is trained on speech and will happily extract meaningful paralinguistic features from it.

**What actually happens:** windows that overlap with the interviewer's speech (but not the target's) get partially-attenuated audio fed into HuBERT, which returns features reflecting a *mixture* of both voices, not the isolated target. The system has no way to know this happened.

The fix isn't just lowering the RMS floor — the attenuation and the silence tripwire need to be designed together as a matched pair. If you attenuate to 5%, your silence floor needs to be above the maximum RMS you'd expect from attenuated speech.

---

## Loophole 5 — The KMeans Codebook Is Built on Whatever Comes First

```python
max_samples = min(len(self.audio_signal), HUBERT_SAMPLE_RATE * 10)
sample_chunk = self.audio_signal[:max_samples]
```

This takes the **first 10 seconds** of the *already-isolated* audio to fit the codebook.

But think about what the isolated audio actually looks like during the first 10 seconds of an interview:
- The baseline period is typically an introduction / briefing
- The target might be nodding silently or not yet speaking
- The first 10 seconds may be largely silence or attenuated interviewer speech that leaked through

If the first 10 seconds contains mostly silence (RMS below certain levels but above the tripwire), HuBERT produces hidden states representing *the acoustic texture of near-silence or noise*, not the target's voice.

The codebook is then fitted on these noise-dominated hidden states. Every subsequent `vocal_entropy` computation in the test period — where the target is actively speaking — is measured against a codebook vocabulary that was built from silence. The cluster assignments are meaningless because the vocabulary was never trained on real speech.

This is a soft corruption: nothing crashes, the numbers come out, but `vocal_entropy` for the whole session is now measuring "how different is this window from the silence of the beginning" rather than "how varied is the target's vocal production across the session." Those are completely different measurements.

---

## Loophole 6 — Stereo to Mono Uses Channel 0 Unconditionally

Two places in the code do this:

In `isolate_voice_channel`:
```python
if audio_signal.ndim > 1:
    audio_signal = audio_signal[:, 0]
```

In `HuBERTAcousticExtractor.__init__`:
```python
if audio_signal.ndim > 1:
    audio_signal = audio_signal[:, 0]
```

Both always take **channel index 0** — the left channel.

In professional interview recording setups, it's common practice to use **dual-mono** or **split-channel recording**:
- Left channel = target's lapel mic
- Right channel = interviewer's lapel mic (or ambient)

Or the reverse. There's no standard.

If you always take channel 0, you will correctly capture the target *exactly half the time*, and completely capture the wrong person the other half. And again — nothing will fail, nothing will warn you. The diarization step will then try to identify the target from *only the interviewer's audio*, which makes the entire cross-modal anchoring exercise not just wrong but actively misleading.

---

## Loophole 7 — The `TARGET_SILENT` Fallback Attenuates Everything

```python
if target_id is None:
    print("⚠️ WARNING: Target exhibited zero acoustic lip movement. Attenuating entire audio track.")
    target_id = "TARGET_SILENT"
    confidence = 0.0
    
self.isolate_voice_channel(input_wav_path, output_wav_path, pyannote_segments, target_id)
```

Then in `isolate_voice_channel`:
```python
if target_speaker_id is not None:
    for speaker_id, start_ms, end_ms in pyannote_segments:
        if speaker_id == target_speaker_id:  # "TARGET_SILENT" matches nothing
            ...
```

`"TARGET_SILENT"` won't match any `speaker_id` from pyannote (which would be something like `"SPEAKER_00"`), so **the target audio mask remains all-False**, and the entire signal gets attenuated by `0.05`.

This means when the target is genuinely silent or has zero lip motion detected, we don't just produce silence for them — we feed HuBERT a fully-attenuated mix of *all* speakers. Then the silence tripwire may or may not fire depending on RMS levels.

But more importantly: a session where the target is genuinely quiet during the first few minutes (extremely common in high-stress interviews — the person is tense, barely moving) would trigger this path, producing an attenuated mess and marking all acoustic windows as NaN. You've lost the entire acoustic channel for a session precisely because the subject was showing behavioral signs of high stress. The most critical sessions are the ones most likely to be corrupted.

---

## The Big Picture Summary for Part 1

When you step back and look at all of these together, a theme emerges: **this stage assumes the "easy" case of interview recording and breaks on every realistic edge case.**

| Assumption Made | When It Breaks |
|---|---|
| Target has more/faster lip movement than interviewer | Soft-spoken or frozen target |
| Visual logs track the correct face | Multi-face frame, occlusion, interviewer is more prominent |
| Channel 0 = target's voice | Dual-channel recording with non-standard channel assignment |
| First 10 seconds of audio contains target's speech | Baseline period, late-start speaker, quiet opening |
| 5% attenuation silences non-target speakers | Any speaker with a reasonably loud voice |
| A wrong selection will produce a clearly bad score | Score of 0.001 still triggers full isolation with no warning |

The most dangerous property of this entire stage is **silent failure**. Every one of these loopholes produces plausible-looking numbers. The manifest gets written, the isolated WAV file exists, HuBERT runs and returns 20 features — but they describe the wrong person. And from that point forward, every cross-modal feature, every HMM state, every session report is built on a foundation that was poisoned at step one.

---

Ready to move to **Part 2 — Cross-Modal Anchoring & Incongruence Features** whenever you are. That one has its own very different class of problems rooted in how the two clocks (audio and video timestamps) are assumed to be synchronized.


Edited issues%20and%20changes
Viewed issues%20and%20changes:236-240

Let me re-read the relevant files for this part carefully before I say a single word.

Viewed temporal_window_generator.py:1-282
Viewed context_mapper.py:1-70

Good. Now I have everything I need. Let me think through this deeply and carefully.

---

# Part 2 — Cross-Modal Anchoring, Incongruence Features & Sliding Window Descriptors

*The relevant files here are `dynamic_window_engine.py`, `temporal_window_generator.py`, `confidence_math.py`, and `context_mapper.py`. But the analysis has to go deeper than the code — it has to question the neuroscience and signal processing assumptions baked into the architecture.*

---

## First — Understand What This Stage Is Actually Claiming to Do

This stage is making a very bold scientific claim:

> *"By slicing a 30fps visual feature stream into 2-second overlapping windows with 1-second strides, and aligning them with HuBERT's ~50Hz audio latent states sampled over the same intervals, we can capture the behavioral signature of cognitive stress, deception, and emotional friction in a time-localized, cross-modal feature vector."*

Every single word in that claim is an assumption. Let's destroy each one.

---

## Loophole 1 — The 2-Second Window Is a Design Choice Presented as a Scientific Truth

The window size is `2000ms`. The stride is `1000ms`. These are hard-coded.

```python
class DynamicWindowEngine:
    def __init__(self, window_size_ms=2000.0, stride_ms=1000.0, ...):
```

```python
class TemporalWindowEngine:
    def __init__(self, window_size_sec=2.0, stride_sec=1.0, ...):
```

**The hidden neuroscientific assumption:**

The 2-second window implies that the cognitive and behavioral signals we're trying to detect — micro-expressions, postural shifts, vocal hesitation, gaze aversion — manifest and resolve themselves within a 2-second epoch. That's the bet.

This is only partially supported by the literature. Here's where it breaks apart in practice:

**Case 1 — Micro-expressions.** By definition, micro-expressions occur in 40–500ms. A 2-second window *averages them out*. The mean of an AU value over 2 seconds, where the micro-expression lasted 80ms and the remaining 1920ms were neutral, will show near-zero AU activation. The expression is diluted into the noise floor. You're not capturing micro-expressions with a 2-second mean — you're suppressing them.

**Case 2 — Cognitive motor lag.** Research suggests that the behavioral response to a cognitively taxing question (speech disfluency, postural shift, gaze aversion) has a **latency of 300ms–1.5 seconds** after the question is posed. If the question ends at `t=5000ms` and the target's cognitive response peaks at `t=6200ms`, a window from `[5000, 7000]ms` captures it. But a window from `[4000, 6000]ms` captures the end of the question and the onset of the response, blended together. The signal is split across two consecutive windows and averaged in both — and in neither window does it appear as a clean peak.

**Case 3 — Long behavioral events.** Genuine behavioral suppression (a sustained postural freeze, prolonged gaze aversion while constructing a lie) can last 4–8 seconds. A 2-second window captures only a slice of it. The window before it, the window during it, and the window after it all look different statistically. The HMM sees three different feature patterns when what it should see is one sustained event.

**The deeper problem:** There is no sensitivity analysis in this system for window size. Nobody has asked the question: *"If we change the window to 1.5 seconds, or 3 seconds, do we get better signal-to-noise on the behavioral events we care about?"* The 2-second window was probably chosen by convention, not by scientific optimization for this specific application.

---

## Loophole 2 — The 30fps Video and ~50Hz HuBERT Are Running on Different Physical Clocks

This is the clock synchronization problem and it's deep.

**How HuBERT's temporal resolution works:**

HuBERT processes audio at 16,000 samples/second. Its convolutional feature extractor produces one hidden state frame every ~20ms, meaning approximately **50 frames per second** in Layer 7 latent space. For a 2-second window, you get roughly `T ≈ 99` hidden state frames.

**How the visual pipeline works:**

Video is captured at 30fps. One frame every ~33.3ms. For a 2-second window, you get **60 frames** of visual data.

**The alignment problem:**

The two streams are already at different temporal resolutions. But here's the critical issue: **they are never explicitly aligned.** Look at how the window engine calls both:

```python
# Visual: slice by timestamp from raw CSV (30fps)
window_df = df[(df['timestamp'] >= start_time_ms) & (df['timestamp'] < end_time_ms)]

# Acoustic: call HuBERT on the same time range
acoustic_features = acoustic_extractor.extract_window_features(start_time_ms, end_time_ms)
```

The visual features are aggregated statistics over 60 video frames. The acoustic features are computed from 99 HuBERT latent states. They are then placed into the same row of the CSV as if they're describing the same instant in time.

They're not. They're describing the same *interval*, but through different temporal lens resolutions — 30fps vs 50fps. When you compute `confidence_weighted_mean` over 60 video frames and `acoustic_volatility` over 99 HuBERT frames, these two numbers are produced at fundamentally different temporal granularities and then treated as equivalent feature dimensions in a linear projection.

**Why this matters for the cross-modal Tensor Fusion:**

The TFN Cartesian outer product `V ⊗ A` is supposed to capture *interaction* between visual and acoustic features. But if visual features are temporally smeared at 30fps resolution and acoustic features are computed at 50fps resolution, the "interaction" you're measuring is not a true cross-modal synchrony signal — it's the product of two differently-blurred versions of the same 2-second interval. The fine temporal structure that would reveal genuine audio-visual coupling (e.g., prosodic stress coinciding with AU12 activation) is averaged away before the outer product is even computed.

---

## Loophole 3 — The `np.arange` Window Iterator Has a Floating-Point Accumulation Bug

Look at this line in both engines:

```python
for start_time in np.arange(0, max_time_ms, self.stride_ms):
```
```python
for start_time_ms in np.arange(0.0, max_time_ms - self.window_size_ms + 1.0, self.stride_ms):
```

`np.arange` with floating-point step values accumulates rounding error. This is a well-documented numpy behavior.

**The concrete problem:**

Imagine a 60-minute session. `stride_ms = 1000.0`. The windows start at:
`0.0, 1000.0, 2000.0, ... 59999.9999999998, ...`

After several thousand iterations, the floating-point accumulated error in `start_time` is no longer zero. It might be `59999.999999997` instead of `60000.0`. The window mask:

```python
mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
```

The video timestamps in `df['timestamp']` are computed separately (from frame index × frame duration). They won't have the same accumulated drift. So the window boundary at time `t` in the video frame doesn't match the window boundary at time `t` in the audio slice that HuBERT receives.

The audio and video windows that are *supposed to be aligned* are now off by anywhere from `0ms` to potentially `2–3ms` depending on session length. That sounds small. But at 30fps, 3ms is nearly 1/10th of a frame. And this drift is **monotonically increasing** over the session — it gets worse the longer the interview runs.

For a 60-minute interview, the last few hundred windows could have video and audio windows misaligned by a consistent ~3ms. That's not catastrophic by itself, but it means the cross-modal features at the end of the session are systematically computed on slightly different time slices than at the beginning. The HMM then sees a slight distributional shift in features toward the end of every session — a drift artifact, not a behavioral one.

---

## Loophole 4 — `confidence_weighted_max` Is Measuring Something Misleading

```python
def confidence_weighted_max(values, weights, threshold=0.35):
    reg_values = [regularize_value(v, w, threshold) for v, w in zip(v_masked, w_masked)]
    return float(np.nanmax(reg_values))
```

The `_velocity_max` feature for AU channels uses this:
```python
record[f"{au}_velocity_max"] = confidence_weighted_max(window_df[vel_col].abs(), weights, thresh)
```

The `regularize_value` function suppresses sub-threshold frame values toward zero:
```
Z_reg = w * Z   (when w < threshold)
Z_reg = Z       (when w >= threshold)
```

So a frame with a genuine high AU velocity spike (say, `AU12_velocity = 0.8`) that happens to have low tracking confidence (`joint_confidence = 0.2`) gets regularized to `0.2 × 0.8 = 0.16`. The `_velocity_max` for that window is then `0.16`, not `0.8`.

**The problem:** Micro-expression onset events — which are exactly what `_velocity_max` is supposed to detect — are *most likely to occur at low-confidence tracking moments*. Here's why: a genuine micro-expression is a rapid, involuntary muscle contraction. The face geometry changes quickly and asymmetrically. The face tracker's confidence drops when the face shape deviates significantly from the trained distribution. The very frames that contain the most behaviorally meaningful signal are the ones most aggressively suppressed by the confidence weighting.

The system is **inverting the signal-to-noise relationship** for velocity features specifically. For mean features, suppressing noisy low-confidence frames makes sense — you're averaging and low-confidence noise can pollute the mean. But for **maximum** features, you're asking: "what was the peak activation during this window?" And suppressing that peak based on tracker confidence is discarding precisely the event you're looking for.

A micro-expression that the tracker found hard to track is **more likely** to be a real micro-expression, not less. Tracker difficulty correlates with genuine facial geometric change.

---

## Loophole 5 — AU Velocity Is Computed on the Absolute Value — Which Destroys Directionality

```python
record[f"{au}_velocity_max"] = confidence_weighted_max(window_df[vel_col].abs(), weights, thresh)
record[f"{au}_velocity_mean"] = confidence_weighted_mean(window_df[vel_col].abs(), weights, thresh)
```

`.abs()` is called on the velocity before aggregation.

AU velocity is a first-order derivative — it's signed. Positive = AU activating (muscle contracting). Negative = AU deactivating (muscle releasing). These are two completely different behavioral events:

- Positive `AU4_velocity` = brow furrow onset — beginning of cognitive load
- Negative `AU4_velocity` = brow furrow release — end of cognitive load, possible relief or resolution

By taking `.abs()` before computing mean and max, you merge these two events into a single undifferentiated "AU4 is changing" signal. A window with rapid brow furrow onset followed by equally rapid release has the same `AU4_velocity_mean` as a window with sustained moderate furrowing at the same average rate.

The directionality information is scientifically important. A pattern of sharp onset followed by suppressed release (the subject consciously relaxing the brow after a stress spike) is a very different behavioral signature from symmetric fluctuation. But the current features can't distinguish them.

---

## Loophole 6 — `variance` Is Being Used as a Proxy for Two Completely Different Things

Throughout the window engine, `_var` features are computed for almost every channel. `AU1_var`, `head_yaw_var`, `gaze_x_var`, etc.

Variance in a 2-second window can mean two fundamentally different things:

**Interpretation A — Behavioral instability.** High variance = the subject's face/body was changing rapidly and unpredictably. This could indicate anxiety, cognitive load, suppressed emotion.

**Interpretation B — Measurement noise.** High variance = the tracker was jittery on this person's face due to lighting, occlusion, or fast natural movement unrelated to the signal we're looking for.

The system has no way to distinguish these two interpretations. It computes the same `confidence_weighted_var` formula regardless of which source of variance dominates in a given window.

And critically: **the confidence weighting doesn't fully solve this.** A tracker can return very high confidence scores while still producing jittery landmark coordinates if the subject's head is moving naturally (nodding while speaking, for example). The `joint_confidence` score reflects detection confidence, not *coordinate stability*. A perfectly detected face that's actively nodding will have high confidence but high variance across all head pose and landmark features — the system interprets this as behavioral signal when it's actually just natural movement.

---

## Loophole 7 — The Context Mapper Looks Up `end_time_ms`, Not the Window Midpoint

```python
phase_label, q_id, phase_elapsed = context_mapper.lookup(end_time_ms)
```

The context mapper is called with the **end** of the window, not the center.

Think about what this means at a phase transition. Suppose a question ends at `t=10000ms` and the subject begins responding at `t=10000ms`. A window spanning `[9000, 11000]ms` contains the last second of the question and the first second of the response.

The context mapper, called with `end_time_ms=11000`, would label this entire window as `"subject_response"` — because at `t=11000ms`, the response phase has already started.

But half this window's visual and acoustic data was actually captured during the question phase. The HMM transition matrix used for this window is calibrated for "subject_response" behavior, when in reality the behavioral features in this window are a blend of late-question anticipation and early-response anxiety.

This is a **systematic off-by-one error in the context label** for every window that straddles a phase transition. And phase transitions are *exactly the moments when the most behaviorally rich events occur*. The anticipatory stress onset happens in the last second before the subject starts speaking. You're mislabeling those windows and feeding them into the wrong HMM transition profile.

---

## Loophole 8 — The Blink Count Uses `diff() == 1` Which Is Fragile at Window Boundaries

```python
blink_transitions = (window_df["is_blinking"].diff() == 1).sum()
```

`diff()` on `is_blinking` gives `1` at the start of a blink (transition from 0 to 1). This is the intended behavior — count blink onsets.

**The boundary problem:** `window_df` is a slice of the full DataFrame. When you call `.diff()` on a slice, the first row's `diff()` is computed as `first_row_value - NaN = NaN`, not `first_row_value - previous_row_value_from_full_df`.

This means: if a blink was *in progress* at the start of the window (blink started at the end of the previous window and carries into this one), that blink is **invisible** to this window's count. The onset event happened outside the slice, so `diff()` at the first row returns NaN, which is not equal to 1, so it's not counted.

Every blink that straddles a window boundary is undercounted by 1 in the window it falls into, and overcounted by 0 in the previous window (since the onset was correctly counted there). This systematically underestimates blink rate in windows that follow a period of high blinking, which is precisely the scientifically important moment — elevated blink rate is a known indicator of cognitive arousal.

The undercounting is not random noise. It has a systematic bias: it always discards blinks that start in the final frames of a previous window and complete in the first frames of the current window.

---

## Loophole 9 — The Co-occurrence Indices Are Multiplied Without Any Non-Linearity Guard

```python
record["duchenne_index"] = float(au6_m * au12_m)
record["cognitive_load_index"] = float(au4_m * au1_m)
record["speech_hesitation_index"] = float(au25_m * au26_m)
record["disgust_leak"] = float(au9_m * (1.0 - au12_m))
```

These are raw multiplications of AU mean values, which are themselves Z-score normalized after calibration.

**After Z-score normalization**, `AU6_mean` and `AU12_mean` can be negative (when the subject's activation is *below* their personal baseline). Multiplying two negative Z-scores gives a *positive* Duchenne index — the system interprets "both AUs are below baseline" as equivalent to "both AUs are above baseline" in terms of the Duchenne smile signal.

That's wrong. A Duchenne index is supposed to detect *co-activation* of AU6 (orbicularis oculi) and AU12 (zygomaticus major). It should only be positive when both are active. After Z-score normalization, you cannot use multiplication to detect co-activation anymore — you need to check the sign of both components before multiplying.

**The `disgust_leak` formula has an additional problem:**

```python
float(au9_m * (1.0 - au12_m))
```

After Z-score normalization, `au12_m` can be any real number — it's no longer bounded in `[0, 1]`. If `au12_m = 2.5` (AU12 is highly active, 2.5 standard deviations above baseline), then `(1.0 - 2.5) = -1.5`. The formula produces a *negative* disgust leak when the subject is smiling intensely, which is the opposite of what it should indicate.

The formula was designed for raw AU intensities in `[0, 1]` range. After Z-score calibration, the mathematical meaning breaks down completely.

---

## The Big Picture Summary for Part 2

The overarching theme across all of these is: **the window architecture was designed as if all data streams are temporally homogeneous, clean, and already in the right mathematical space — and none of that is true.**

| Assumption | Reality |
|---|---|
| 2-second windows capture behavioral events completely | Micro-expressions are diluted; long events are split across windows |
| Video (30fps) and audio (50Hz HuBERT) cover the same temporal slice equivalently | Different temporal resolutions; cross-modal interaction is blurred before fusion |
| `np.arange` produces precise window boundaries | Floating-point drift accumulates over long sessions |
| `_velocity_max` with confidence weighting detects micro-expression peaks | The suppression is strongest exactly at the frames where micro-expressions occur |
| `.abs()` on velocity is a reasonable simplification | It destroys onset/offset directionality — two different behavioral events become identical |
| `variance` means behavioral instability | Tracker jitter from natural movement is indistinguishable from genuine arousal |
| Context label reflects the window's content | Label is assigned from window end — straddle-frames at transitions are systematically mislabeled |
| Blink count is accurate | Cross-boundary blinks are systematically dropped |
| Co-occurrence indices work on Z-scored data | Negative Z-scores break the multiplication logic — and `disgust_leak` has an unbounded term |

The most structurally dangerous of these is the **Z-score incompatibility of the co-occurrence formulas** — because that one isn't just imprecise, it actively produces *semantically inverted* output values after calibration. And the **context label assignment from end_time** issue is dangerous because it's precisely at phase transitions where the most critical behavioral events occur.

3
I now have the complete picture. Let me think through this with full depth.

---

# Part 3 — Visual Feature Extraction, MediaPipe/OpenFace Parallelism, and Spatial Coherence

*Files examined: `parallel_pool.py`, `unified_pipeline.py`, `hand_face_distance.py`, `wrist_velocity.py`. I am also paying close attention to the architectural fork between the two pipelines — because there's something important hiding there.*

---

## First — Notice That Two Completely Different Pipelines Exist

Before we get into the specific loopholes, there is an architectural fact that needs to be stated plainly:

There are **two separate visual extraction pipelines** that exist in this codebase:

1. **`UnifiedMediaPipePipeline`** in `unified_pipeline.py` — the original sequential pipeline. Uses `static_image_mode=False` (tracking mode), processes frames one-by-one, maintains state across frames in instance variables (`self._prev_mor`, `self._prev_macro_pose`, etc.).

2. **`ParallelMediaPipePool`** in `parallel_pool.py` — the Target #12 parallel replacement. Uses `static_image_mode=True` (stateless per-frame detection), 12 pre-forked workers, re-ordering buffer in master.

These two pipelines produce coordinates in fundamentally different ways because `static_image_mode=True` vs `static_image_mode=False` are **not equivalent**. In tracking mode, MediaPipe uses temporal context from previous frames to stabilize landmark predictions. In static mode, each frame is solved independently from scratch. The landmark positions from these two modes on the same frame can differ meaningfully — especially on profile faces, partial occlusions, or fast-moving subjects.

The changelog says the parallel pool achieves "100% mathematical parity" with the sequential pipeline. That claim is wrong at the detector level. The downstream math may be the same, but the raw input coordinates are not equivalent. Everything built on top of those coordinates inherits that divergence.

---

## Loophole 1 — All Spatial Coordinates Are in Raw Pixel Space Without Camera Normalization

This is the central, foundational problem of the entire visual pipeline. It affects every single kinematic feature.

Look at how landmarks are extracted everywhere:

```python
result["nose_x"] = lms[0].x * width
result["nose_y"] = lms[0].y * height
result["left_wrist_x"] = lms[15].x * width
```

And in `unified_pipeline.py`:
```python
nose = curr_pose_dict[0]   # [x, y, z] — normalized, not pixel
left_wrist = curr_pose_dict[15]
left_hand_face_dist = float(np.linalg.norm(left_wrist - nose))
```

MediaPipe's normalized landmark coordinates are in `[0, 1]` space relative to the input frame dimensions. When you multiply by `width` and `height`, you get pixel coordinates. Pixel coordinates have no physical meaning — they are entirely dependent on three things:

- **Camera zoom / focal length** — a subject filmed with a telephoto lens appears much larger in pixel space than the same subject at the same physical distance filmed with a wide-angle lens
- **Subject-to-camera distance** — a subject who leans 15cm closer to the camera during an answer will have all pixel coordinates expand proportionally
- **Frame resolution** — a 1080p crop gives coordinates twice as large as a 540p crop of the same scene

**The proximity bias problem you asked about:**

When the subject leans forward — which is a completely natural and common behavior during an emphatic answer, a nervous response, or a moment of high engagement — every pixel-space coordinate changes. The wrist moves toward the camera too. The nose moves toward the camera. But they don't move at the same rate, because the wrist is at a different physical depth than the face.

So `left_hand_face_distance` computed in pixel space for a subject leaning forward:
- Physically, the hand and face got closer together (the subject leaned in)
- In pixel space, the hand pixels and face pixels might actually move *farther apart* — because the face (closer to camera) subtended a larger angle and expanded faster in pixel space than the wrist

The system would record this as "hand moved away from face" when physically the subject brought their hand closer to their face. The feature is geometrically inverted.

And for wrist velocity:
- Subject leans forward by 10cm over 1 second
- The camera doesn't move
- All body pixel coordinates expand proportionally as the subject gets closer
- The "wrist velocity" in pixel space spikes — even if the wrist is physically stationary relative to the subject's own body

A subject doing nothing but shifting their weight toward the camera registers as a burst of high wrist velocity. An anxious lean-forward gesture — which has genuine behavioral significance — is mathematically indistinguishable from the pixel expansion artifact of camera proximity. And the system has absolutely no way to separate these two.

---

## Loophole 2 — The Hand-Face Distance Is 2D in the Legacy Module, 3D Elsewhere, and These Are Silently Mixed

In `hand_face_distance.py`:
```python
left_merge["left_hand_face_distance"] = np.sqrt(
    (left_merge["x_left"] - left_merge["x_nose"])**2 +
    (left_merge["y_left"] - left_merge["y_nose"])**2
)
```

This is a **2D Euclidean distance** in the XY pixel plane. The Z coordinate (depth) is completely ignored.

In `unified_pipeline.py`:
```python
curr_pose_dict = {i: np.array([lm.x, lm.y, lm.z]) for i, lm in enumerate(pose_lm)}
nose = curr_pose_dict[0]
left_wrist = curr_pose_dict[15]
left_hand_face_dist = float(np.linalg.norm(left_wrist - nose))  # 3D norm
```

This is a **3D norm** in normalized MediaPipe coordinate space (where Z is relative depth).

So depending on which code path a given session runs through, the `left_hand_face_distance` column can represent two mathematically different things. If these sessions are ever combined in a training dataset, the ML is learning from features with different geometric semantics labeled identically.

But even in the 3D case, there is a hidden problem: MediaPipe's Z coordinate is not metric depth. It's a relative depth estimate scaled roughly to the hip width. It is not in the same unit as X and Y, which are in normalized-image-plane coordinates. Computing a Euclidean norm across three dimensions where the units are heterogeneous produces a geometrically meaningless number. The 3D distance computed from MediaPipe landmarks is not a real spatial distance — it's a mixed-unit artifact.

---

## Loophole 3 — The `collect_results()` Re-Ordering Buffer Blocks Forever If Any Worker Crashes

Look at the drain loop:

```python
for _ in range(self._tasks_submitted):
    result = self.result_queue.get()
    raw_results[result["frame_id"]] = result
```

This loop runs exactly `self._tasks_submitted` times. It expects exactly that many results to come back from the result queue.

**The crash scenario:**

1. Frame #847 is dispatched to Worker #3
2. Worker #3 encounters an unhandled exception during `holistic.process()` that is not caught by the broad `except Exception` block — for example, a segfault in the MediaPipe C++ graph, which Python's exception system cannot catch
3. Worker #3 dies without putting anything into `result_queue`
4. The master's drain loop is now waiting for `self._tasks_submitted` results, but only `self._tasks_submitted - 1` will ever arrive
5. `result_queue.get()` on the last iteration blocks indefinitely — the master is deadlocked

The broad `except Exception` only catches Python exceptions. C++ segfaults, memory violations, and CUDA errors that propagate through the C++ extension layer are signals, not Python exceptions. They kill the process without going through Python's exception handling at all.

This deadlock is permanent. The batch daemon's `process.join()` call will wait forever. The GPU semaphore slot for this session is never released. Eventually no more sessions can be processed until the operator manually kills the stuck daemon.

And there's no heartbeat, no timeout, no watchdog — just an infinite block on a queue get.

---

## Loophole 4 — The solvePnP Camera Matrix Is Fabricated From Frame Dimensions

Look at the camera matrix used in both pipelines:

```python
focal_length = width
center = (width / 2, height / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1))
```

The focal length is set to `width` — the pixel width of the frame. This is a rough approximation based on the assumption that the camera has a roughly 90-degree field of view. It's a heuristic that's commonly used in quick demos and prototypes.

**Why this is a problem for a clinical analysis system:**

solvePnP's accuracy is entirely dependent on having an accurate camera intrinsic matrix. The focal length is the most important parameter. If the actual camera focal length differs from `width` — which it almost certainly does for any real interview setup — then:

- The angular projection of the face landmarks onto the image plane is computed with the wrong scale factor
- The solved rotation angles (yaw, pitch, roll) are systematically biased
- The magnitude of the bias depends on how far the actual focal length deviates from `width`

For a camera with a narrow field of view (telephoto, zoom), the actual focal length in pixels is much larger than `width`. solvePnP then underestimates the true head angle. A subject who is actually turned 25° appears to be at 15°. For a wide-angle camera (conference room, security cam), the actual focal length is smaller than `width`, and solvePnP overestimates angles.

Every session conducted on different hardware — different camera, different lens, different zoom level — produces head pose angles that are systematically wrong in a different direction. These are not random errors that cancel out. They are deterministic, per-device biases.

After Z-score normalization against the baseline, this might partially cancel out *within* a session (if the camera doesn't move). But across sessions with different hardware, the head pose features are not comparable at all.

---

## Loophole 5 — `RQDecomp3x3` Angle Scaling Is Applied With an Arbitrary Multiplier

```python
angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
result["head_pitch"] = float(angles[0] * 360)
result["head_yaw"]   = float(angles[1] * 360)
result["head_roll"]  = float(angles[2] * 360)
```

`cv2.RQDecomp3x3` returns Euler angles in **degrees** from the RQ decomposition. The output is already in degrees. Multiplying by `360` has no geometric basis whatsoever.

Let's think about what this means concretely. If `RQDecomp3x3` returns `yaw = 0.07` (which in its convention represents 0.07 degrees of rotation — essentially forward-facing), the pipeline records `head_yaw = 25.2`. A subject who is looking almost directly at the camera is reported as turned 25 degrees.

If the function returns `yaw = 0.2` (20 degrees of yaw — a visible but moderate turn), the pipeline records `head_yaw = 72`. 

If the actual yaw is already in radians (which it is in some OpenCV versions), multiplying by 360 converts from radians to a number that is 360/(2π) ≈ 57x too large.

The exact interpretation depends on the OpenCV version and how `RQDecomp3x3` is implemented in that version — but regardless, the `* 360` multiplier is not a documented conversion factor for this function. It's either wrong by convention, or it produces values in a completely non-standard range that makes head pose numbers uninterpretable.

More importantly: this exact same code appears in *both* `unified_pipeline.py` and `parallel_pool.py`, meaning the bug propagates through both pipeline paths identically. The FFT spectral engine then runs on these inflated yaw/pitch values, looking for micro-tremors in the 4–10 Hz band. It's computing the frequency spectrum of numbers that are 57x to 360x too large — any frequency or amplitude analysis is meaningless.

---

## Loophole 6 — The Macro Motion Energy Resets to Zero After a Gap, Then Immediately Computes a False Spike

In the parallel pool's `collect_results()`:
```python
if fid in self._gap_frames:
    ...
    prev_macro_pose = None  # Reset
    continue

# Valid frame
if raw["_has_pose"] and raw["macro_landmarks"] is not None:
    if prev_macro_pose is not None:
        deltas = np.linalg.norm(raw["macro_landmarks"] - prev_macro_pose, axis=1)
        macro_energy = float(deltas.mean())
    else:
        macro_energy = 0.0  # First frame after gap → energy = 0
    prev_macro_pose = raw["macro_landmarks"].copy()
```

The logic says: after a gap, the first valid frame gets `macro_energy = 0.0`. This makes sense — you can't compute a delta without a previous frame.

**But the second valid frame after the gap has a catastrophic problem.**

Suppose the gap is 10 frames long. The subject's body moved significantly during those 10 frames (completely unobserved). Then tracking resumes.

- Frame N-10 (last before gap): subject's wrist was at pixel position (300, 400)
- Gap: frames N-9 through N-1 lost
- Frame N (first after gap): macro_energy = 0.0 ✓ (correctly reset)
- Frame N+1: the system computes `delta = landmark[N+1] - landmark[N]`

This is correct — the delta between two consecutive recovered frames. But here's what was lost: during the gap, the body moved from position A to position B over 10 frames. The system assigns **zero energy to that entire period**, and then correctly measures only the small delta between frames N and N+1.

A 10-frame absence where the subject stood up, gestured, and sat back down registers as a period of zero motion followed by normal small deltas. The most physically active segment of the session is reported as the most still. And `postural_stillness = e^{-0.5 * 0.0} = e^0 = 1.0` — the system assigns maximum stillness score to a period where the subject was physically most active.

This inverts the signal for exactly the behavioral moments you most need to track — large gestures and postural shifts are precisely what gets dropped.

---

## Loophole 7 — `static_image_mode=True` in Workers Destroys EAR Reliability

In `parallel_pool.py`, workers use `static_image_mode=True`. In `unified_pipeline.py`, the sequential version uses `static_image_mode=False`.

The Eye Aspect Ratio (EAR) blink detection depends on landmark #159 (upper eyelid) and #145 (lower eyelid) being accurately localized. In tracking mode, MediaPipe stabilizes these landmarks temporally across frames — the lower eyelid position in frame N is informed by its position in frame N-1. This temporal context makes the eyelid detection smooth and reliable.

In static mode, every frame is independently solved. Eyelid landmarks are among the least geometrically constrained in the 3D face mesh — they're thin, small, and highly sensitive to illumination changes. Without temporal context, the predicted eyelid landmark positions jump stochastically between frames, even on a completely still face.

**The consequence:**

EAR values in the parallel pool are significantly noisier than in the sequential pipeline. A subject with eyes wide open might produce EAR values randomly dipping below the `0.21` blink threshold due to landmark jitter — registering phantom blinks on completely open eyes. Conversely, a genuine slow blink might not produce a clean EAR dip if the stochastic error pushes the landmark slightly upward during the blink itself.

Blink rate is described in the changelog as "among the most validated cognitive load indicators in the behavioral literature." If the blink detection is corrupted by static mode jitter, one of the most scientifically grounded features in the entire schema becomes noise.

---

## Loophole 8 — The MOR Velocity Threshold Is Applied in Pixel Units Without Normalization

```python
self.velocity_threshold = 0.01
is_moving = 1 if mor_velocity > self.velocity_threshold else 0
```

`mor_velocity = abs(raw_mor - prev_mor)`

`raw_mor` is a ratio: `vertical_lip_dist_pixels / horizontal_lip_dist_pixels`. Being a ratio, it's approximately dimensionless with respect to frame resolution — which is good.

But look at how the landmarks are computed:
```python
p_top = np.array([face[13].x * width, face[13].y * height])
p_btm = np.array([face[14].x * width, face[14].y * height])
vert = np.linalg.norm(p_top - p_btm)
horiz = np.linalg.norm(p_lft - p_rgt)
result["raw_mor"] = float(vert / horiz)
```

The ratio `vert / horiz` is in pixel units on both sides, so the resolution cancels — the ratio itself is stable. A MOR of 0.1 means the mouth opening is 10% of the mouth width, regardless of frame resolution.

**But then the velocity:**

`mor_velocity = abs(MOR_t - MOR_{t-1})` — this is a delta in the ratio space.

The threshold `0.01` means: if the mouth opening ratio changes by more than 1% of the mouth width in one frame (33ms), the lips are considered "moving."

At 30fps, normal connected speech produces MOR deltas of `0.05–0.15` per frame. So a threshold of `0.01` will mark almost any speech as `is_moving=1`. This might be fine for detecting whether someone is speaking at all.

The real problem is this: **the threshold was chosen without accounting for the subject's natural resting MOR variation.** Some subjects have slightly open resting mouths (high baseline MOR). Their resting lip-droop oscillation due to breathing, swallowing, or chin movement produces MOR deltas of `0.008–0.015` — right at the threshold. Their lips are constantly triggering `is_moving=1` even during complete silence.

For these subjects, `is_moving` is a noisy binary signal that fires constantly, and the cross-modal anchoring in Part 1 would then assign them a high `binary_overlap_ratio` during every speaker's audio segments — because their lips appear to be "always moving."

---

## The Big Picture Summary for Part 3

The theme across these loopholes is different from Parts 1 and 2. Here the dominant failure mode is **coordinate system confusion** — the pipeline never formally defines what units its spatial features are in, and never enforces consistency between sessions, camera setups, or pipeline modes.

| Assumption | Reality |
|---|---|
| Pixel coordinates represent physical body positions | They represent camera projection — a subject leaning forward halluccinates wrist velocity |
| Hand-face distance is consistent across modules | 2D in legacy, 3D in unified — different geometric semantics, same column name |
| solvePnP with `focal=width` gives accurate head pose | The camera matrix is a heuristic guess — produces deterministic per-device angular bias |
| `RQDecomp3x3 * 360` is a valid angle conversion | The multiplier has no geometric basis — produces angles 57–360x outside the true range |
| A gap followed by valid frames gives energy = 0, then correct values | The first post-gap frame gets zero energy while the subject's actual motion during the gap is permanently lost |
| `static_image_mode=True` workers produce equivalent EAR to tracking mode | Static mode landmark jitter creates phantom blinks and misses real slow blinks |
| `collect_results()` always drains all submitted tasks | A worker C++ crash leaves one result permanently missing — the drain loop deadlocks |

The most dangerous of these is the **RQDecomp3x3 scaling bug** — because it doesn't just introduce noise, it produces numbers in a completely wrong range that *look reasonable* in isolation (25 degrees of yaw doesn't seem crazy) but are mathematically ungrounded. Every downstream consumer of head pose — the FFT spectral engine, the variance features, the window aggregation — is processing these inflated numbers and producing results that appear valid but are not.

The second most dangerous is the **proximity bias** in pixel-space kinematics — because it means the wrist velocity and hand-face distance features are contaminated by camera geometry artifacts that are indistinguishable from genuine behavioral signal. A nervous lean-forward and a camera-proximity artifact produce exactly the same feature values.

4
Understood. Clean pivot. Let me read every relevant file before I say a word.

Listed directory SPOVNOB_NEW
Listed directory app
Listed directory Yolo_v8
Listed directory PersonTracking4
Listed directory openface_pipeline
Listed directory src
Listed directory api
Viewed batch_daemon.py:1-632
Viewed face_lock.py:1-108
Viewed detector.py:1-90
Viewed extractor.py:1-253
Viewed tracker.py:1-22

Now I need to find `main_pipeline.py` — the orchestrator that connects all of this.

Searched for "class MultimodalProductionOrchestrator"
Searched for "MultimodalProductionOrchestrator"
Searched for "main_pipeline.py"
Ran command: `find /Users/anshu/Documents/SPOVNOB_NEW -name "main_pipeline.py" 2>/dev/null`

`main_pipeline.py` doesn't exist on disk yet — it's imported dynamically by the daemon but hasn't been written. That itself is a critical finding. Now I have the full picture. Let me write the analysis.

---

# Part 4 — Raw Extraction Layer Adversarial Audit
*(Scope: `batch_daemon.py`, `unified_pipeline.py`, `extractor.py`, `acoustic_extractor.py`, `face_lock.py`, `detector.py`)*

*This analysis covers only what the SAFF + ST-GAT graph network will consume directly: the raw node feature streams. No windowing, no calibration, no HMM.*

---

## The First Thing You Need to Know

`main_pipeline.py` — the file that `batch_daemon.py` dynamically imports at the heart of every session — **does not exist on disk.** The daemon at line 297 does `from main_pipeline import MultimodalProductionOrchestrator`. There is no such file anywhere in the project tree.

This means the current pipeline has never run end-to-end in its current form. The batch daemon, the GPU semaphore, the subprocess spawning, the ELAN label injection — all of that machinery activates and immediately crashes on the import line. The "production engine" produces no output at all right now.

This is not a small bug. This is the foundation. Everything we analyze below is the design of a system whose central orchestration layer is missing.

---

## Section 1 — OpenFace and MediaPipe Parallelism: Chronological Ordering Risk

**What the design intends:**

Two extraction paths run across the same video:
- MediaPipe (`parallel_pool.py`) — 12 CPU workers processing frames in parallel, re-ordering by `frame_id` in master
- OpenFace (`extractor.py`) — sequential GPU batched processing, one `process_batch()` call per batch of frames

The intent is to produce two synchronized per-frame CSVs that share a `frame_id` and `timestamp_ms` index, which will become node features in the ST-GAT.

**The fundamental ordering architecture between the two:**

MediaPipe explicitly sorts its output:
```python
all_frame_ids = sorted(set(raw_results.keys()) | set(self._gap_frames.keys()))
```

OpenFace's `process_batch()` preserves positional order via:
```python
results = [None] * B  # Pre-allocated slots
# ...
results[idx]["faces"] = [...]  # Written back to position
```

So far, so good — each pipeline individually guarantees chronological order within itself.

**Where the ordering guarantee breaks down:**

The two pipelines run independently with no shared clock signal between them. When their outputs are eventually merged into a single per-frame feature row (either in `main_pipeline.py` which doesn't exist yet, or wherever fusion happens), the merge key will be `frame_id` or `timestamp_ms`.

But look at how timestamps are computed in each pipeline:

- **MediaPipe** (`parallel_pool.py`): `timestamp_ms` is passed as a parameter at task submission time — likely computed as `frame_id / fps * 1000` by whatever dispatches the tasks
- **OpenFace** (`extractor.py`): `timestamp_ms` is passed as a parameter to `process_batch()` from `timestamps_list` — also computed externally

Both timestamps are externally computed by the calling code — which is `main_pipeline.py`, which doesn't exist. If the non-existent orchestrator computes timestamps differently for each path (e.g., one uses `cap.get(cv2.CAP_PROP_POS_MSEC)` and the other uses `frame_id / fps * 1000`), the timestamps will not align perfectly.

`cv2.CAP_PROP_POS_MSEC` drifts with demuxer latency. `frame_id / fps * 1000` assumes perfectly constant FPS. For a variable-frame-rate (VFR) recording — common in modern smartphones, body cameras, and interview recording setups — these two approaches will diverge by 10–50ms over a one-hour session.

When the ST-GAT's graph edges connect visual nodes to acoustic nodes by timestamp proximity, a 50ms drift means the edge between an AU activation and the acoustic frame it's supposed to co-represent is systematically wrong for every node in the second half of a long session.

---

## Section 2 — OpenFace Has No Identity Filter

This is the most structurally dangerous gap in the raw extraction layer.

Look at `extractor.py`'s `process_batch()`:

```python
# We lock onto the single primary face in the person crop (highest conf)
bbox = dets[0, :4].cpu().numpy()
conf = float(dets[0, 4].cpu().item())
```

The comment says "lock onto the primary face in the person crop." But this is a RetinaFace detection inside an already-cropped region — the crop comes from the YOLO person bounding box.

**The critical question: whose person bounding box?**

`FaceLock` in `face_lock.py` is a separate module that provides target identity verification via InsightFace cosine similarity. It is called from `detector.py` → `PersonDetector`. But `OpenFaceExtractor` doesn't use `FaceLock` at all. It receives crops and processes them with zero knowledge of whether the crop belongs to the target or the interviewer.

The OpenFace pipeline has **no identity gate**. It will happily extract 114 AU values from whatever face is in the crop it receives. If the crop passed to `process_batch()` is the interviewer's crop (because the non-existent `main_pipeline.py` passed the wrong YOLO box), every AU, every gaze vector, and every emotion label in that batch reflects the interviewer's face — with no warning, no flag, and no way to detect this after the fact.

The YOLO + FaceLock system is responsible for ensuring only the target's crop reaches OpenFace. But since `main_pipeline.py` doesn't exist, that routing logic hasn't been written. The connection between `FaceLock.match()` and `OpenFaceExtractor.process_batch()` exists only in intention, not in code.

---

## Section 3 — The Identity Lock Has a Catastrophic Threshold Design

Even setting aside the missing orchestrator, the identity lock mechanism itself has deep vulnerabilities.

```python
self.target_embedding = None
self.threshold = 0.45

def lock(self, frame, bbox):
    faces = self.app.get(crop)
    if len(faces) == 0:
        return False
    self.target_embedding = faces[0].embedding
    return True

def match(self, frame, detections):
    if self.target_embedding is None:
        return None, 0.0
    ...
    if best_score > self.threshold:
        return best_bbox, clamped_score
    return None, 0.0
```

**Loophole A — Lock is called once, never updated, never verified:**

`lock()` is a one-shot operation. It takes the first face found in the provided bounding box crop and stores that embedding forever as `self.target_embedding`.

If the operator calls `lock()` on a frame where the target's face is partially occluded, poorly lit, wearing glasses they later remove, or turned at a steep angle — the reference embedding is computed from a degraded face representation. Every subsequent `match()` call then compares live frames against a poor-quality reference. Similarity scores will be systematically lower for the actual target and the `threshold=0.45` floor will reject valid matches more frequently.

**Loophole B — The threshold is a scalar constant with no calibration:**

`0.45` is a hard-coded cosine similarity threshold. InsightFace's `buffalo_l` model's cosine similarity distribution varies meaningfully across ethnicities, age groups, and illumination conditions. A threshold calibrated on one demographic may produce an unacceptable false-rejection rate on another.

More critically: when the target's face becomes occluded (they look down, cover their face, turn away), `match()` returns `None, 0.0`. This is the correct behavior. But the pipeline then has no face to crop for OpenFace. The gap handling logic — what gets written for those frames — is again in `main_pipeline.py` which doesn't exist.

**Loophole C — `faces[0]` is always selected during `lock()` and `match()`:**

In `lock()`:
```python
self.target_embedding = faces[0].embedding
```

In `match()`:
```python
emb = faces[0].embedding
```

InsightFace returns detected faces sorted by confidence, not by size or position. If the crop contains two faces (unlikely inside a tight YOLO person box, but possible if two people are physically close), `faces[0]` is the one InsightFace detected with highest confidence — which may not be the target.

**Loophole D — YOLO's `track()` IDs can swap on occlusion:**

`PersonDetector.detect_and_track()` uses YOLO's built-in ByteTrack:
```python
results = self.model.track(frame, persist=True, classes=[0], conf=0.4, verbose=False)
ids = results[0].boxes.id.cpu().numpy()
```

ByteTrack assigns track IDs and attempts to maintain them across occlusion. But ByteTrack's re-identification is appearance-based — it uses IoU and object appearance, not face embeddings. When the target and interviewer cross paths physically, or when one leaves frame and re-enters, ByteTrack can swap their track IDs.

After an ID swap, the YOLO box that was `id=1` (target) is now `id=2` (interviewer) and vice versa. The system then crops the interviewer's region and passes it to InsightFace for matching. InsightFace's cosine similarity against the target embedding will be low (different person), so `match()` returns `None`. But this looks identical to a genuine occlusion event. The pipeline has no way to distinguish "target is genuinely hidden" from "YOLO swapped IDs and now can't find the target because it's looking in the wrong place."

Every frame after an undetected ID swap is an occlusion gap in the output data, when in fact the target is clearly visible — just being tracked under the wrong YOLO ID.

---

## Section 4 — HuBERT Chronological Indexing Has a Hidden Alignment Problem

`HuBERTAcousticExtractor` produces features per call to `extract_window_features(start_ms, end_ms)`. For the SAFF + ST-GAT, this means each acoustic node needs a precise timestamp.

The current system assigns acoustic features by **window boundary**, not by **frame timestamp**:
```python
acoustic_features = acoustic_extractor.extract_window_features(start_time_ms, end_time_ms)
record.update(acoustic_features)
```

The acoustic feature vector represents a 2-second chunk of audio, but it's stamped with `start_time_ms` and `end_time_ms` of a visual window. When building graph edges in the ST-GAT, temporal proximity between nodes requires precise timestamps. But a HuBERT window spanning `[4000, 6000]ms` is a single point in the graph — it has no sub-2-second temporal resolution.

More specifically: HuBERT Layer 7 internally produces ~99 latent frames over 2 seconds, each spaced ~20ms apart. These 99 frames are then **collapsed into a single feature vector** via mean/variance/norm operations. All of that internal temporal structure is destroyed before the graph ever sees it.

For the SAFF architecture specifically — which is designed to capture synchronization between modalities — collapsing 99 acoustic frames into 1 graph node means you have at most 30fps temporal resolution on the visual side and 0.5fps effective temporal resolution on the acoustic side (one node per 2 seconds). The synchronization the SAFF is supposed to capture doesn't exist at the node level.

If the new architecture is to work correctly, the acoustic stream needs to contribute one graph node every ~20ms (50Hz), not one node every 2 seconds. The current `extract_window_features()` architecture is fundamentally incompatible with the SAFF + ST-GAT design at its current granularity.

---

## Section 5 — VRAM Isolation Correctly Structured, But Has a Single Point of Failure

The GPU isolation strategy in `batch_daemon.py` is well-designed:
```python
self.spawn_ctx = multiprocessing.get_context("spawn")
process = self.spawn_ctx.Process(target=_gpu_worker_entrypoint, ...)
```

Spawning a new process with `"spawn"` guarantees a fresh CUDA context. When the session ends, the subprocess terminates and all VRAM is returned to the OS. This correctly prevents VRAM leaks across sessions.

**The single point of failure:**

```python
await loop.run_in_executor(None, process.join)
```

There is no timeout on `process.join`. If the subprocess hangs — due to a CUDA kernel freeze, an infinite loop inside `main_pipeline.py`, or a deadlock in the MediaPipe pool — the `run_in_executor` call blocks indefinitely.

The asyncio event loop is not blocked (the executor runs in a thread pool), so other sessions can still queue up and wait for the GPU semaphore. But the hung session holds the GPU semaphore slot permanently. With `GPU_SEMAPHORE_LIMIT = 2`, if two sessions hang simultaneously, no further sessions can be processed. The daemon appears alive (the watchdog is still monitoring the intake directory, the event loop is still running) but is silently inoperable. Nothing is logged, no alert is raised, and the ledger never transitions the hung sessions out of `TENSORRT_ACTIVE`.

---

## The Big Picture: What the SAFF + ST-GAT Actually Inherits

When you build the new graph architecture on top of these raw outputs, here is what you're inheriting:

| Raw Node Type | Current State | Risk to SAFF + ST-GAT |
|---|---|---|
| MediaPipe kinematics | Chronologically ordered within its own pipeline | Merge with OpenFace requires a timestamp contract that doesn't exist yet |
| OpenFace AUs (114) | No identity gate — can process any face | Every AU node could silently describe the interviewer |
| HuBERT acoustic | Collapsed to 1 node per 2 seconds | Incompatible with 50Hz SAFF synchronization design |
| YOLO identity track | ByteTrack IDs can swap on occlusion | Graph edges built on swapped IDs encode wrong identity |
| InsightFace lock | One-shot lock, no recalibration, hard threshold | False rejections after occlusion leave gaps; ID swaps are invisible |
| `main_pipeline.py` | Does not exist | The entire data collection pipeline produces zero output |

The most urgent facts for the new architecture are two:

**One** — `main_pipeline.py` must be written before anything else, because it defines how crops are routed, how timestamps are unified, how OpenFace receives only the target's crop, and how the two extraction streams are merged into a single per-frame row.

**Two** — The acoustic stream's granularity must be redesigned from "one vector per 2-second window" to "one vector per HuBERT frame (~20ms)" if SAFF synchronization is to work. That means exposing Layer 7 hidden states as individual timestamped nodes, not aggregated statistics. The current `extract_window_features()` API is not the right interface for the new architecture.

---

5
Read the code completely before writing a single word.

Viewed baseline_calibrator.py:1-160

Good. Now I need to look at what data feeds into this calibrator — specifically the windowed CSV structure, the NaN handling, and the `dynamic_window_engine.py` to understand exactly what shape of data arrives here.

Viewed confidence_math.py:1-437

I now have everything. Let me think through this fully and carefully.

---

# Part 5 — Baseline Calibration & Z-Score Normalization

*File: `baseline_calibrator.py` (160 lines). Supporting context from `confidence_math.py` for understanding the NaN landscape arriving at this stage.*

---

## What This Stage Is Claiming to Do

The calibrator's design thesis is stated in its docstring:

> *"This transforms raw features into behavioral deviation signals, enabling the downstream ML to detect anomalies relative to the subject's personal behavioral fingerprint — not absolute thresholds."*

That thesis is sound in principle. Z-score normalization against a personal baseline is scientifically justified — it's exactly how you compare individuals across sessions without demographic bias. The *idea* is correct. But the implementation makes four specific assumptions that can each silently destroy the mathematical validity of the entire output. Let's go through them one by one.

---

## Loophole 1 — The "Terrified Baseline" Trap: Temporal Assumption With No Content Validation

```python
baseline_mask = df['start_time_ms'] < self.calibration_duration_ms  # 30,000ms
baseline_df = df[baseline_mask]
baseline_mean = baseline_df[feature_cols].mean()
baseline_std = baseline_df[feature_cols].std()
```

The baseline is defined as a **pure time boundary**: anything before 30 seconds is "neutral." Nothing in the code verifies what actually happened during that 30 seconds.

**The Terrified Baseline scenario:**

A subject enters the interview room already at high psychological arousal. They've been waiting, they're nervous, they've been told the nature of the interview in advance. The first 30 seconds — while they're being seated, given instructions, and the recording starts — they are already operating at elevated:
- AU4 (brow furrow) — significantly above their true resting state
- head_yaw_var — actively looking around the room nervously
- EAR — elevated blink rate from anxiety
- gaze_entropy — scanning erratically

The calibrator computes `baseline_mean` from these 30 seconds of already-elevated behavior. It then Z-scores the entire session against this contaminated reference.

**What happens to the math:**

`μ_baseline = high` (because the baseline was already anxious)
`σ_baseline = moderate` (because there was natural variation in the anxiety)

Later, when the subject calms down (common — people often relax after the first few minutes once the interview routine becomes familiar), their `AU4` returns to a genuinely neutral low value. The Z-score is:

`Z = (low_value - high_baseline_mean) / baseline_std`

This is **negative**. The system interprets the subject's genuine relaxation as "below baseline AU4 activity" — meaning the model sees what looks like emotional suppression or emotional flattening during what is actually the subject's most honest, relaxed behavior.

And conversely: if the subject encounters a genuinely stressful question in minute 10 and returns to their initial anxiety level, their `AU4` Z-score is near `0` — "baseline equivalent." The system sees no deviation. The most psychologically significant moments register as neutral, because the baseline itself was pathological.

This is not a rare edge case. This is the default scenario for high-stakes interview subjects. The system optimistically labels the "most anxious possible moment of the session" as "neutral" and builds all subsequent comparisons against it.

There is no content validation anywhere in the calibrator — no check on whether `mean_baseline_dev` is already high, no flag if the baseline period has high internal variance (which would indicate the subject wasn't neutral at all), no mechanism for an operator to define an alternative baseline window.

---

## Loophole 2 — Zero-Std Replacement With NaN Is Mathematically Correct But Operationally Catastrophic

```python
zero_std_features = baseline_std[baseline_std == 0].index.tolist()
baseline_std = baseline_std.replace(0, np.nan)
```

The docstring comment explains the design intent:

> *"A zero std means the feature was perfectly constant during baseline — any deviation from that constant will be marked as infinite z-score, which NaN correctly represents as 'uncalibrateable.'"*

This reasoning is logical in isolation. But follow what happens downstream.

**Step 1:** `baseline_std` has NaN for the zero-std feature.

**Step 2:**
```python
df_calibrated[feature_cols] = (df[feature_cols] - baseline_mean) / baseline_std
```

Dividing by NaN produces NaN for every row of that feature column — including the baseline period itself and the entire test period.

**Step 3:**
```python
z_scores = df_calibrated[feature_cols]
df_calibrated['deviation_magnitude'] = np.sqrt((z_scores ** 2).sum(axis=1))
```

`(NaN ** 2) = NaN`. And `NaN + finite_number = NaN`. So if **any** feature has a zero std during baseline, the `deviation_magnitude` for **every single window** in the session becomes NaN.

Wait — actually `sum(axis=1)` in pandas uses `skipna=True` by default. Let's be precise:

```python
(z_scores ** 2).sum(axis=1)
```

With `skipna=True` (the default), the NaN terms are skipped. So if 3 features are NaN and 97 are valid, the sum is over the 97 valid squared z-scores. That means `deviation_magnitude` is computed but **silently excludes the NaN features from the L2 norm**. The magnitude is no longer the L2 norm of the full feature vector — it's the L2 norm of the subset of features that survived calibration, without any normalization for the reduced dimensionality.

**The hidden consequence:**

Sessions where more features have zero baseline std (naturally still subjects, controlled posture, minimal facial movement) will have `deviation_magnitude` computed from fewer features — a smaller-dimensional vector. Their magnitudes are systematically smaller (fewer terms in the sum) not because they're behaviorally more stable in the test period, but because their baseline happened to suppress more features to constant values.

The `deviation_percentile` ranking is then meaningless across sessions — you're comparing L2 norms from different dimensional spaces.

---

## Loophole 3 — No `epsilon` in the Formula: The Division Hits a Hard Cliff

You asked specifically about `(x - mu) / (sigma + epsilon)`. Let's look at the actual formula:

```python
df_calibrated[feature_cols] = (df[feature_cols] - baseline_mean) / baseline_std
```

There is **no epsilon**. It's `/ baseline_std` directly.

The zero-std case is handled by replacing 0 with NaN before the division. This avoids the literal division-by-zero error — Python/numpy won't crash. But it means there is no graceful degradation. It's a binary: either you have a non-zero std and calibration works, or you have zero std and the feature becomes permanently NaN for the entire session.

**Why this matters for nearly-zero std features:**

Consider a subject who is extremely controlled during the baseline — barely moving, barely expressing. Their `AU4_mean` baseline std might be `0.003`. It's not zero — so it doesn't get replaced with NaN. It's just very small.

Then in the test period, they encounter an emotionally loaded question and produce `AU4_mean = 0.04` (not a huge absolute value, just a moderate brow furrow).

The Z-score: `(0.04 - baseline_mean) / 0.003`

If `baseline_mean ≈ 0.01`, then `Z = (0.04 - 0.01) / 0.003 = 10.0`.

A Z-score of **10** from a completely normal AU4 activation. Not a pathological value. Not a micro-expression spike. Just a mild brow furrow that the baseline happened to have almost none of.

**This is not prevented by epsilon.** Epsilon is not in the formula at all. And even if you added it at `1e-9`, it would do nothing for a sigma of `0.003` — the ratio would still be 10. The epsilon prevents the infinite case (sigma = 0) but does nothing for the very-small-sigma case, which is the realistic edge case in a controlled baseline scenario.

Z-scores of 10–50 will appear regularly in sessions with controlled baseline subjects. These values will dominate the `deviation_magnitude` L2 norm and make every high-Z feature look like a critical anomaly, regardless of the absolute behavioral magnitude it represents.

---

## Loophole 4 — NaN Contamination From ID Swaps Is Invisible to `pandas.mean()` and `pandas.std()`

You asked how the calibrator handles NaN values from face occlusions and YOLO ID swaps. The answer is: `pandas.mean()` and `pandas.std()` both use `skipna=True` by default. NaN values are silently excluded.

```python
baseline_mean = baseline_df[feature_cols].mean()   # skipna=True by default
baseline_std = baseline_df[feature_cols].std()     # skipna=True by default
```

So NaN windows in the baseline don't cause errors — they're just excluded from the calculation.

**But "excluded" has a specific and dangerous meaning here.**

Suppose during the baseline period, frames 100–400 (roughly 10 seconds of a 30-second baseline) were lost due to a YOLO ID swap — the system couldn't track the target's face. Those frames produced NaN for all face-related features: all AUs, gaze vectors, head pose, EAR.

The calibrator's baseline statistics for `AU4_mean` are then computed from only the non-NaN windows in the baseline — the first and last 10 seconds, but not the middle 10 seconds.

**The specific failure:** if the subject happened to be *more active* in the first and last 10 seconds (leaning forward as they sat down, looking around as they settled) and *calmer* in the middle 10 seconds (which were lost), then `baseline_mean` for `AU4` reflects a more active period than the subject's true resting state. The calibration reference is biased toward the non-missing periods — and there's no way to know how biased it is.

There is no NaN rate warning in the calibrator. It doesn't log how many windows were dropped from the baseline calculation. It doesn't flag if more than 20% of baseline windows were NaN. The calibrator processes whatever valid rows it finds, computes statistics, and proceeds — silently producing calibration statistics that might be based on 2 windows out of an expected 28.

The `baseline_window_count >= 2` minimum check at line 68 is the only guard. If even 2 windows survived the baseline period, the calibrator proceeds as if the calibration is valid.

---

## Loophole 5 — The Calibrator Normalizes The Baseline Period Against Itself: A Mathematical Tautology

```python
# Z-score normalize ALL windows (including baseline)
df_calibrated[feature_cols] = (df[feature_cols] - baseline_mean) / baseline_std
```

The baseline windows themselves are Z-scored against the baseline statistics. Since `baseline_mean` was computed from the baseline windows, the Z-scores of the baseline windows are centered near zero by construction. This means:

```python
mean_baseline_dev = df_calibrated.loc[baseline_mask, 'deviation_magnitude'].mean()
```

This value will **always** be near zero — it's mathematically guaranteed by the way Z-scoring works. The calibration summary log then says:

> *"Baseline mean deviation: 0.012 (should be near 0 by construction)"*

The comment "should be near 0 by construction" is the tell. It's not a validation check. It's a tautology. The baseline *always* produces a deviation near zero regardless of whether the baseline was actually neutral or anxious — because the mean is subtracted out.

The log summary then shows the "ratio (test/baseline)" — how much more deviant the test period is compared to the baseline. But since the baseline always looks flat (because you normalized it to be flat), this ratio can never warn you that the baseline itself was already abnormal. A session where a terrified subject gradually calms down will show a high "test/baseline ratio" — falsely appearing as if the test period was highly deviant — when in reality the subject was *less* deviant in the test period. The ratio is inverted.

---

## Loophole 6 — The `deviation_magnitude` L2 Norm Is Dominated by High-Variance Features and Ignores Low-Variance Ones

```python
z_scores = df_calibrated[feature_cols]
df_calibrated['deviation_magnitude'] = np.sqrt((z_scores ** 2).sum(axis=1))
```

The `deviation_magnitude` is a straight L2 norm across all feature Z-scores. Every feature contributes equally to the sum of squared Z-scores.

**The dominance problem:**

Some features have naturally high variability relative to their baseline even under neutral conditions — acoustic features like `hubert_latent_0` through `hubert_latent_15` can fluctuate substantially even when the subject is speaking neutrally (because prosodic variation is inherent to connected speech). These 16 channels might each contribute Z-scores of 1–3 in every non-baseline window just from natural speech.

Meanwhile, a genuinely anomalous AU4 spike — the most scientifically important event in the session — contributes one large Z-score.

The `deviation_magnitude` is then `sqrt(16 channels × 4 + 1 × 100) = sqrt(164) ≈ 12.8` in the speech-noise case versus `sqrt(16 × 4 + 1 × 0) = sqrt(64) = 8` in the anomaly case.

The window with the AU4 spike has a *lower* deviation magnitude than the window with normal speech acoustic variation. The L2 aggregation systematically suppresses sparse high-magnitude signals (real behavioral events) and rewards dense low-magnitude noise (acoustic speech variation).

No feature weighting, no feature selection, and no dimensionality consideration is applied before the L2 norm. The magnitude is a raw sum that favors whichever features have the most columns, not whichever features are most behaviorally informative.

---

## The Full Picture for Calibration

The most important thing to understand about this stage is that its failures are **the ones most likely to be mistaken for behavioral signal**. The terrified baseline produces inverted Z-scores. The zero-std NaN contamination silently shrinks the magnitude's effective dimensionality. The very-small-sigma amplification turns normal behavior into Z-score spikes. And the L2 norm buries real anomalies under acoustic noise.

None of this produces an error. None of this logs a warning. The calibrator completes, writes a CSV with plausible-looking numbers, reports a ratio that sounds reasonable, and marks the session `COMPLETED` in the ledger.

| Assumption | Reality |
|---|---|
| First 30 seconds are neutral | Could be the most anxious moment — baseline is contaminated, all comparisons are inverted |
| Zero-std features become NaN cleanly | They silently change the L2 norm's effective dimensionality across different sessions |
| No epsilon needed because zero-std is handled | Very-small-sigma features produce Z-score spikes of 10–50 from ordinary behavior |
| `pandas.mean()` handles NaN from ID swaps gracefully | It excludes them silently — if 80% of baseline is NaN, statistics come from 20% without warning |
| Baseline Z-scores near zero validates calibration | It's a mathematical tautology — baseline is always near zero after subtracting its own mean |
| L2 norm across all features captures total deviation | Acoustic speech variation (16 HuBERT channels) dominates and buries sparse AU anomalies |

The one thing the calibrator does correctly is the atomic file write protection in `batch_daemon.py`. The calibrated CSV, if produced, won't be partially written. But the numbers inside it may be mathematically incoherent from the first byte.

---

That completes the full diagnostic autopsy. All five stages — Voice Isolation, Cross-Modal Anchoring & Windowing, Visual Extraction, Raw Extraction Layer, and Baseline Calibration — have been torn apart. We have a complete vulnerability map of the old architecture. Ready to begin planning the SAFF + ST-GAT refactor whenever you are.


