import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Shared confidence-weighted math (single source of truth)
from analytics.confidence_math import (
    calculate_gaze_entropy,
    confidence_weighted_mean,
    confidence_weighted_var,
    confidence_weighted_max,
    confidence_weighted_min,
    compute_fft_block_features,
    FFT_COLUMN_NAMES,
)

# Top-level import: prevents repeated module lookup inside hot loop
from audio_isolation.core.acoustic_extractor import ACOUSTIC_COLUMN_NAMES


class TemporalWindowEngine:
    def __init__(self, window_size_sec: float = 2.0, stride_sec: float = 1.0, min_fill_rate: float = 0.25, assumed_fps: float = 30.0, min_confidence_threshold: float = 0.35):
        """
        Ingests high-resolution data and slices it into overlapping temporal windows.
        STRICTLY uses absolute milliseconds to prevent desync during tracking drops.
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("Temporal_Engine")
        
        # Convert to milliseconds for strict floating-point comparisons
        self.window_size_ms = window_size_sec * 1000.0
        self.stride_ms = stride_sec * 1000.0
        self.min_confidence_threshold = min_confidence_threshold
        
        # Calculate the absolute minimum number of frames allowed in a window
        self.max_possible_frames = window_size_sec * assumed_fps
        self.min_frames_required = int(self.max_possible_frames * min_fill_rate)

    def generate_windows(self, raw_csv_path: str, output_csv_path: str = None, acoustic_extractor=None, context_mapper=None) -> str:
        """
        Executes the sliding window aggregation strictly based on absolute time.
        """
        input_path = Path(raw_csv_path)
        if not input_path.exists():
            self.logger.error(f"Cannot find raw tensor at {input_path}")
            return None

        if not output_csv_path:
            output_csv_path = input_path.parent / f"{input_path.stem}_windowed.csv"

        self.logger.info(f"Loading high-res tensor: {input_path.name}")
        df = pd.read_csv(input_path)
        
        if df.empty:
            self.logger.warning("Raw tensor is empty. Aborting.")
            return None

        max_time_ms = df['timestamp'].max()
        if max_time_ms < self.window_size_ms:
            self.logger.warning("Data is shorter than a single window. Aborting.")
            return None

        self.logger.info(f"Slicing into {self.window_size_ms}ms windows with {self.stride_ms}ms strides...")
        
        windowed_data = []
        window_id = 0

        au_columns = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]

        # --- STRICT TIME-BASED SLICING ---
        # We iterate over time, not rows. This guarantees perfect WavLM audio alignment.
        for start_time_ms in np.arange(0.0, max_time_ms - self.window_size_ms + 1.0, self.stride_ms):
            end_time_ms = start_time_ms + self.window_size_ms
            
            # Filter by absolute timestamps
            window_df = df[(df['timestamp'] >= start_time_ms) & (df['timestamp'] < end_time_ms)]
            
            # If tracking was completely lost for this entire 2-second block, skip it
            if window_df.empty:
                continue

            expected_frames = (self.window_size_ms / 1000.0) * 30.0  # Assumes 30 FPS
            
            # Retrieve weights (default to 1.0 if joint_confidence not present in raw file yet)
            if "joint_confidence" in window_df.columns:
                weights = window_df["joint_confidence"].fillna(0.0).values
            else:
                weights = np.ones(len(window_df))
            
            # Quality Gate: Drop windows with insufficient data
            # Requires at least min_frames_required (e.g., 15) valid frames worth of confidence
            cumulative_confidence = np.sum(weights)

            thresh = self.min_confidence_threshold
            
            # --- CONTEXT LOOKUP ---
            if context_mapper:
                phase_label, q_id, phase_elapsed = context_mapper.lookup(end_time_ms)
            else:
                phase_label, q_id, phase_elapsed = np.nan, -1, np.nan

            if cumulative_confidence < self.min_frames_required:
                # Below minimum required confidence: nullify to prevent hallucination
                record = {
                    "window_id": window_id,
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "frame_count": len(window_df),
                    "cumulative_confidence": float(cumulative_confidence),
                    "context_phase": phase_label,
                    "question_id": q_id,
                    "phase_elapsed_ms": phase_elapsed
                }
                null_keys = [
                    "left_wrist_velocity_mean", "left_wrist_velocity_max",
                    "right_wrist_velocity_mean", "right_wrist_velocity_max",
                    "motion_energy_mean", "motion_energy_var",
                    "left_hand_face_distance_min", "right_hand_face_distance_min",
                    "emotion_confidence_mean", "gaze_x_mean", "gaze_x_var",
                    "gaze_y_mean", "gaze_y_var", "gaze_z_mean", "gaze_z_var",
                    "gaze_entropy", "gaze_velocity_mean", "gaze_velocity_var",
                    "head_yaw_mean", "head_yaw_var",
                    "head_pitch_mean", "head_pitch_var",
                    "head_roll_mean", "head_roll_var",
                    "macro_motion_energy_mean", "macro_motion_energy_var",
                    "postural_stillness_mean", "postural_stillness_var",
                    "mismatch_ratio", "silent_speech_duration_ms",
                ]
                for k in null_keys:
                    record[k] = np.nan
                record["emotion_label_mode"] = "N/A"
                for au in au_columns:
                    record[f"{au}_mean"] = np.nan
                    record[f"{au}_max"] = np.nan
                    record[f"{au}_var"] = np.nan
                    record[f"{au}_velocity_max"] = np.nan
                    record[f"{au}_velocity_mean"] = np.nan
                # Blink / EAR nullification
                record["blink_count"] = 0
                record["blink_rate"] = np.nan
                record["ear_mean"] = np.nan
                record["ear_var"] = np.nan
                # Co-occurrence nullification
                record["duchenne_index"] = np.nan
                record["cognitive_load_index"] = np.nan
                record["speech_hesitation_index"] = np.nan
                record["disgust_leak"] = np.nan
                record["postural_freeze_index"] = np.nan

                # WavLM acoustic nullification
                for col in ACOUSTIC_COLUMN_NAMES:
                    record[col] = np.nan

                # FFT periodicity nullification
                for col in FFT_COLUMN_NAMES:
                    record[col] = np.nan
            else:
                # Calculate the explicit features
                record = {
                    "window_id": window_id,
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "frame_count": len(window_df),
                    "cumulative_confidence": float(cumulative_confidence),
                    "context_phase": phase_label,
                    "question_id": q_id,
                    "phase_elapsed_ms": phase_elapsed,
                    
                    # Macro-kinematics
                    "left_wrist_velocity_mean": confidence_weighted_mean(window_df["left_wrist_velocity"], weights, thresh),
                    "left_wrist_velocity_max": confidence_weighted_max(window_df["left_wrist_velocity"], weights, thresh),
                    "right_wrist_velocity_mean": confidence_weighted_mean(window_df["right_wrist_velocity"], weights, thresh),
                    "right_wrist_velocity_max": confidence_weighted_max(window_df["right_wrist_velocity"], weights, thresh),
                    "motion_energy_mean": confidence_weighted_mean(window_df["macro_motion_energy"], weights, thresh),
                    "motion_energy_var": confidence_weighted_var(window_df["macro_motion_energy"], weights, thresh),
                    
                    "left_hand_face_distance_min": confidence_weighted_min(window_df["left_hand_face_distance"], weights, thresh),
                    "right_hand_face_distance_min": confidence_weighted_min(window_df["right_hand_face_distance"], weights, thresh),
                    
                    # Micro-geometry / Psychological
                    "emotion_confidence_mean": confidence_weighted_mean(window_df["emotion_confidence"], weights, thresh),
                    "emotion_label_mode": window_df["emotion_label"].mode().iloc[0] if not window_df["emotion_label"].mode().empty else "N/A",
                    "gaze_x_mean": confidence_weighted_mean(window_df["gaze_x"], weights, thresh),
                    "gaze_x_var": confidence_weighted_var(window_df["gaze_x"], weights, thresh),
                    "gaze_y_mean": confidence_weighted_mean(window_df["gaze_y"], weights, thresh),
                    "gaze_y_var": confidence_weighted_var(window_df["gaze_y"], weights, thresh),
                    "gaze_z_mean": confidence_weighted_mean(window_df["gaze_z"], weights, thresh),
                    "gaze_z_var": confidence_weighted_var(window_df["gaze_z"], weights, thresh),
                    "gaze_entropy": calculate_gaze_entropy(window_df),
                    "gaze_velocity_mean": confidence_weighted_mean(window_df["gaze_velocity"], weights, thresh) if "gaze_velocity" in window_df.columns else np.nan,
                    "gaze_velocity_var": confidence_weighted_var(window_df["gaze_velocity"], weights, thresh) if "gaze_velocity" in window_df.columns else np.nan,

                    "head_yaw_mean": confidence_weighted_mean(window_df["head_yaw"], weights, thresh),
                    "head_yaw_var": confidence_weighted_var(window_df["head_yaw"], weights, thresh),
                    "head_pitch_mean": confidence_weighted_mean(window_df["head_pitch"], weights, thresh),
                    "head_pitch_var": confidence_weighted_var(window_df["head_pitch"], weights, thresh),
                    "head_roll_mean": confidence_weighted_mean(window_df["head_roll"], weights, thresh),
                    "head_roll_var": confidence_weighted_var(window_df["head_roll"], weights, thresh),

                    "macro_motion_energy_mean": confidence_weighted_mean(window_df["macro_motion_energy"], weights, thresh),
                    "macro_motion_energy_var": confidence_weighted_var(window_df["macro_motion_energy"], weights, thresh),
                    "postural_stillness_mean": confidence_weighted_mean(window_df["postural_stillness"], weights, thresh) if "postural_stillness" in window_df.columns else np.nan,
                    "postural_stillness_var": confidence_weighted_var(window_df["postural_stillness"], weights, thresh) if "postural_stillness" in window_df.columns else np.nan,

                    "mismatch_ratio": confidence_weighted_mean(window_df["mismatch_incongruence"], weights, thresh) if "mismatch_incongruence" in window_df.columns else np.nan,
                    "silent_speech_duration_ms": confidence_weighted_mean(window_df["silent_incongruence"], weights, thresh) * self.window_size_ms if "silent_incongruence" in window_df.columns else np.nan,
                }
                # Individual AU aggregation: mean, max, var per AU channel
                for au in au_columns:
                    record[f"{au}_mean"] = confidence_weighted_mean(window_df[au], weights, thresh)
                    record[f"{au}_max"] = confidence_weighted_max(window_df[au], weights, thresh)
                    record[f"{au}_var"] = confidence_weighted_var(window_df[au], weights, thresh)
                # AU Onset Velocity: max velocity per AU per window
                for au in au_columns:
                    vel_col = f"{au}_velocity"
                    if vel_col in window_df.columns:
                        record[f"{au}_velocity_max"] = confidence_weighted_max(window_df[vel_col].abs(), weights, thresh)
                        record[f"{au}_velocity_mean"] = confidence_weighted_mean(window_df[vel_col].abs(), weights, thresh)
                # Blink Rate aggregation
                if "is_blinking" in window_df.columns:
                    blink_transitions = (window_df["is_blinking"].diff() == 1).sum()
                    window_duration_sec = self.window_size_ms / 1000.0
                    record["blink_count"] = int(blink_transitions)
                    record["blink_rate"] = blink_transitions / window_duration_sec
                if "ear" in window_df.columns:
                    record["ear_mean"] = confidence_weighted_mean(window_df["ear"], weights, thresh)
                    record["ear_var"] = confidence_weighted_var(window_df["ear"], weights, thresh)

                # AU Co-occurrence Patterns
                # Use np.nan as default to distinguish "missing data" from "zero expression"
                au6_m = record.get("AU6_mean", np.nan)
                au12_m = record.get("AU12_mean", np.nan)
                au4_m = record.get("AU4_mean", np.nan)
                au1_m = record.get("AU1_mean", np.nan)
                au25_m = record.get("AU25_mean", np.nan)
                au26_m = record.get("AU26_mean", np.nan)
                au9_m = record.get("AU9_mean", np.nan)

                record["duchenne_index"] = float(au6_m * au12_m) if not (np.isnan(au6_m) or np.isnan(au12_m)) else np.nan
                record["cognitive_load_index"] = float(au4_m * au1_m) if not (np.isnan(au4_m) or np.isnan(au1_m)) else np.nan
                record["speech_hesitation_index"] = float(au25_m * au26_m) if not (np.isnan(au25_m) or np.isnan(au26_m)) else np.nan
                record["disgust_leak"] = float(au9_m * (1.0 - au12_m)) if not (np.isnan(au9_m) or np.isnan(au12_m)) else np.nan

                # Postural Freeze Index (PFI): stillness_mean * sum of AU variances
                au_variances = [record.get(f"{au}_var", np.nan) for au in au_columns]
                clean_au_vars = [v for v in au_variances if not np.isnan(v)]
                sum_au_vars = sum(clean_au_vars) if clean_au_vars else np.nan
                stillness_m = record.get("postural_stillness_mean", np.nan)
                if not np.isnan(stillness_m) and not np.isnan(sum_au_vars):
                    record["postural_freeze_index"] = float(stillness_m * sum_au_vars)
                else:
                    record["postural_freeze_index"] = np.nan

                # --- WavLM Acoustic Feature Injection ---
                # Gated by is_audio_active — see the identical comment in
                # analytics/dynamic_window_engine.py (mirrored bug found +
                # verified by review 2026-07-07): extract_window_features
                # only nulls on the isolated WAV's own RMS floor, which is
                # blind to who is speaking, so an attenuated-but-still-loud
                # interviewer segment can leak through as if it were the
                # target's own acoustic behavior.
                if acoustic_extractor is not None:
                    target_speaking = True
                    if "is_audio_active" in window_df.columns:
                        active_frac = confidence_weighted_mean(window_df["is_audio_active"], weights, thresh)
                        target_speaking = not np.isnan(active_frac) and active_frac >= 0.5
                    if target_speaking:
                        acoustic_features = acoustic_extractor.extract_window_features(start_time_ms, end_time_ms)
                    else:
                        acoustic_features = {col: np.nan for col in ACOUSTIC_COLUMN_NAMES}
                    record.update(acoustic_features)
                else:
                    # No extractor provided: populate schema with NaN placeholders
                    for col in ACOUSTIC_COLUMN_NAMES:
                        record[col] = np.nan

                # --- Target #16: Behavioral Periodicity (FFT) ---
                # 4-second lookback on the FULL raw DataFrame (not the 2s window slice)
                fft_features = compute_fft_block_features(df, end_time_ms)
                record.update(fft_features)
            
            windowed_data.append(record)
            window_id += 1

        # Save to CSV
        final_df = pd.DataFrame(windowed_data)
        final_df.to_csv(output_csv_path, index=False)
        
        self.logger.info(f"✅ Generated {len(final_df)} absolute chronological windows.")
        self.logger.info(f"Saved to: {output_csv_path}")
        return str(output_csv_path)

# --- Execution Block ---
if __name__ == "__main__":
    engine = TemporalWindowEngine(window_size_sec=2.0, stride_sec=1.0)
    # engine.generate_windows("pipeline_system_outputs/SESSION_001/SESSION_001_raw_features_30fps.csv")