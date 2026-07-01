import pandas as pd
import numpy as np
from pathlib import Path
import logging


class BaselineCalibrator:
    def __init__(self, calibration_duration_ms: float = 30000.0):
        """
        Stage G — Baseline Calibration Engine.

        Learns what is "normal" for a specific subject using the first N
        milliseconds of windowed behavioral data, then Z-score normalizes
        all subsequent windows against that baseline.

        This transforms raw features into *behavioral deviation signals*,
        enabling the downstream ML to detect anomalies relative to the
        subject's personal behavioral fingerprint — not absolute thresholds.

        Args:
            calibration_duration_ms: Duration of the neutral baseline period
                in milliseconds. Default 30,000ms (30 seconds) as specified
                in the architectural blueprint.
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("Baseline_Calibrator")
        self.calibration_duration_ms = calibration_duration_ms

    def calibrate(self, windowed_csv_path: str, output_csv_path: str = None) -> str:
        """
        Executes Z-score normalization against the subject's baseline period.

        Steps:
            1. Partition windows into baseline (first 30s) and test periods
            2. Compute per-feature mean and standard deviation from baseline
            3. Z-score normalize ALL windows: z = (x - μ) / σ
            4. Compute per-window deviation_magnitude (L2 norm of z-scores)

        Args:
            windowed_csv_path: Path to the windowed features CSV
                (output of DynamicWindowEngine or temporal_window_generator).
            output_csv_path: Optional output path. If None, appends
                '_calibrated' to the input filename.

        Returns:
            Path to the calibrated CSV file.
        """
        input_path = Path(windowed_csv_path)
        if not input_path.exists():
            self.logger.error(f"Cannot find windowed tensor at {input_path}")
            return None

        if not output_csv_path:
            output_csv_path = str(input_path.parent / f"{input_path.stem}_calibrated.csv")

        self.logger.info(f"Loading windowed features: {input_path.name}")
        df = pd.read_csv(input_path)

        if df.empty:
            self.logger.warning("Windowed tensor is empty. Aborting calibration.")
            return None

        # --- 1. Partition into baseline and test periods ---
        baseline_mask = df['start_time_ms'] < self.calibration_duration_ms
        baseline_df = df[baseline_mask]
        baseline_window_count = len(baseline_df)

        if baseline_window_count < 2:
            self.logger.warning(
                f"Insufficient baseline windows ({baseline_window_count}). "
                f"Need at least 2 windows within the first {self.calibration_duration_ms}ms. "
                f"Calibration cannot proceed — outputting raw features unchanged."
            )
            df.to_csv(output_csv_path, index=False)
            return output_csv_path

        self.logger.info(
            f"Baseline period: {baseline_window_count} windows "
            f"(first {self.calibration_duration_ms / 1000.0:.0f} seconds)"
        )

        # --- 2. Identify numeric feature columns (exclude metadata) ---
        metadata_cols = [
            'window_id', 'start_time_ms', 'end_time_ms',
            'frame_count', 'cumulative_confidence', 'blink_count',
            'emotion_label_mode'
        ]
        feature_cols = [
            c for c in df.columns
            if c not in metadata_cols
            and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

        self.logger.info(f"Calibrating {len(feature_cols)} numeric features...")

        # --- 3. Compute per-feature baseline statistics ---
        baseline_mean = baseline_df[feature_cols].mean()
        baseline_std = baseline_df[feature_cols].std()

        # Guard: Replace zero std with NaN to prevent division by zero.
        # A zero std means the feature was perfectly constant during baseline —
        # any deviation from that constant will be marked as infinite z-score,
        # which NaN correctly represents as "uncalibrateable."
        zero_std_features = baseline_std[baseline_std == 0].index.tolist()
        if zero_std_features:
            self.logger.warning(
                f"Constant features during baseline (zero std): "
                f"{zero_std_features}. These will be NaN in calibrated output."
            )
        baseline_std = baseline_std.replace(0, np.nan)

        # --- 4. Z-score normalize ALL windows (including baseline) ---
        df_calibrated = df.copy()
        df_calibrated[feature_cols] = (df[feature_cols] - baseline_mean) / baseline_std

        # --- 5. Compute per-window deviation magnitude ---
        # L2 norm of all z-scores across features. This is a single scalar
        # that captures "how far from baseline is this window, overall?"
        # High deviation_magnitude = the subject is behaving unusually.
        z_scores = df_calibrated[feature_cols]
        df_calibrated['deviation_magnitude'] = np.sqrt(
            (z_scores ** 2).sum(axis=1)
        )

        # --- 6. Compute per-window deviation rank ---
        # Percentile rank of deviation magnitude (0–1 scale).
        # Useful for ML as a pre-computed anomaly indicator.
        df_calibrated['deviation_percentile'] = (
            df_calibrated['deviation_magnitude']
            .rank(pct=True, na_option='keep')
        )

        # --- 7. Save calibrated output ---
        df_calibrated.to_csv(output_csv_path, index=False)

        # --- 8. Log calibration summary ---
        test_mask = ~baseline_mask
        if test_mask.any():
            mean_baseline_dev = df_calibrated.loc[baseline_mask, 'deviation_magnitude'].mean()
            mean_test_dev = df_calibrated.loc[test_mask, 'deviation_magnitude'].mean()
            ratio_str = f"{mean_test_dev / mean_baseline_dev:.2f}x" if mean_baseline_dev > 1e-9 else "N/A (baseline = 0)"
            self.logger.info(
                f"Calibration Summary:\n"
                f"  Baseline mean deviation: {mean_baseline_dev:.3f} "
                f"(should be near 0 by construction)\n"
                f"  Test period mean deviation: {mean_test_dev:.3f}\n"
                f"  Ratio (test/baseline): {ratio_str}"
            )

        self.logger.info(f"✅ Baseline Calibration Complete. Output saved to: {output_csv_path}")
        return output_csv_path


# --- Execution Block ---
if __name__ == "__main__":
    calibrator = BaselineCalibrator(calibration_duration_ms=30000.0)
    # calibrator.calibrate(
    #     "pipeline_system_outputs/SESSION_001/SESSION_001_raw_features_30fps_windows.csv"
    # )
