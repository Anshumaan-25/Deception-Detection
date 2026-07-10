import json
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
import logging


def parse_baseline_file_index(source_csv, default: int = 0) -> int:
    """Recover the baseline clip's diarization file_index from a baseline_stats
    ``source_csv`` whose basename is ``<recording_id>_<NNN>_windowed_features.csv``
    (session_id = f"{recording_id}_{file_index:03d}"). Returns ``default`` if the
    suffix cannot be parsed. Downstream consumers (analyst report, replication
    scorecard) must NOT assume the baseline is file_index 0 — process_recording_session
    accepts ``baseline_file_index`` != 0 for mis-named batches."""
    m = re.search(r"_(\d{3})_windowed_features\.csv$", str(source_csv))
    return int(m.group(1)) if m else default


# Columns that are never behavioral features: window bookkeeping, context labels,
# provenance columns added by the recording assembler, and derived deviation
# outputs. Used by the recording-mode fit/apply path. (The legacy single-clip
# calibrate() keeps its historical, narrower list for output parity — note that
# it therefore z-scores question_id / phase_elapsed_ms, a pre-existing wart.)
NON_FEATURE_COLUMNS = [
    'window_id', 'start_time_ms', 'end_time_ms',
    'frame_count', 'cumulative_confidence', 'blink_count',
    'emotion_label_mode',
    'context_phase', 'question_id', 'phase_elapsed_ms',
    'file_index', 'clip_window_id',
    'deviation_magnitude', 'deviation_percentile',
    'target_ground_truth',
]


class BaselineCalibrationError(RuntimeError):
    """The dedicated baseline clip cannot support calibration (recording mode
    fails loudly — uncalibrated deviations must never masquerade as calibrated)."""


@dataclass
class BaselineStats:
    """Frozen per-feature baseline statistics fitted on the dedicated
    baseline/calibration clip. Persisted as JSON next to the recording outputs
    so any later run can reproduce the exact normalization."""

    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]  # NaN where the baseline was constant (zero std)
    baseline_window_count: int
    source_csv: str

    def to_json(self, path: str) -> str:
        payload = asdict(self)
        # NaN is not valid JSON — encode as null, decode back to NaN in from_json.
        payload["feature_stds"] = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in self.feature_stds.items()
        }
        payload["feature_means"] = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in self.feature_means.items()
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        return str(path)

    @classmethod
    def from_json(cls, path: str) -> "BaselineStats":
        with open(path, "r") as f:
            payload = json.load(f)
        payload["feature_stds"] = {
            k: (np.nan if v is None else float(v))
            for k, v in payload["feature_stds"].items()
        }
        payload["feature_means"] = {
            k: (np.nan if v is None else float(v))
            for k, v in payload["feature_means"].items()
        }
        return cls(**payload)


def _feature_columns(df: pd.DataFrame) -> list:
    """Numeric columns that are behavioral features (recording-mode rule)."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]


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

    # ──────────────────────────────────────────────────────────────────
    # Recording mode (Phase A): fit on the dedicated baseline clip,
    # apply to every clip. The legacy calibrate() above stays untouched
    # for the single-clip path.
    # ──────────────────────────────────────────────────────────────────

    def fit(self, baseline_windowed_csv: str) -> BaselineStats:
        """
        Fit per-feature baseline statistics from the dedicated baseline clip.

        Uses EVERY window of the clip (no duration cap — the whole video is
        generic/neutral by design). Raises BaselineCalibrationError if the
        clip cannot support calibration: fewer than 2 windows, or no feature
        produced any usable (non-NaN) data.
        """
        path = Path(baseline_windowed_csv)
        if not path.exists():
            raise BaselineCalibrationError(
                f"Baseline windowed CSV not found: {path}. The baseline clip "
                f"failed upstream — recording cannot be calibrated."
            )

        df = pd.read_csv(path)
        if len(df) < 2:
            raise BaselineCalibrationError(
                f"Baseline clip yielded {len(df)} window(s); need >= 2 to "
                f"estimate variance. Re-record or fix the baseline video."
            )

        feature_cols = _feature_columns(df)
        means = df[feature_cols].mean()
        stds = df[feature_cols].std()

        # Zero-std → NaN: a feature constant during baseline cannot be
        # z-scored; NaN marks it uncalibrateable (same guard as calibrate()).
        zero_std = stds[stds == 0].index.tolist()
        if zero_std:
            self.logger.warning(
                f"Constant features during baseline (zero std): {zero_std}. "
                f"These will be NaN in calibrated output."
            )
        stds = stds.replace(0, np.nan)

        if means.isna().all():
            raise BaselineCalibrationError(
                "Every feature is NaN across the baseline clip (all windows "
                "nullified?). Baseline is unusable — recording cannot be "
                "calibrated."
            )

        self.logger.info(
            f"Baseline fitted: {len(df)} windows (whole clip), "
            f"{len(feature_cols)} features, from {path.name}"
        )
        return BaselineStats(
            feature_means={k: float(v) for k, v in means.items()},
            feature_stds={k: float(v) for k, v in stds.items()},
            baseline_window_count=len(df),
            source_csv=str(path),
        )

    def apply(self, windowed_csv_path: str, stats: BaselineStats,
              output_csv_path: str) -> str:
        """
        Z-score a clip's windowed features against fitted baseline stats.

        Applied to interview clips AND the baseline clip itself (whose
        deviations land near 0 by construction — a built-in sanity check).
        Adds deviation_magnitude; deviation_percentile is deliberately NOT
        computed here — a percentile rank is only meaningful over the whole
        recording, so the recording assembler computes it after concatenation.
        """
        df = pd.read_csv(windowed_csv_path)

        feature_cols = _feature_columns(df)
        fitted = [c for c in feature_cols if c in stats.feature_means]
        unfitted = [c for c in feature_cols if c not in stats.feature_means]
        if unfitted:
            self.logger.warning(
                f"{len(unfitted)} feature column(s) absent from baseline stats "
                f"(left raw): {unfitted[:5]}"
            )

        means = pd.Series({c: stats.feature_means[c] for c in fitted})
        stds = pd.Series({c: stats.feature_stds[c] for c in fitted})

        df_calibrated = df.copy()
        df_calibrated[fitted] = (df[fitted] - means) / stds

        z_scores = df_calibrated[fitted]
        df_calibrated['deviation_magnitude'] = np.sqrt((z_scores ** 2).sum(axis=1))

        df_calibrated.to_csv(output_csv_path, index=False)
        self.logger.info(
            f"✅ Applied baseline stats ({stats.baseline_window_count} baseline "
            f"windows) to {Path(windowed_csv_path).name} → {output_csv_path}"
        )
        return str(output_csv_path)


# --- Execution Block ---
if __name__ == "__main__":
    calibrator = BaselineCalibrator(calibration_duration_ms=30000.0)
    # calibrator.calibrate(
    #     "pipeline_system_outputs/SESSION_001/SESSION_001_raw_features_30fps_windows.csv"
    # )
