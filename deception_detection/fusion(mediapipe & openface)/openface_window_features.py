import pandas as pd
import os

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(
    BASE_DIR,
    "processed_openface_features.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "windowed_openface_features.csv"
)

# =========================
# CONFIG
# =========================

FPS = 30
WINDOW_SIZE_SECONDS = 2

WINDOW_SIZE_FRAMES = FPS * WINDOW_SIZE_SECONDS

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_CSV)

print("Loaded processed OpenFace features")
print(df.head())

# =========================
# DETECT AU COLUMNS
# =========================

au_columns = [
    col for col in df.columns
    if col.startswith("AU_")
]

print(f"\nDetected {len(au_columns)} AU columns")

# =========================
# WINDOWING
# =========================

window_results = []

num_frames = len(df)

window_id = 0

for start_idx in range(
    0,
    num_frames,
    WINDOW_SIZE_FRAMES
):

    end_idx = start_idx + WINDOW_SIZE_FRAMES

    window_df = df.iloc[start_idx:end_idx]

    # Skip incomplete windows
    if len(window_df) < WINDOW_SIZE_FRAMES:
        continue

    # =========================
    # WINDOW INFO
    # =========================

    start_frame = (
        window_df["frame"].iloc[0]
    )

    end_frame = (
        window_df["frame"].iloc[-1]
    )

    # =========================
    # EMOTION FEATURES
    # =========================

    emotion_mode = (
        window_df["emotion"]
        .mode()[0]
    )

    emotion_mean = (
        window_df["emotion"]
        .mean()
    )

    # =========================
    # GAZE FEATURES
    # =========================

    gaze_x_mean = (
        window_df["gaze_x"]
        .mean()
    )

    gaze_x_std = (
        window_df["gaze_x"]
        .std()
    )

    gaze_y_mean = (
        window_df["gaze_y"]
        .mean()
    )

    gaze_y_std = (
        window_df["gaze_y"]
        .std()
    )

    # =========================
    # AU FEATURES
    # =========================

    au_features = {}

    for au_col in au_columns:

        au_features[f"{au_col}_mean"] = (
            window_df[au_col].mean()
        )

        au_features[f"{au_col}_std"] = (
            window_df[au_col].std()
        )

        au_features[f"{au_col}_max"] = (
            window_df[au_col].max()
        )

    # =========================
    # SAVE WINDOW FEATURES
    # =========================

    result = {

        "window_id": window_id,

        "start_frame": start_frame,
        "end_frame": end_frame,

        # Emotion
        "emotion_mode": emotion_mode,
        "emotion_mean": emotion_mean,

        # Gaze
        "gaze_x_mean": gaze_x_mean,
        "gaze_x_std": gaze_x_std,

        "gaze_y_mean": gaze_y_mean,
        "gaze_y_std": gaze_y_std
    }

    # Add AU features
    result.update(au_features)

    window_results.append(result)

    window_id += 1

# =========================
# SAVE OUTPUT
# =========================

output_df = pd.DataFrame(window_results)

output_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nOpenFace windowing completed.")

print("\nSaved to:")
print(OUTPUT_CSV)
