import pandas as pd
import os

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OPENFACE_CSV = os.path.join(
    BASE_DIR,
    "windowed_openface_features.csv"
)

POSE_CSV = os.path.join(
    BASE_DIR,
    "../mediapipe_pose/feature_engineering/windowed_pose_features.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "fused_features.csv"
)

# =========================
# LOAD DATA
# =========================

openface_df = pd.read_csv(OPENFACE_CSV)

pose_df = pd.read_csv(POSE_CSV)

print("Loaded OpenFace windows:")
print(openface_df.shape)

print("\nLoaded Pose windows:")
print(pose_df.shape)

# =========================
# MERGE
# =========================

fused_df = pd.merge(
    openface_df,
    pose_df,
    on="window_id",
    how="inner"
)

# =========================
# SAVE OUTPUT
# =========================

fused_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nFusion completed.")

print("\nSaved to:")
print(OUTPUT_CSV)

print("\nFused Shape:")
print(fused_df.shape)