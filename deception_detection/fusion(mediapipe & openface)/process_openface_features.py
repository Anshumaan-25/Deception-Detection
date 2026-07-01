import pandas as pd
import ast
import os

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(
    BASE_DIR,
    "../OpenFace-3.0/outputs/predictions.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "processed_openface_features.csv"
)

# =========================
# DEBUG INFO
# =========================

print("Current Working Directory:")
print(os.getcwd())

print("\nReading from:")
print(INPUT_CSV)

# =========================
# LOAD CSV
# =========================

df = pd.read_csv(INPUT_CSV)

print("\nOriginal Columns:")
print(df.columns.tolist())

# =========================
# CONVERT STRING LIST → LIST
# =========================

df["au_values"] = df["au_values"].apply(
    ast.literal_eval
)

# =========================
# CREATE AU COLUMNS
# =========================

num_aus = len(df["au_values"].iloc[0])

print(f"\nDetected {num_aus} AU features")

for i in range(num_aus):

    df[f"AU_{i}"] = df["au_values"].apply(
        lambda x: x[i]
    )

# =========================
# DROP ORIGINAL COLUMN
# =========================

df = df.drop(columns=["au_values"])

# =========================
# SAVE OUTPUT
# =========================

df.to_csv(OUTPUT_CSV, index=False)

print("\nProcessed OpenFace features saved.")

print("\nSaved to:")
print(OUTPUT_CSV)

print("\nFinal Columns:")
print(df.columns.tolist())
