
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("gold_features_timebased.csv")
OUTPUT_FILE = Path("gold_model_ready.csv")

df = pd.read_csv(INPUT_FILE)

features = [
    "return_1h",
    "return_4h",
    "return_12h",
    "sma_5_gap",
    "sma_20_gap",
    "trend_5_20",
    "volatility_10",
    "volatility_20",
    "rsi_14",
]

df["actual_horizon_hours"] = pd.to_numeric(
    df["actual_horizon_hours"], errors="coerce"
)
df["target_up"] = pd.to_numeric(df["target_up"], errors="coerce")

# Keep rows with all features available and a valid four-hour target.
valid_horizon = df["actual_horizon_hours"].between(4, 4.5)
valid_features = df[features].notna().all(axis=1)
valid_target = df["target_up"].notna()

model_df = df.loc[
    valid_horizon & valid_features & valid_target
].copy()

model_df["target_up"] = model_df["target_up"].astype(int)
model_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- REBUILT MODEL-READY DATASET ---")
print("Source rows:", len(df))
print("Valid features:", int(valid_features.sum()))
print("Valid four-hour targets:", int(valid_horizon.sum()))
print("Rows with both:", len(model_df))
print("\nTarget distribution:")
print(model_df["target_up"].value_counts().sort_index())
print("\nSaved to:", OUTPUT_FILE.resolve())
