
import pandas as pd

df = pd.read_csv("gold_features_clean.csv")

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

print("\n--- CLEAN FEATURE DIAGNOSTIC ---")
print("Total rows:", len(df))

print("\nRows with each feature available:")
print(df[features].notna().sum())

print("\nMissing values in the original feature dataset:")
original = pd.read_csv("gold_features_timebased.csv")
print(original[features].isna().sum())

print("\nTarget availability:")
print("Exact four-hour targets:",
      original["actual_horizon_hours"].eq(4).sum())

print("\nFeature completeness by gap size:")
raw = pd.read_csv("gold_data.csv")
raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
raw = raw.sort_values("datetime")
raw["gap_hours"] = raw["datetime"].diff().dt.total_seconds() / 3600

print(raw["gap_hours"].value_counts().sort_index().head(15))
