
import pandas as pd

df = pd.read_csv("gold_model_ready.csv")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

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

print("\n--- FINAL DATA VALIDATION ---")
print("Rows:", len(df))
print("Duplicate timestamps:", df["datetime"].duplicated().sum())
print("Chronologically sorted:", df["datetime"].is_monotonic_increasing)
print("Missing feature values:", df[features].isna().sum().sum())
print("Missing targets:", df["target_up"].isna().sum())
print("Unexpected target values:",
      sorted(set(df["target_up"].dropna().unique()) - {0, 1}))
print("First timestamp:", df["datetime"].min())
print("Last timestamp:", df["datetime"].max())

print("\nTarget proportions:")
print(df["target_up"].value_counts(normalize=True).sort_index().round(3))
