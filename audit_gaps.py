
import pandas as pd
from pathlib import Path

FILE = Path("gold_features_timebased.csv")

df = pd.read_csv(FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

# Calculate the time between consecutive observations.
df["gap_hours"] = (
    df["datetime"].diff().dt.total_seconds() / 3600
)

print("\n--- TIME GAP AUDIT ---")
print(f"Total rows: {len(df)}")
print(f"Duplicate timestamps: {df['datetime'].duplicated().sum()}")
print("\nGap distribution:")
print(df["gap_hours"].value_counts().sort_index().head(20))

print("\nGaps greater than one hour:")
print(f"{(df['gap_hours'] > 1).sum()}")

print("\nGap percentages:")
print(
    (df["gap_hours"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
    .head(20)
)

print("\nSaved dataset columns:")
print(df.columns.tolist())
