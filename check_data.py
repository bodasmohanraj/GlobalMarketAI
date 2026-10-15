
import pandas as pd

df = pd.read_csv("gold_data.csv")

df["datetime"] = pd.to_datetime(
    df["datetime"], utc=True, errors="coerce"
)

df = df.dropna(subset=["datetime", "close"])
df = df.sort_values("datetime").reset_index(drop=True)

gaps = df["datetime"].diff()

print("\nGOLD DATA QUALITY CHECK")
print(f"Total rows: {len(df)}")
print(f"Duplicate timestamps: {df['datetime'].duplicated().sum()}")
print(f"Missing close values: {df['close'].isna().sum()}")
print(f"First timestamp: {df['datetime'].iloc[0]}")
print(f"Last timestamp: {df['datetime'].iloc[-1]}")

print("\nTime gap distribution:")
print(gaps.value_counts().head(10))

print("\nLargest 10 gaps:")
print(
    pd.DataFrame({
        "previous_time": df["datetime"].shift(1),
        "current_time": df["datetime"],
        "gap": gaps,
    }).nlargest(10, "gap").to_string(index=False)
)
