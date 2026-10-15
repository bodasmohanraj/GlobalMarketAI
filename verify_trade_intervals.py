
import pandas as pd

df = pd.read_csv("gold_backtest_nonoverlap.csv")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

df = df.sort_values("datetime").reset_index(drop=True)

gaps = df["datetime"].diff().dropna().dt.total_seconds() / 3600

print("\n--- TRADE INTERVAL VERIFICATION ---")
print("Rows:", len(df))
print("Minimum hours between decisions:", gaps.min())
print("Median hours between decisions:", gaps.median())
print("Intervals under 4 hours:", int((gaps < 4).sum()))
print("Intervals exactly 4 hours:", int((gaps == 4).sum()))
print("Intervals over 4 hours:", int((gaps > 4).sum()))

duration = (
    df["target_time"] - df["datetime"]
).dt.total_seconds() / 3600

print("\nHolding period hours:")
print(duration.value_counts().sort_index().to_string())

print("\nFirst 10 decision intervals:")
print(
    df[["datetime", "target_time", "signal"]]
    .head(10)
    .to_string(index=False)
)
