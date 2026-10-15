
import pandas as pd
import numpy as np

FILE = "gold_data.csv"

df = pd.read_csv(FILE)

print("\n--- RAW GOLD DATA AUDIT ---")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

required = ["datetime", "open", "high", "low", "close", "volume"]
missing_columns = [c for c in required if c not in df.columns]
print("Missing required columns:", missing_columns)

if missing_columns:
    raise ValueError("Required columns are missing.")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values:")
print(df[required].isna().sum())

print("\nDuplicate timestamps:", df["datetime"].duplicated().sum())

bad_prices = (
    (df["open"] <= 0)
    | (df["high"] <= 0)
    | (df["low"] <= 0)
    | (df["close"] <= 0)
)

bad_ohlc = (
    (df["high"] < df[["open", "low", "close"]].max(axis=1))
    | (df["low"] > df[["open", "high", "close"]].min(axis=1))
)

print("Invalid/nonpositive prices:", int(bad_prices.sum()))
print("Inconsistent OHLC candles:", int(bad_ohlc.sum()))
print("Negative volume rows:", int((df["volume"] < 0).sum()))
print("Zero volume rows:", int((df["volume"] == 0).sum()))

print("\nTimestamp range:")
print(df["datetime"].min(), "to", df["datetime"].max())

print("\nFirst 5 rows:")
print(df[required].head().to_string(index=False))

print("\nLast 5 rows:")
print(df[required].tail().to_string(index=False))
