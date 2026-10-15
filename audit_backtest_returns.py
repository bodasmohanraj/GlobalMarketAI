
import pandas as pd
import numpy as np

df = pd.read_csv("gold_backtest_nonoverlap.csv")

print("\n--- BACKTEST RETURN AUDIT ---")
print("Rows:", len(df))
print("Duplicate decision times:", df["datetime"].duplicated().sum())

for col in [
    "future_return",
    "gross_return",
    "transaction_cost",
    "net_return",
]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()

    print(f"\n{col}")
    print("Mean:", round(s.mean(), 6))
    print("Median:", round(s.median(), 6))
    print("Minimum:", round(s.min(), 6))
    print("Maximum:", round(s.max(), 6))
    print("1st percentile:", round(s.quantile(0.01), 6))
    print("99th percentile:", round(s.quantile(0.99), 6))

print("\nReturn thresholds:")
r = pd.to_numeric(df["future_return"], errors="coerce")
for threshold in [0.01, 0.03, 0.05, -0.01, -0.03, -0.05]:
    count = int((r >= threshold).sum()) if threshold > 0 else int((r <= threshold).sum())
    print(f"{threshold:+.0%}: {count} observations")

print("\nMissing values:")
print(df[["close", "future_close_4h", "future_return"]].isna().sum())
