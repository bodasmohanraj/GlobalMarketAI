
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "gold_5m_data.csv"

if not FILE.exists():
    raise SystemExit(f"Missing data file: {FILE}")

df = pd.read_csv(FILE)

df.columns = [
    str(col).strip().lower().replace(" ", "_")
    for col in df.columns
]

if not {"datetime", "close"}.issubset(df.columns):
    raise SystemExit(f"Unexpected columns: {list(df.columns)}")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["datetime", "close"])
df = df.sort_values("datetime")

duplicates = int(df["datetime"].duplicated().sum())
df = df.drop_duplicates("datetime", keep="last")
df = df.set_index("datetime")

times = df.index
close = df["close"]
intervals = times.to_series().diff().dropna()

print("=" * 55)
print("GOLD DATA LEAKAGE AUDIT — RESEARCH ONLY")
print("=" * 55)
print(f"Candles: {len(df)}")
print(f"First UTC: {times.min()}")
print(f"Last UTC:  {times.max()}")
print(f"Duplicate timestamps: {duplicates}")
print(f"Gaps longer than 5 minutes: {(intervals > pd.Timedelta(minutes=5)).sum()}")
print(f"Intervals not exactly 5 minutes: {(intervals != pd.Timedelta(minutes=5)).sum()}")

print("\nEXACT FUTURE-LABEL CHECK")

results = []

for name, minutes in [
    ("15min", 15),
    ("30min", 30),
    ("1hour", 60),
    ("4hour", 240),
]:
    future_times = times + pd.Timedelta(minutes=minutes)
    future_close = close.reindex(future_times).to_numpy()
    current_close = close.to_numpy()
    valid = pd.notna(future_close)

    # Independently calculate direction only when the exact future candle exists.
    labels = np.where(
        valid,
        (future_close > current_close).astype(int),
        np.nan
    )

    mismatches = 0
    for i in np.flatnonzero(valid):
        expected = int(future_close[i] > current_close[i])
        if labels[i] != expected:
            mismatches += 1

    results.append({
        "horizon": name,
        "exact_future_matches": int(valid.sum()),
        "missing_future": int((~valid).sum()),
        "label_mismatches": mismatches,
    })

    print(
        f"{name}: exact matches={int(valid.sum())}, "
        f"missing future={int((~valid).sum())}, "
        f"mismatches={mismatches}"
    )

pd.DataFrame(results).to_csv(
    ROOT / "final_label_audit.csv", index=False
)

print("\nSaved final_label_audit.csv")
print("\nLIMITATION:")
print("This checks raw timestamps and label alignment.")
print("It does not yet verify every model feature or validation fold.")
print("Forecasts must be issued after the current candle closes.")
