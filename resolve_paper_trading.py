
from pathlib import Path
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
LOG_PATH = BASE / "paper_trading_log.csv"

# Load the log and explicitly use nullable/object-friendly columns.
log = pd.read_csv(LOG_PATH)

log["prediction_time"] = pd.to_datetime(
    log["prediction_time"], utc=True, errors="coerce"
)
log["target_time"] = pd.to_datetime(
    log["target_time"], utc=True, errors="coerce"
)

# Normalize columns that will receive new values.
for col in ["actual_close", "forward_return_pct"]:
    log[col] = pd.to_numeric(log[col], errors="coerce")

log["actual_up"] = log["actual_up"].astype("object")
log["resolved"] = (
    log["resolved"].astype(str).str.lower().eq("true")
)

# Download hourly prices.
prices = yf.download(
    "GC=F",
    period="5d",
    interval="1h",
    progress=False,
    auto_adjust=False,
)

if prices.empty:
    raise RuntimeError("No hourly data downloaded. Log was not updated.")

if isinstance(prices.columns, pd.MultiIndex):
    close = prices["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
else:
    close = prices["Close"]

close.index = pd.to_datetime(close.index, utc=True)
close = close[~close.index.duplicated(keep="last")].dropna()
price_map = close.to_dict()

updated = 0
missing = []

for idx, row in log.iterrows():
    if row["resolved"] or pd.isna(row["target_time"]):
        continue

    target = row["target_time"]
    prediction_close = pd.to_numeric(
        pd.Series([row["prediction_close"]]), errors="coerce"
    ).iloc[0]

    # Only resolve when the exact target candle exists.
    if target not in price_map or pd.isna(prediction_close):
        missing.append(str(target))
        continue

    actual_close = float(price_map[target])
    actual_up = bool(actual_close > float(prediction_close))
    forward_return = (
        actual_close / float(prediction_close) - 1
    ) * 100

    log.at[idx, "actual_close"] = actual_close
    log.at[idx, "actual_up"] = actual_up
    log.at[idx, "forward_return_pct"] = forward_return
    log.at[idx, "resolved"] = True
    updated += 1

# Save only after processing completes.
log.to_csv(LOG_PATH, index=False)

print(f"Resolved records: {updated}")
print(f"Still unavailable: {len(missing)}")

for timestamp in sorted(set(missing)):
    print("Missing target:", timestamp)

print("\nUpdated paper-trading log:")
print(
    log[
        [
            "model",
            "prediction_time",
            "target_time",
            "prediction_close",
            "actual_close",
            "actual_up",
            "forward_return_pct",
            "resolved",
        ]
    ].to_string(index=False)
)

print(f"\nSaved to: {LOG_PATH.name}")
