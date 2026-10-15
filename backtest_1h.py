import pandas as pd
import numpy as np

PREDICTIONS_FILE = "validation_audit_oos_predictions_v2.csv"
PRICES_FILE = "gold_5m_data.csv"

HOLD_MINUTES = 60
COST_PER_SIDE_BPS = 2.0
SLIPPAGE_PER_SIDE_BPS = 1.0
ROUND_TRIP_COST = 2 * (
    COST_PER_SIDE_BPS + SLIPPAGE_PER_SIDE_BPS
) / 10000

pred = pd.read_csv(PREDICTIONS_FILE)
prices = pd.read_csv(PRICES_FILE)

# Normalize price column names.
prices.columns = [c.strip().lower() for c in prices.columns]

pred["datetime"] = pd.to_datetime(pred["datetime"], utc=True)
pred["target_time"] = pd.to_datetime(pred["target_time"], utc=True)
pred["test_start"] = pd.to_datetime(pred["test_start"], utc=True)
pred["test_end"] = pd.to_datetime(pred["test_end"], utc=True)

prices["datetime"] = pd.to_datetime(prices["datetime"], utc=True)
prices["open"] = pd.to_numeric(prices["open"], errors="coerce")

prices = (
    prices.dropna(subset=["datetime", "open"])
    .drop_duplicates("datetime")
    .sort_values("datetime")
)

open_prices = prices.set_index("datetime")["open"]

# Only evaluate the 1-hour horizon.
df = pred[pred["horizon"].astype(str).str.lower().isin(
    ["1hour", "1h", "60m", "60min"]
)].copy()

if df.empty:
    raise ValueError(
        "No 1-hour rows found. Check the values in the horizon column."
    )

df["decision_model"] = pd.to_numeric(
    df["decision_model"], errors="coerce"
)
df = df.dropna(subset=["decision_model"])

# The signal is known after the candle at datetime closes.
# Enter at the next 5-minute candle's Open.
df["entry_time"] = df["datetime"] + pd.Timedelta(minutes=5)

# Hold for 60 minutes from entry, then exit at the Open.
df["exit_time"] = df["entry_time"] + pd.Timedelta(
    minutes=HOLD_MINUTES
)

df["entry_price"] = df["entry_time"].map(open_prices)
df["exit_price"] = df["exit_time"].map(open_prices)

# Restrict trades to their own chronological test fold.
df["fold_valid"] = (
    (df["datetime"] >= df["test_start"])
    & (df["entry_time"] < df["test_end"])
    & (df["exit_time"] <= df["test_end"])
)

df["side"] = np.where(df["decision_model"] == 1, 1, -1)

# Missing exact timestamps are rejected, not silently approximated.
df["price_valid"] = (
    df["entry_price"].notna()
    & df["exit_price"].notna()
    & (df["entry_price"] > 0)
)

df["gross_return"] = (
    df["side"] * (df["exit_price"] - df["entry_price"])
    / df["entry_price"]
)

df["cost_return"] = ROUND_TRIP_COST
df["net_return"] = df["gross_return"] - df["cost_return"]

candidates = df[
    df["fold_valid"] & df["price_valid"]
].sort_values("entry_time").copy()

# One position at a time globally, including across fold boundaries.
selected = []
last_exit_time = None

for _, row in candidates.iterrows():
    if (
        last_exit_time is not None
        and row["entry_time"] < last_exit_time
    ):
        continue

    selected.append(row)
    last_exit_time = row["exit_time"]

trades = pd.DataFrame(selected)

if trades.empty:
    raise ValueError(
        "No valid trades remain. Inspect timestamps and fold boundaries."
    )

# A simple compounded-return illustration, not account-level sizing.
trades["equity_multiple"] = (1 + trades["net_return"]).cumprod()
trades["running_peak"] = trades["equity_multiple"].cummax()
trades["drawdown"] = (
    trades["equity_multiple"] / trades["running_peak"] - 1
)

summary = {
    "strategy": "next_5m_open_entry_60m_hold_1h_OOS",
    "trades": len(trades),
    "candidates": len(candidates),
    "missing_price_candidates": int(
        (df["fold_valid"] & ~df["price_valid"]).sum()
    ),
    "rejected_fold_candidates": int((~df["fold_valid"]).sum()),
    "cost_per_side_bps": COST_PER_SIDE_BPS,
    "slippage_per_side_bps": SLIPPAGE_PER_SIDE_BPS,
    "round_trip_cost_bps": ROUND_TRIP_COST * 10000,
    "gross_mean_return": trades["gross_return"].mean(),
    "net_mean_return": trades["net_return"].mean(),
    "net_median_return": trades["net_return"].median(),
    "net_win_rate": (trades["net_return"] > 0).mean(),
    "net_compounded_return": trades["equity_multiple"].iloc[-1] - 1,
    "max_drawdown": trades["drawdown"].min(),
    "note": (
        "Illustrative costs; uses next-bar Open prices. "
        "Not a broker execution simulation or evidence of profitability."
    ),
}

trades.to_csv("backtest_1h_trades_corrected.csv", index=False)
pd.DataFrame([summary]).to_csv(
    "backtest_1h_results_corrected.csv", index=False
)

print("\nCORRECTED BACKTEST SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFIRST 3 TRADES")
print(trades[
    ["datetime", "entry_time", "exit_time", "entry_price",
     "exit_price", "side", "gross_return", "cost_return",
     "net_return", "fold"]
].head(3).to_string(index=False))

print("\nLAST 3 TRADES")
print(trades[
    ["datetime", "entry_time", "exit_time", "entry_price",
     "exit_price", "side", "gross_return", "cost_return",
     "net_return", "fold"]
].tail(3).to_string(index=False))

print("\nSaved:")
print("backtest_1h_trades_corrected.csv")
print("backtest_1h_results_corrected.csv")

