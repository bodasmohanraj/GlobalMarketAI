from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_features_clean.csv"
TRADES_OUT = BASE / "breakout20_gap_aware_trades.csv"
RESULTS_OUT = BASE / "breakout20_gap_aware_results.csv"

CAPITAL = 100_000.0
HOLDOUT_FRACTION = 0.20
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
WINDOW = 20
MAX_EXIT_DELAY_HOURS = 1

df = pd.read_csv(SOURCE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

required = ["datetime", "open", "high", "low", "close"]
if any(c not in df.columns for c in required):
    raise ValueError("Required OHLC columns are missing.")

# Levels use only prior candles.
df["resistance"] = df["high"].rolling(WINDOW, min_periods=WINDOW).max().shift(1)
df["support"] = df["low"].rolling(WINDOW, min_periods=WINDOW).min().shift(1)
df["previous_close"] = df["close"].shift(1)

df["signal"] = np.select(
    [
        (df["close"] > df["resistance"]) &
        (df["previous_close"] <= df["resistance"]),
        (df["close"] < df["support"]) &
        (df["previous_close"] >= df["support"])
    ],
    [1, -1],
    default=0
)

split = int(len(df) * (1 - HOLDOUT_FRACTION))
holdout_start = df.loc[split, "datetime"]

# Build timestamp lookup; no synthetic candles are created.
df = df.reset_index(drop=True)
timestamp_to_index = {t: i for i, t in enumerate(df["datetime"])}
records = []
next_entry_allowed = None
skipped_no_entry = 0
skipped_exit_gap = 0
skipped_overlap = 0

for i, row in df.iterrows():
    if row["datetime"] < holdout_start or row["signal"] == 0:
        continue

    # Enter at the next actual candle's open.
    entry_i = i + 1
    if entry_i >= len(df):
        skipped_no_entry += 1
        continue

    entry_time = df.loc[entry_i, "datetime"]
    entry_price = float(df.loc[entry_i, "open"])

    if next_entry_allowed is not None and entry_time < next_entry_allowed:
        skipped_overlap += 1
        continue

    desired_exit = entry_time + pd.Timedelta(hours=4)

    # Find the first actual candle at or after the desired exit.
    exit_i = int(df["datetime"].searchsorted(desired_exit, side="left"))
    if exit_i >= len(df):
        skipped_no_entry += 1
        continue

    exit_time = df.loc[exit_i, "datetime"]
    delay_hours = (exit_time - desired_exit).total_seconds() / 3600

    if delay_hours > MAX_EXIT_DELAY_HOURS:
        skipped_exit_gap += 1
        continue

    exit_price = float(df.loc[exit_i, "close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        continue

    side = int(row["signal"])
    gross_return = side * (exit_price / entry_price - 1)

    records.append({
        "signal_time": row["datetime"],
        "entry_time": entry_time,
        "exit_time": exit_time,
        "holding_hours": (exit_time - entry_time).total_seconds() / 3600,
        "side": "LONG" if side == 1 else "SHORT",
        "entry_open": entry_price,
        "exit_close": exit_price,
        "gross_return_pct": gross_return * 100
    })
    next_entry_allowed = exit_time

trades = pd.DataFrame(records)
if trades.empty:
    raise RuntimeError("No valid holdout trades. Review data and exit-delay rule.")

trades.to_csv(TRADES_OUT, index=False)
results = []

for cost in COSTS:
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    total_cost = 0.0

    for ret_pct in trades["gross_return_pct"]:
        start_equity = equity
        charge = start_equity * cost
        total_cost += charge
        equity = start_equity * (1 + ret_pct / 100) - charge
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)

    results.append({
        "round_trip_cost_pct": cost * 100,
        "trades": len(trades),
        "long_trades": int((trades["side"] == "LONG").sum()),
        "short_trades": int((trades["side"] == "SHORT").sum()),
        "average_holding_hours": trades["holding_hours"].mean(),
        "final_equity": equity,
        "net_return_pct": (equity / CAPITAL - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "total_cost_currency": total_cost
    })

result = pd.DataFrame(results)
result.to_csv(RESULTS_OUT, index=False)

print("\n=== GAP-AWARE BREAKOUT HOLDOUT ===")
print("Holdout starts:", holdout_start)
print("Holdout ends:", df["datetime"].iloc[-1])
print("Level: previous 20 candles, not necessarily 20 clock hours")
print("Entry: next available candle open")
print("Exit: first available close at/after 4 hours, maximum 1-hour delay")
print("Valid non-overlapping trades:", len(trades))
print("Skipped due to unavailable entry/exit:", skipped_no_entry)
print("Skipped due to exit gap:", skipped_exit_gap)
print("Skipped due to overlapping entry:", skipped_overlap)
print("\n=== COST SENSITIVITY ===")
print(result.to_string(index=False, formatters={
    "round_trip_cost_pct": "{:.2f}%".format,
    "average_holding_hours": "{:.2f}".format,
    "final_equity": "{:,.2f}".format,
    "net_return_pct": "{:.2f}%".format,
    "max_drawdown_pct": "{:.2f}%".format,
    "total_cost_currency": "{:,.2f}".format
}))
print("\nSaved:", TRADES_OUT.name, "and", RESULTS_OUT.name)
print("Historical research only; costs are illustrative.")
