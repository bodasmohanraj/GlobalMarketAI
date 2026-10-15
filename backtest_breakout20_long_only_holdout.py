from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_features_clean.csv"
OUT = BASE / "breakout20_long_only_proper_holdout.csv"

CAPITAL = 100_000.0
HOLDOUT_FRACTION = 0.20
WINDOW = 20
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
MAX_EXIT_DELAY_HOURS = 1

df = pd.read_csv(SOURCE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

required = ["datetime", "open", "high", "low", "close"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Chronological split happens BEFORE signal evaluation.
split = int(len(df) * (1 - HOLDOUT_FRACTION))
holdout_start = df.loc[split, "datetime"]

# The level is based exclusively on the previous 20 candles.
df["resistance"] = df["high"].rolling(WINDOW).max().shift(1)
df["previous_close"] = df["close"].shift(1)

# Breakout confirmed at candle close.
df["signal"] = (
    (df["close"] > df["resistance"]) &
    (df["previous_close"] <= df["resistance"])
).astype(int)

# Only long signals are allowed. No short positions.
times = df["datetime"]
records = []
next_entry_allowed = None
skipped_no_exit = 0
skipped_overlap = 0

for i, row in df.iterrows():
    if row["datetime"] < holdout_start or row["signal"] != 1:
        continue

    # Enter at next actual candle's open.
    entry_i = i + 1
    if entry_i >= len(df):
        skipped_no_exit += 1
        continue

    entry_time = df.loc[entry_i, "datetime"]
    entry_price = float(df.loc[entry_i, "open"])

    if next_entry_allowed is not None and entry_time < next_entry_allowed:
        skipped_overlap += 1
        continue

    desired_exit = entry_time + pd.Timedelta(hours=4)
    exit_i = int(times.searchsorted(desired_exit, side="left"))

    if exit_i >= len(df):
        skipped_no_exit += 1
        continue

    exit_time = df.loc[exit_i, "datetime"]
    delay_hours = (exit_time - desired_exit).total_seconds() / 3600
    if delay_hours > MAX_EXIT_DELAY_HOURS:
        skipped_no_exit += 1
        continue

    exit_price = float(df.loc[exit_i, "close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        continue

    records.append({
        "signal_time": row["datetime"],
        "entry_time": entry_time,
        "exit_time": exit_time,
        "holding_hours": (exit_time - entry_time).total_seconds() / 3600,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return_pct": (exit_price / entry_price - 1) * 100
    })
    next_entry_allowed = exit_time

trades = pd.DataFrame(records)
trades.to_csv(OUT, index=False)

print("\n=== LONG-ONLY BREAKOUT: CHRONOLOGICAL HOLDOUT ===")
print("Total source candles:", len(df))
print("Holdout starts:", holdout_start)
print("Data ends:", df["datetime"].iloc[-1])
print("Rule: close breaks above previous 20-candle high")
print("Entry: next candle open")
print("Exit: first close at/after 4 hours, maximum 1-hour delay")
print("Valid non-overlapping long trades:", len(trades))
print("Skipped for missing/late exit:", skipped_no_exit)
print("Skipped for overlap:", skipped_overlap)

if trades.empty:
    print("No valid trades; no performance conclusions can be drawn.")
    raise SystemExit(0)

results = []
for cost in COSTS:
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    wins = 0
    costs_paid = 0.0

    for ret_pct in trades["gross_return_pct"]:
        start_equity = equity
        charge = start_equity * cost
        costs_paid += charge
        equity = start_equity * (1 + ret_pct / 100) - charge
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
        wins += int(ret_pct > 0)

    results.append({
        "round_trip_cost_pct": cost * 100,
        "trades": len(trades),
        "gross_win_rate_pct": (trades["gross_return_pct"] > 0).mean() * 100,
        "final_equity": equity,
        "net_return_pct": (equity / CAPITAL - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "total_cost_currency": costs_paid
    })

result = pd.DataFrame(results)
print("\n=== COST SENSITIVITY ===")
print(result.to_string(index=False, formatters={
    "round_trip_cost_pct": "{:.2f}%".format,
    "gross_win_rate_pct": "{:.2f}%".format,
    "final_equity": "{:,.2f}".format,
    "net_return_pct": "{:.2f}%".format,
    "max_drawdown_pct": "{:.2f}%".format,
    "total_cost_currency": "{:,.2f}".format
}))

print("\nMonthly returns at 0.10% round-trip cost:")
trades["month"] = trades["entry_time"].dt.strftime("%Y-%m")
trades["net_return"] = (
    (1 + trades["gross_return_pct"] / 100) * (1 - 0.001) - 1
)
monthly = trades.groupby("month").agg(
    trades=("net_return", "size"),
    avg_net_trade_pct=("net_return", "mean")
)
monthly["avg_net_trade_pct"] *= 100
print(monthly.to_string(float_format=lambda x: f"{x:.3f}"))

print("\nSaved:", OUT.name)
print("Research only. Costs are illustrative; not live trading advice.")
