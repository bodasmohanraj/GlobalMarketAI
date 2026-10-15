from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_features_clean.csv"
TRADES_OUT = BASE / "breakout20_holdout_trades.csv"
RESULTS_OUT = BASE / "breakout20_holdout_results.csv"

CAPITAL = 100_000.0
HOLDOUT_FRACTION = 0.20
ROUND_TRIP_COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
WINDOW = 20

df = pd.read_csv(SOURCE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

required = ["datetime", "open", "high", "low", "close"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["datetime"].diff().dropna().ne(pd.Timedelta(hours=1)).any():
    raise ValueError(
        "Data contains gaps or non-hourly intervals. "
        "Fix the timeline before running this test."
    )

# Resistance/support are calculated from candles strictly before the signal candle.
df["resistance"] = df["high"].rolling(WINDOW).max().shift(1)
df["support"] = df["low"].rolling(WINDOW).min().shift(1)

# Signal confirmed at the current candle's close.
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

# Chronological holdout: final 20% of timestamps.
split = int(len(df) * (1 - HOLDOUT_FRACTION))
holdout_start = df.loc[split, "datetime"]
df["test_period"] = df["datetime"] >= holdout_start

# Entry: next hourly candle OPEN.
# Exit: close exactly four hours after entry timestamp.
open_lookup = df.set_index("datetime")["open"]
close_lookup = df.set_index("datetime")["close"]
available_times = set(df["datetime"])

records = []
next_entry_allowed = None

for i, row in df.iterrows():
    if not row["test_period"] or row["signal"] == 0:
        continue

    signal_time = row["datetime"]
    entry_time = signal_time + pd.Timedelta(hours=1)
    exit_time = entry_time + pd.Timedelta(hours=4)

    if entry_time not in available_times or exit_time not in available_times:
        continue

    if next_entry_allowed is not None and entry_time < next_entry_allowed:
        continue

    entry = float(open_lookup.loc[entry_time])
    exit_price = float(close_lookup.loc[exit_time])

    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
        continue

    side = int(row["signal"])
    gross_return = side * (exit_price / entry - 1)

    records.append({
        "signal_time": signal_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": "LONG" if side == 1 else "SHORT",
        "signal_close": float(row["close"]),
        "entry_open": entry,
        "exit_close": exit_price,
        "gross_return_pct": gross_return * 100
    })
    next_entry_allowed = exit_time

trades = pd.DataFrame(records)
if trades.empty:
    raise RuntimeError("No valid holdout trades found.")

trades.to_csv(TRADES_OUT, index=False)

results = []
for cost in ROUND_TRIP_COSTS:
    equity = CAPITAL
    peak = equity
    max_drawdown = 0.0
    total_cost = 0.0

    for ret_pct in trades["gross_return_pct"]:
        start_equity = equity
        charge = start_equity * cost
        total_cost += charge
        equity = start_equity * (1 + ret_pct / 100) - charge
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)

    results.append({
        "round_trip_cost_pct": cost * 100,
        "trades": len(trades),
        "long_trades": int((trades["side"] == "LONG").sum()),
        "short_trades": int((trades["side"] == "SHORT").sum()),
        "gross_trade_return_sum_pct": trades["gross_return_pct"].sum(),
        "final_equity": equity,
        "net_return_pct": (equity / CAPITAL - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "total_cost_currency": total_cost
    })

result = pd.DataFrame(results)
result.to_csv(RESULTS_OUT, index=False)

print("\n=== 20-HOUR BREAKOUT: CHRONOLOGICAL HOLDOUT ===")
print("Data rows:", len(df))
print("Holdout starts:", holdout_start)
print("Holdout ends:", df["datetime"].iloc[-1])
print("Entry: next candle open")
print("Exit: close four hours after entry")
print("Non-overlapping trades:", len(trades))
print("\n=== COST SENSITIVITY ===")
print(result.to_string(index=False, formatters={
    "round_trip_cost_pct": "{:.2f}%".format,
    "gross_trade_return_sum_pct": "{:.2f}%".format,
    "final_equity": "{:,.2f}".format,
    "net_return_pct": "{:.2f}%".format,
    "max_drawdown_pct": "{:.2f}%".format,
    "total_cost_currency": "{:,.2f}".format
}))
print("\nSaved trades:", TRADES_OUT.name)
print("Saved results:", RESULTS_OUT.name)
print("Historical research only; no live trades.")
print("Caution: the 20-hour window was selected after earlier full-sample testing.")
