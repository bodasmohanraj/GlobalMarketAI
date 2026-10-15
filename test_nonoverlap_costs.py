from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "reliable_backtest_walkforward_predictions.csv"
OUTPUT = BASE / "reliable_backtest_nonoverlap_cost_sensitivity.csv"

THRESHOLD = 0.55
INITIAL_CAPITAL = 100_000.0
ROUND_TRIP_COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]

df = pd.read_csv(SOURCE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

required = ["prob_up", "forward_return", "target_time", "datetime"]
if any(c not in df.columns for c in required):
    raise ValueError("Required prediction columns are missing.")

# Select the same kind of non-overlapping four-hour periods
selected = []
next_allowed = None

for _, row in df.iterrows():
    if next_allowed is None or row["datetime"] >= next_allowed:
        selected.append(row)
        next_allowed = row["target_time"]

test = pd.DataFrame(selected).reset_index(drop=True)
p = pd.to_numeric(test["prob_up"], errors="raise").to_numpy()
r = pd.to_numeric(test["forward_return"], errors="raise").to_numpy()

positions = np.where(p >= THRESHOLD, 1, np.where(p <= 1-THRESHOLD, -1, 0))
active = positions != 0

if not (test["datetime"].iloc[1:].reset_index(drop=True) >=
        test["target_time"].iloc[:-1].reset_index(drop=True)).all():
    raise ValueError("Overlapping periods detected.")

results = []
for cost in ROUND_TRIP_COSTS:
    equity = INITIAL_CAPITAL
    peak = equity
    max_drawdown = 0.0
    total_cost = 0.0

    for position, ret in zip(positions, r):
        start_equity = equity
        trade_cost = start_equity * cost if position != 0 else 0.0
        total_cost += trade_cost
        equity = start_equity * (1 + position * ret) - trade_cost
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)

    results.append({
        "round_trip_cost_pct": cost * 100,
        "periods": len(test),
        "trades": int(active.sum()),
        "total_cost_currency": total_cost,
        "final_equity": equity,
        "net_return_pct": (equity / INITIAL_CAPITAL - 1) * 100,
        "max_drawdown_pct": max_drawdown * 100,
    })

out = pd.DataFrame(results)
out.to_csv(OUTPUT, index=False)

print("\n=== NON-OVERLAPPING COST SENSITIVITY ===")
print("Selected periods:", len(test))
print("Active trades:", int(active.sum()))
print("Test start:", test["datetime"].iloc[0])
print("Last target time:", test["target_time"].iloc[-1])
print(out.to_string(index=False, formatters={
    "round_trip_cost_pct": "{:.2f}%".format,
    "total_cost_currency": "{:,.2f}".format,
    "final_equity": "{:,.2f}".format,
    "net_return_pct": "{:.2f}%".format,
    "max_drawdown_pct": "{:.2f}%".format,
}))
print("\nSaved:", OUTPUT.name)
print("Research only; costs are illustrative, not broker quotes.")
