from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "breakout20_gap_aware_trades.csv"
OUT = BASE / "breakout20_long_only_comparison.csv"

CAPITAL = 100_000.0
COSTS = [0.0005, 0.001]
FINAL_HOLDOUT_FRACTION = 0.40

df = pd.read_csv(SOURCE)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
df["gross_return"] = pd.to_numeric(df["gross_return_pct"]) / 100
df = df.sort_values("entry_time").reset_index(drop=True)

if df.empty:
    raise ValueError("No trades found in source file.")

# Reserve the last 40% of the existing holdout trades chronologically.
cut = int(len(df) * (1 - FINAL_HOLDOUT_FRACTION))
explore = df.iloc[:cut].copy()
final_test = df.iloc[cut:].copy()

def simulate(trades, mode, cost):
    if mode == "long_only":
        trades = trades[trades["side"] == "LONG"]
    elif mode != "long_short":
        raise ValueError("Unknown mode.")

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    costs_paid = 0.0
    wins = 0

    for _, row in trades.iterrows():
        start = equity
        ret = float(row["gross_return"])
        charge = start * cost
        costs_paid += charge
        equity = start * (1 + ret) - charge
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
        wins += int(ret > 0)

    return {
        "trades": len(trades),
        "long_trades": int((trades["side"] == "LONG").sum()),
        "short_trades": int((trades["side"] == "SHORT").sum()),
        "gross_win_rate_pct": (
            (trades["gross_return"] > 0).mean() * 100
            if len(trades) else np.nan
        ),
        "final_equity": equity,
        "net_return_pct": (equity / CAPITAL - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "costs_paid": costs_paid
    }

rows = []
for period_name, subset in [
    ("exploratory_earlier", explore),
    ("reserved_final_holdout", final_test)
]:
    for mode in ["long_short", "long_only"]:
        for cost in COSTS:
            result = simulate(subset, mode, cost)
            result.update({
                "period": period_name,
                "strategy": mode,
                "round_trip_cost_pct": cost * 100,
                "start": subset["entry_time"].min(),
                "end": subset["exit_time"].max()
            })
            rows.append(result)

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)

print("\n=== LONG-ONLY VS LONG-AND-SHORT ===")
print("Source trades:", len(df))
print("Earlier exploratory trades:", len(explore))
print("Reserved final holdout trades:", len(final_test))
print("Exploratory cutoff:", final_test["entry_time"].min())
print("\n=== RESULTS ===")
print(result[[
    "period", "strategy", "round_trip_cost_pct", "trades",
    "long_trades", "short_trades", "gross_win_rate_pct",
    "net_return_pct", "max_drawdown_pct", "costs_paid"
]].to_string(index=False, formatters={
    "round_trip_cost_pct": "{:.2f}%".format,
    "gross_win_rate_pct": lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a",
    "net_return_pct": "{:.2f}%".format,
    "max_drawdown_pct": "{:.2f}%".format,
    "costs_paid": "{:,.2f}".format
}))
print("\nSaved:", OUT.name)
print("Research only; costs are illustrative.")
print("Note: the reserved holdout is small and is not a fully independent dataset.")
