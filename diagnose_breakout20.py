from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "breakout20_gap_aware_trades.csv"
MONTHLY_OUT = BASE / "breakout20_monthly_diagnostics.csv"
SIDE_OUT = BASE / "breakout20_side_diagnostics.csv"

COST = 0.001  # illustrative 0.10% round-trip cost

if not SOURCE.exists():
    raise FileNotFoundError(f"Cannot find {SOURCE.name}")

df = pd.read_csv(SOURCE)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
df = df.sort_values("entry_time").reset_index(drop=True)

required = ["side", "gross_return_pct", "entry_time", "exit_time"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["gross_return"] = pd.to_numeric(df["gross_return_pct"]) / 100
df["net_return"] = (1 + df["gross_return"]) * (1 - COST) - 1
df["win_gross"] = df["gross_return"] > 0
df["win_net"] = df["net_return"] > 0
df["month"] = df["entry_time"].dt.strftime("%Y-%m")

def summarize(group):
    r = group["gross_return"]
    wins = r[r > 0]
    losses = r[r < 0]
    net = group["net_return"]
    return pd.Series({
        "trades": len(group),
        "win_rate_gross_pct": group["win_gross"].mean() * 100,
        "win_rate_after_cost_pct": group["win_net"].mean() * 100,
        "avg_gross_trade_pct": r.mean() * 100,
        "avg_net_trade_pct": net.mean() * 100,
        "avg_winning_trade_pct": wins.mean() * 100 if len(wins) else np.nan,
        "avg_losing_trade_pct": losses.mean() * 100 if len(losses) else np.nan,
        "gross_profit_factor": (
            wins.sum() / abs(losses.sum()) if len(losses) and abs(losses.sum()) else np.nan
        ),
        "gross_return_sum_pct": r.sum() * 100,
        "net_compounded_return_pct": (np.prod(1 + net) - 1) * 100
    })

print("\n=== OVERALL TRADE DIAGNOSTICS ===")
print("Trades:", len(df))
print("Entry period:", df["entry_time"].min(), "to", df["entry_time"].max())
print("Mean holding hours:",
      ((df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600).mean().round(2))

overall = summarize(df)
print(overall.to_string(float_format=lambda x: f"{x:.3f}"))

print("\n=== LONG VS SHORT ===")
side = df.groupby("side", sort=True).apply(summarize, include_groups=False)
print(side.to_string(float_format=lambda x: f"{x:.3f}"))
side.to_csv(SIDE_OUT)

print("\n=== MONTHLY PERFORMANCE ===")
monthly_rows = []
equity = 100_000.0

for month, group in df.groupby("month", sort=True):
    group = group.sort_values("entry_time")
    start_equity = equity
    trade_returns = group["net_return"].to_numpy()
    for ret in trade_returns:
        equity *= (1 + ret)

    monthly_rows.append({
        "month": month,
        "trades": len(group),
        "long_trades": int((group["side"] == "LONG").sum()),
        "short_trades": int((group["side"] == "SHORT").sum()),
        "gross_win_rate_pct": group["win_gross"].mean() * 100,
        "net_month_return_pct": (equity / start_equity - 1) * 100,
        "equity_after_month": equity
    })

monthly = pd.DataFrame(monthly_rows)
print(monthly.to_string(index=False, formatters={
    "gross_win_rate_pct": "{:.1f}%".format,
    "net_month_return_pct": "{:.2f}%".format,
    "equity_after_month": "{:,.2f}".format
}))
monthly.to_csv(MONTHLY_OUT, index=False)

print("\n=== CONCENTRATION CHECK ===")
df["net_pnl_per_100k_start"] = df["net_return"] * 100_000
print("Best 5 trades' combined net P&L per 100k starting equity:",
      f"{df.nlargest(5, 'net_pnl_per_100k_start')['net_pnl_per_100k_start'].sum():,.2f}")
print("Worst 5 trades' combined net P&L per 100k starting equity:",
      f"{df.nsmallest(5, 'net_pnl_per_100k_start')['net_pnl_per_100k_start'].sum():,.2f}")

# Compare contribution of top winners with all positive net trade returns.
positive = df.loc[df["net_return"] > 0, "net_return"]
if len(positive):
    print("Share of positive trade-return sum from top 5 winners:",
          f"{positive.nlargest(5).sum() / positive.sum() * 100:.1f}%")

print("\nSaved:", MONTHLY_OUT.name, "and", SIDE_OUT.name)
print("Costs are illustrative. Monthly results include only months with trades.")
