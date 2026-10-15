import pandas as pd
import numpy as np

COST = 0.0006

# This file already contains predictions and corrected trade prices.
data = pd.read_csv("backtest_1h_trades_corrected.csv")
data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
data = data.sort_values("datetime").reset_index(drop=True)

required = ["decision_model", "entry_price", "exit_price", "fold"]
missing = [c for c in required if c not in data.columns]
if missing:
    raise ValueError(f"Missing columns in trades file: {missing}")

data = data.dropna(subset=required).copy()

# Build exact 1-hour momentum from historical 5-minute prices.
prices = pd.read_csv("gold_5m_data.csv")
prices.columns = [c.strip().lower() for c in prices.columns]
prices["datetime"] = pd.to_datetime(prices["datetime"], utc=True)
prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
prices = (
    prices.dropna(subset=["datetime", "close"])
    .drop_duplicates("datetime")
    .sort_values("datetime")
)

past = prices[["datetime", "close"]].copy()
past["datetime"] += pd.Timedelta(minutes=60)
past = past.rename(columns={"close": "close_1h_ago"})

data = data.merge(
    past, on="datetime", how="left", validate="many_to_one"
)

data["trend_pct"] = (
    data["close"] / data["close_1h_ago"] - 1
) * 100

valid = data.dropna(subset=["trend_pct"]).copy()

entry = valid["entry_price"].to_numpy(dtype=float)
exit_price = valid["exit_price"].to_numpy(dtype=float)
price_return = (exit_price - entry) / entry

strategies = {
    "Model": np.where(valid["decision_model"].to_numpy() == 1, 1, -1),
    "Momentum": np.where(valid["trend_pct"].to_numpy() > 0, 1, -1),
    "Always short": -np.ones(len(valid)),
    "Always long": np.ones(len(valid)),
}

rows = []

for name, side in strategies.items():
    gross = side * price_return
    net = gross - COST
    equity = np.cumprod(1 + net)
    drawdown = equity / np.maximum.accumulate(equity) - 1

    rows.append({
        "strategy": name,
        "trades": len(net),
        "mean_gross_pct": gross.mean() * 100,
        "mean_net_pct": net.mean() * 100,
        "win_rate_pct": (net > 0).mean() * 100,
        "compounded_return_pct": (equity[-1] - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
    })

rows.append({
    "strategy": "No trade",
    "trades": 0,
    "mean_gross_pct": 0.0,
    "mean_net_pct": 0.0,
    "win_rate_pct": np.nan,
    "compounded_return_pct": 0.0,
    "max_drawdown_pct": 0.0,
})

report = pd.DataFrame(rows)
report.to_csv("momentum_comparison_1h.csv", index=False)

print("\n1-HOUR TREND AND MOMENTUM COMPARISON")
print("Valid common timestamps:", len(valid))
print("Missing exact 1-hour historical prices:",
      int(data["trend_pct"].isna().sum()))
print(report.round(4).to_string(index=False))
print("\nSaved: momentum_comparison_1h.csv")
print("Research only; assumed round-trip costs are 6 basis points.")
