import pandas as pd
import numpy as np

COST = 0.0006

d = pd.read_csv("backtest_1h_trades_corrected.csv")
d["datetime"] = pd.to_datetime(d["datetime"], utc=True)

p = pd.read_csv("gold_5m_data.csv")
p.columns = [c.strip().lower() for c in p.columns]
p["datetime"] = pd.to_datetime(p["datetime"], utc=True)
p["close"] = pd.to_numeric(p["close"], errors="coerce")
p = p.dropna(subset=["datetime", "close"]).drop_duplicates("datetime")
p = p.sort_values("datetime")

past = p[["datetime", "close"]].copy()
past["datetime"] += pd.Timedelta(minutes=60)
past = past.rename(columns={"close": "close_1h_ago"})

d = d.merge(past, on="datetime", how="left", validate="many_to_one")
d["momentum_pct"] = (d["close"] / d["close_1h_ago"] - 1) * 100
d = d.dropna(subset=["momentum_pct"]).sort_values("datetime").copy()

# Exploratory thresholds; do not select a threshold from this test
# and then claim the same test is independent validation.
thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
results = []

for threshold in thresholds:
    side = np.where(
        d["momentum_pct"] > threshold, 1,
        np.where(d["momentum_pct"] < -threshold, -1, 0)
    )

    active = side != 0
    if active.sum() == 0:
        continue

    gross = side[active] * (
        d.loc[active, "exit_price"].to_numpy()
        - d.loc[active, "entry_price"].to_numpy()
    ) / d.loc[active, "entry_price"].to_numpy()

    net = gross - COST
    equity = np.cumprod(1 + net)
    dd = equity / np.maximum.accumulate(equity) - 1

    results.append({
        "threshold_pct": threshold,
        "trades": int(active.sum()),
        "trade_coverage_pct": active.mean() * 100,
        "gross_mean_pct": gross.mean() * 100,
        "net_mean_pct": net.mean() * 100,
        "win_rate_pct": (net > 0).mean() * 100,
        "compounded_return_pct": (equity[-1] - 1) * 100,
        "max_drawdown_pct": dd.min() * 100
    })

report = pd.DataFrame(results)
report.to_csv("momentum_threshold_experiment.csv", index=False)

print("\nMOMENTUM THRESHOLD EXPERIMENT")
print(report.round(4).to_string(index=False))
print("\nSaved: momentum_threshold_experiment.csv")
print("Cost: 6 bps per completed trade, illustrative.")
print("Research only: these same-period results are not independent validation.")
