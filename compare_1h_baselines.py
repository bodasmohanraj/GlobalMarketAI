import pandas as pd
import numpy as np

pred = pd.read_csv("validation_audit_oos_predictions_v2.csv")
trades = pd.read_csv("backtest_1h_trades_corrected.csv")

pred["datetime"] = pd.to_datetime(pred["datetime"], utc=True)
trades["datetime"] = pd.to_datetime(trades["datetime"], utc=True)

pred = pred[pred["horizon"] == "1hour"].copy()
pred = pred.drop_duplicates(["fold", "datetime"])

# Use the same timestamps selected by the model backtest.
selected = trades[["datetime", "fold"]].drop_duplicates()
data = selected.merge(
    pred[["datetime", "fold", "close", "future_close",
          "actual_change", "decision_model"]],
    on=["datetime", "fold"],
    how="left",
    validate="one_to_one"
)

# Match the corrected backtest's actual entry/exit prices.
data = data.merge(
    trades[["datetime", "fold", "entry_price", "exit_price"]],
    on=["datetime", "fold"],
    how="left",
    validate="one_to_one"
)

data = data.dropna(
    subset=["entry_price", "exit_price", "decision_model"]
)

cost = 0.0006
price_change = (
    data["exit_price"] - data["entry_price"]
) / data["entry_price"]

strategies = {
    "Model": np.where(data["decision_model"] == 1, 1, -1),
    "Always long": np.ones(len(data)),
    "Always short": -np.ones(len(data)),
}

results = []

for name, side in strategies.items():
    gross = side * price_change
    net = gross - cost
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1

    results.append({
        "strategy": name,
        "trades": len(net),
        "gross_mean_pct": gross.mean() * 100,
        "net_mean_pct": net.mean() * 100,
        "win_rate_pct": (net > 0).mean() * 100,
        "compounded_return_pct": (equity.iloc[-1] - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
    })

results.append({
    "strategy": "No trade",
    "trades": 0,
    "gross_mean_pct": 0,
    "net_mean_pct": 0,
    "win_rate_pct": np.nan,
    "compounded_return_pct": 0,
    "max_drawdown_pct": 0,
})

report = pd.DataFrame(results)
report.to_csv("backtest_1h_baseline_comparison.csv", index=False)

print("\nBASELINE COMPARISON")
print(report.round(4).to_string(index=False))
print("\nSaved: backtest_1h_baseline_comparison.csv")
print("Note: All trading strategies use the same selected timestamps and assumed costs.")
