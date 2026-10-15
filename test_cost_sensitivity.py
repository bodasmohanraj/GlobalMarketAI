
import pandas as pd
import numpy as np

df = pd.read_csv("gold_backtest_nonoverlap.csv")

signal = df["signal"].astype(float)
future_return = df["future_return"].astype(float)

print("\n--- TRANSACTION COST SENSITIVITY ---")
print("Observations:", len(df))
print("Long trades:", int(signal.sum()))

results = []

for bps_per_side in [0, 2, 5, 10]:
    cost_rate = bps_per_side / 10_000

    gross_returns = signal * future_return
    net_returns = gross_returns - signal * 2 * cost_rate

    # Geometric return across the selected intervals.
    equity = (1 + net_returns).cumprod()
    total_return = equity.iloc[-1] - 1

    results.append({
        "cost_bps_per_side": bps_per_side,
        "mean_net_return": net_returns.mean(),
        "total_compounded_return": total_return,
    })

result = pd.DataFrame(results)

print("\nCost sensitivity results:")
print(result.to_string(
    index=False,
    formatters={
        "mean_net_return": "{:.5%}".format,
        "total_compounded_return": "{:.2%}".format,
    }
))

print(
    "\nThese are hypothetical results under fixed cost assumptions, "
    "not verified live-trading performance."
)
