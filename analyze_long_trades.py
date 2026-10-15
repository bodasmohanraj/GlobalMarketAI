
import pandas as pd
import numpy as np

df = pd.read_csv("gold_backtest_nonoverlap.csv")

trades = df[df["signal"] == 1].copy()
returns = trades["future_return"].astype(float)

print("\n--- LONG TRADE ANALYSIS ---")
print("Long trades:", len(trades))
print("Mean gross return:", f"{returns.mean():.5%}")
print("Median gross return:", f"{returns.median():.5%}")
print("Win rate:", f"{(returns > 0).mean():.2%}")
print("Average winning return:", f"{returns[returns > 0].mean():.5%}")
print("Average losing return:", f"{returns[returns < 0].mean():.5%}")

print("\n--- COST SCENARIOS ---")
for bps in [0, 1, 2, 5, 10]:
    cost = 2 * bps / 10_000
    net = returns - cost

    print(
        f"{bps} bps per side | "
        f"Mean net return: {net.mean():.5%} | "
        f"Trades net positive: {(net > 0).mean():.2%}"
    )

print("\n--- RETURN QUANTILES ---")
print(returns.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_string())
