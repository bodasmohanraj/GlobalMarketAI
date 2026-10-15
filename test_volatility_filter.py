
import pandas as pd
import numpy as np

FILE = "gold_backtest_nonoverlap.csv"
COST_BPS_PER_SIDE = 5

df = pd.read_csv(FILE)

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

required = ["volatility_20", "signal", "future_return"]
df = df.dropna(subset=required).reset_index(drop=True)

# Chronological 70/30 split.
split = int(len(df) * 0.70)

train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# Calculate the volatility threshold using training data only.
threshold = train["volatility_20"].median()

# Original momentum strategy.
test["original_signal"] = test["signal"].astype(int)

# Filtered strategy: go long only when volatility is above
# the training-period median and the momentum signal is positive.
test["filtered_signal"] = (
    (test["signal"] == 1)
    & (test["volatility_20"] > threshold)
).astype(int)

# Assumed entry + exit cost for every long trade.
cost = 2 * COST_BPS_PER_SIDE / 10_000


def evaluate(name, signal, future_return):
    signal = np.asarray(signal, dtype=float)
    future_return = np.asarray(future_return, dtype=float)

    gross_returns = signal * future_return
    net_returns = gross_returns - signal * cost

    # Hypothetical compounded equity for the test observations.
    equity = np.cumprod(1 + net_returns)

    total_return = equity[-1] - 1

    # Calculate drawdown using NumPy arrays.
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    max_drawdown = drawdown.min()

    active_returns = future_return[signal == 1]

    if len(active_returns) > 0:
        win_rate = (active_returns > 0).mean()
    else:
        win_rate = np.nan

    print(f"\n{name}")
    print("Trades:", int(signal.sum()))
    print("Exposure:", f"{signal.mean():.2%}")
    print(
        "Mean gross return per interval:",
        f"{gross_returns.mean():.5%}"
    )
    print(
        "Mean net return per interval:",
        f"{net_returns.mean():.5%}"
    )
    print("Win rate before costs:", f"{win_rate:.2%}")
    print("Compounded test return:", f"{total_return:.2%}")
    print("Maximum drawdown:", f"{max_drawdown:.2%}")


print("\n--- CHRONOLOGICAL VOLATILITY FILTER TEST ---")
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training-only volatility threshold:", threshold)
print("Assumed cost:", COST_BPS_PER_SIDE, "bps per side")

evaluate(
    "Original momentum strategy",
    test["original_signal"],
    test["future_return"],
)

evaluate(
    "High-volatility-only strategy",
    test["filtered_signal"],
    test["future_return"],
)

test.to_csv("volatility_filter_test_results.csv", index=False)

print("\nSaved: volatility_filter_test_results.csv")
print(
    "\nIMPORTANT: This is still a simplified research test. "
    "It assumes execution at recorded prices, uses fixed costs, "
    "and does not model real spreads, slippage, or contract sizing."
)
