
import pandas as pd
import numpy as np

FILE = "gold_features_clean.csv"
STARTING_CAPITAL = 100_000
COST_PER_SIDE_BPS = 5

df = pd.read_csv(FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

df = df.sort_values("datetime").reset_index(drop=True)

df = df.dropna(
    subset=["return_4h", "close", "future_close_4h", "target_time"]
).copy()

# Keep only verified exact four-hour outcomes.
df = df[
    df["target_time"] == df["datetime"] + pd.Timedelta(hours=4)
].reset_index(drop=True)

df["future_return"] = (
    df["future_close_4h"] / df["close"] - 1
)

# Select decision times at least four hours apart.
selected = []
next_decision_time = None

for i, row in df.iterrows():
    if (
        next_decision_time is None
        or row["datetime"] >= next_decision_time
    ):
        selected.append(i)
        next_decision_time = row["target_time"]

bt = df.loc[selected].copy().reset_index(drop=True)

# Long if the trailing four-hour return is positive;
# otherwise stay in cash for that interval.
bt["signal"] = (bt["return_4h"] > 0).astype(int)

cost_rate = COST_PER_SIDE_BPS / 10_000

# Every long trade enters and exits, so charge both sides.
bt["gross_return"] = bt["signal"] * bt["future_return"]
bt["transaction_cost"] = bt["signal"] * 2 * cost_rate
bt["net_return"] = bt["gross_return"] - bt["transaction_cost"]

# Cash earns zero in this simplified simulation.
bt["equity"] = STARTING_CAPITAL * (
    1 + bt["net_return"]
).cumprod()

bt["peak"] = bt["equity"].cummax()
bt["drawdown"] = bt["equity"] / bt["peak"] - 1

# Comparable buy-and-hold benchmark over the same selected intervals.
bt["benchmark_net_return"] = (
    bt["future_return"] - 2 * cost_rate
)
bt["benchmark_equity"] = STARTING_CAPITAL * (
    1 + bt["benchmark_net_return"]
).cumprod()

trades = bt[bt["signal"] == 1]

def total_return(series):
    return (1 + series).prod() - 1

print("\n--- NON-OVERLAPPING GOLD BACKTEST ---")
print("Selected four-hour intervals:", len(bt))
print("Long trades:", len(trades))
print("Long exposure:", f"{bt['signal'].mean():.2%}")
print("Assumed cost per side:", COST_PER_SIDE_BPS, "bps")

print("\nStrategy:")
print("Gross return:", f"{total_return(bt['gross_return']):.2%}")
print("Net return:", f"{total_return(bt['net_return']):.2%}")
print("Maximum drawdown:", f"{bt['drawdown'].min():.2%}")

if len(trades):
    print(
        "Win rate before costs:",
        f"{(trades['future_return'] > 0).mean():.2%}"
    )

print("\nBuy-and-hold benchmark:")
print(
    "Net return:",
    f"{total_return(bt['benchmark_net_return']):.2%}"
)

bt.to_csv("gold_backtest_nonoverlap.csv", index=False)
print("\nSaved: gold_backtest_nonoverlap.csv")
print(
    "\nResearch assumptions only: no actual contract sizing, "
    "spread data, slippage calibration, or financing costs."
)
