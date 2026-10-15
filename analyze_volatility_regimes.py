
import pandas as pd

FILE = "gold_backtest_nonoverlap.csv"

df = pd.read_csv(FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

df = df.dropna(
    subset=["volatility_20", "signal", "future_return"]
).copy()

# Split observations into low- and high-volatility groups.
median_vol = df["volatility_20"].median()

df["volatility_regime"] = df["volatility_20"].apply(
    lambda x: "Low volatility"
    if x <= median_vol
    else "High volatility"
)

# Apply the same assumed cost: 5 bps per side for long trades.
df["gross_strategy_return"] = (
    df["signal"] * df["future_return"]
)

cost_rate = 5 / 10_000

df["net_strategy_return"] = (
    df["gross_strategy_return"]
    - df["signal"] * 2 * cost_rate
)

def average_long_return(group):
    long_returns = group.loc[
        group["signal"] == 1, "future_return"
    ]
    return long_returns.mean()

summary = df.groupby("volatility_regime").agg(
    observations=("future_return", "size"),
    long_trades=("signal", "sum"),
    average_volatility=("volatility_20", "mean"),
    average_gross_return=("gross_strategy_return", "mean"),
    average_net_return=("net_strategy_return", "mean"),
)

long_means = df.groupby("volatility_regime").apply(
    average_long_return, include_groups=False
)
summary["average_long_return"] = long_means

print("\n--- VOLATILITY REGIME ANALYSIS ---")
print("Median volatility threshold:", median_vol)

print(summary.to_string(
    formatters={
        "average_volatility": "{:.6f}".format,
        "average_gross_return": "{:.5%}".format,
        "average_net_return": "{:.5%}".format,
        "average_long_return": "{:.5%}".format,
    }
))

print(
    "\nDescriptive research only: the median uses the full sample. "
    "These results do not validate an out-of-sample trading strategy."
)
