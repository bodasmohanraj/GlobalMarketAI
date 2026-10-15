
import pandas as pd
import numpy as np

# Research assumptions — not live market quotes.
DATA_FILE = "gold_features_clean.csv"
TRANSACTION_COST_BPS = 5
STARTING_CAPITAL = 100_000
POSITION_SIZE = 1.0  # Fully exposed when a signal is active.

df = pd.read_csv(DATA_FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

df = df.sort_values("datetime").reset_index(drop=True)
df = df.dropna(
    subset=["return_4h", "target_up", "target_time"]
).copy()

# Signal at time t: positive trailing 4h return = long;
# otherwise remain in cash. No short positions.
df["signal"] = (df["return_4h"] > 0).astype(int)

# The target price is exactly four elapsed hours ahead.
# Only keep observations with a valid four-hour target.
df = df.dropna(subset=["future_close_4h", "close"]).copy()
df = df[df["target_time"] == df["datetime"] + pd.Timedelta(hours=4)]
df = df.reset_index(drop=True)

# Forward price return over the next four hours.
df["future_return"] = (
    df["future_close_4h"] / df["close"] - 1
)

# Apply the signal's return, then charge an estimated cost
# for entering/exiting positions. These costs are assumptions.
df["previous_signal"] = df["signal"].shift(1).fillna(0)
df["turnover"] = (
    df["signal"] - df["previous_signal"]
).abs()

cost_rate = TRANSACTION_COST_BPS / 10_000

df["gross_return"] = df["signal"] * df["future_return"]
df["net_return"] = (
    df["gross_return"] - df["turnover"] * cost_rate
)

# Avoid treating overlapping four-hour forecasts as separate
# non-overlapping trades: this initial version is a diagnostic
# signal test, not a portfolio execution simulator.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["net_return"]).reset_index(drop=True)

df["equity"] = STARTING_CAPITAL * (1 + df["net_return"]).cumprod()
df["peak"] = df["equity"].cummax()
df["drawdown"] = df["equity"] / df["peak"] - 1

total_return = df["equity"].iloc[-1] / STARTING_CAPITAL - 1
max_drawdown = df["drawdown"].min()
mean_return = df["net_return"].mean()
volatility = df["net_return"].std()

active = df[df["signal"] == 1]
win_rate = (
    (active["future_return"] > 0).mean()
    if len(active) else np.nan
)

print("\n--- GOLD MOMENTUM BACKTEST (DIAGNOSTIC) ---")
print("Observations:", len(df))
print("Assumed cost per turnover:", TRANSACTION_COST_BPS, "bps")
print("Long exposure:", f"{df['signal'].mean():.2%}")
print("Long signals:", len(active))
print("Gross mean return per observation:", f"{df['gross_return'].mean():.5%}")
print("Net mean return per observation:", f"{mean_return:.5%}")
print("Long-signal win rate:", f"{win_rate:.2%}")
print("Compounded return (diagnostic):", f"{total_return:.2%}")
print("Maximum drawdown (diagnostic):", f"{max_drawdown:.2%}")
print("Return standard deviation:", f"{volatility:.5%}")

df.to_csv("gold_backtest_diagnostic.csv", index=False)
print("\nSaved: gold_backtest_diagnostic.csv")
print(
    "\nWARNING: Four-hour outcomes overlap, so compounded equity "
    "is NOT a valid portfolio performance estimate."
)
print(
    "This diagnostic omits actual contract sizing, spreads, slippage, "
    "financing, and execution constraints."
)
