from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "sr_holdout_trades.csv"

if not SOURCE.exists():
    raise FileNotFoundError("Run backtest_sr_holdout.py first.")

df = pd.read_csv(SOURCE)
df = df[df["strategy"] == "breakout_20"].copy()

if df.empty:
    raise ValueError("No breakout_20 trades found.")

df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

returns = df["strategy_return_pct"].astype(float).to_numpy() / 100
n = len(returns)

if n < 2:
    raise ValueError("Not enough trades for statistical analysis.")

mean_return = returns.mean()
std_return = returns.std(ddof=1)
standard_error = std_return / np.sqrt(n)

# Approximate 95% confidence interval for the mean trade return.
ci_low = mean_return - 1.96 * standard_error
ci_high = mean_return + 1.96 * standard_error

# One-sample t statistic; normal approximation for the displayed p-value
# is intentionally avoided, so no unsupported p-value is reported.
t_stat = mean_return / standard_error if standard_error > 0 else np.nan

print("\n=== BREAKOUT_20 STATISTICAL REVIEW ===")
print("Trades:", n)
print("Long trades:", int((df["direction"] == "LONG").sum()))
print("Short trades:", int((df["direction"] == "SHORT").sum()))
print("First entry:", df["entry_time"].min())
print("Last exit:", df["exit_time"].max())
print("Win rate:", f"{(returns > 0).mean() * 100:.2f}%")
print("Mean return per trade before costs:", f"{mean_return * 100:.4f}%")
print("Median return per trade:", f"{np.median(returns) * 100:.4f}%")
print("Trade-return standard deviation:", f"{std_return * 100:.4f}%")
print("Approx. 95% CI for mean trade return:",
      f"[{ci_low * 100:.4f}%, {ci_high * 100:.4f}%]")
print("One-sample t statistic:", f"{t_stat:.4f}")
print("\nImportant: trades may be dependent across time.")
print("This interval is approximate and does not establish a profitable edge.")
