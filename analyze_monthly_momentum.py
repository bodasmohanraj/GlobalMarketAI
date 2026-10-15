
import pandas as pd

df = pd.read_csv("gold_backtest_nonoverlap.csv")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["signal"] = pd.to_numeric(df["signal"], errors="raise")
df["future_return"] = pd.to_numeric(
    df["future_return"], errors="raise"
)

# Only analyze intervals where the strategy goes long.
trades = df[df["signal"] == 1].copy()
trades["month"] = trades["datetime"].dt.to_period("M")

monthly = trades.groupby("month").agg(
    trades=("future_return", "size"),
    mean_gross_return=("future_return", "mean"),
    win_rate=("future_return", lambda x: (x > 0).mean()),
)

monthly["mean_net_return_2bps_per_side"] = (
    monthly["mean_gross_return"] - 2 * 2 / 10_000
)

print("\n--- MONTHLY MOMENTUM ANALYSIS ---")
print(monthly.tail(24).to_string(
    formatters={
        "mean_gross_return": "{:.5%}".format,
        "win_rate": "{:.2%}".format,
        "mean_net_return_2bps_per_side": "{:.5%}".format,
    }
))

print("\nMonths analyzed:", len(monthly))
print(
    "Months with positive gross mean return:",
    int((monthly["mean_gross_return"] > 0).sum())
)
print(
    "Months with positive mean return after assumed costs:",
    int((monthly["mean_net_return_2bps_per_side"] > 0).sum())
)

monthly.to_csv("monthly_momentum_results.csv")
print("\nSaved: monthly_momentum_results.csv")
