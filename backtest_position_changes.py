import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = "gold_backtest_nonoverlap.csv"

TEST_RATIO = 0.30
COST_BPS_PER_SIDE = 5

# 5 basis points = 0.05% per side
COST_RATE = COST_BPS_PER_SIDE / 10000


# -----------------------------
# LOAD AND PREPARE DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

df = df.sort_values("datetime").reset_index(drop=True)

required_columns = [
    "datetime",
    "target_time",
    "future_return",
    "volatility_20",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_columns).reset_index(drop=True)

# Keep only exact four-hour holding intervals.
duration = df["target_time"] - df["datetime"]
df = df[duration == pd.Timedelta(hours=4)].copy()
df = df.reset_index(drop=True)

if len(df) < 20:
    raise ValueError("Not enough valid rows to run the backtest.")

# Chronological 70% training / 30% testing split.
split_index = int(len(df) * (1 - TEST_RATIO))

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# Calculate the volatility threshold using training data only.
volatility_threshold = train["volatility_20"].median()

# Long/cash signals.
# Original strategy: long when the previous 4-hour return is positive.
test["original_signal"] = (
    pd.to_numeric(test["return_4h"], errors="coerce") > 0
).astype(int)

# Filtered strategy: only take the original long signal in high volatility.
test["filtered_signal"] = (
    (test["original_signal"] == 1)
    & (test["volatility_20"] > volatility_threshold)
).astype(int)

test = test.dropna(
    subset=["future_return", "volatility_20"]
).reset_index(drop=True)


# -----------------------------
# POSITION-CHANGE BACKTEST
# -----------------------------
def run_backtest(data, signal_column):
    result = data.copy()

    signals = result[signal_column].astype(int).to_numpy()
    returns = pd.to_numeric(
        result["future_return"], errors="coerce"
    ).fillna(0).to_numpy()

    times = result["datetime"].to_numpy()
    targets = result["target_time"].to_numpy()

    turnover = []
    previous_signal = 0

    for i, signal in enumerate(signals):
        if i == 0:
            # Start flat; entering a long position costs one side.
            trade_amount = abs(signal - previous_signal)
        else:
            # If intervals are not contiguous, assume the portfolio
            # was flat during the gap.
            previous_target = pd.Timestamp(targets[i - 1])
            current_time = pd.Timestamp(times[i])

            if current_time > previous_target:
                # Exit the previous position at its target time,
                # then enter the new position if required.
                trade_amount = previous_signal + signal
            else:
                # Continuous intervals: pay only for position changes.
                trade_amount = abs(signal - previous_signal)

        turnover.append(trade_amount)
        previous_signal = signal

    result["position"] = signals
    result["turnover"] = turnover

    # Long/cash strategy return for each 4-hour interval.
    result["gross_return"] = result["position"] * returns

    # Charge transaction costs only for actual position changes.
    result["transaction_cost"] = result["turnover"] * COST_RATE
    result["net_return"] = (
        result["gross_return"] - result["transaction_cost"]
    )

    result["equity"] = (1 + result["net_return"]).cumprod()
    result["peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["peak"] - 1

    total_return = result["equity"].iloc[-1] - 1
    gross_equity = (1 + result["gross_return"]).cumprod()
    gross_total_return = gross_equity.iloc[-1] - 1

    active = result[result["position"] == 1]

    metrics = {
        "test_intervals": len(result),
        "long_intervals": int(result["position"].sum()),
        "exposure_pct": result["position"].mean() * 100,
        "position_changes_and_reentries": int(result["turnover"].sum()),
        "gross_total_return_pct": gross_total_return * 100,
        "net_total_return_pct": total_return * 100,
        "mean_gross_return_per_interval_pct":
            result["gross_return"].mean() * 100,
        "mean_net_return_per_interval_pct":
            result["net_return"].mean() * 100,
        "long_interval_win_rate_pct":
            (active["gross_return"] > 0).mean() * 100
            if len(active) else 0,
        "max_drawdown_pct": result["drawdown"].min() * 100,
        "total_transaction_cost_pct":
            result["transaction_cost"].sum() * 100,
    }

    return result, metrics


# -----------------------------
# RUN BOTH STRATEGIES
# -----------------------------
original_results, original_metrics = run_backtest(
    test, "original_signal"
)

filtered_results, filtered_metrics = run_backtest(
    test, "filtered_signal"
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\n========== BACKTEST SETTINGS ==========")
print(f"Total valid intervals: {len(df)}")
print(f"Training intervals: {len(train)}")
print(f"Testing intervals: {len(test)}")
print(f"Volatility threshold: {volatility_threshold:.8f}")
print(f"Transaction cost: {COST_BPS_PER_SIDE} bps per side")

print("\n========== ORIGINAL MOMENTUM STRATEGY ==========")
for key, value in original_metrics.items():
    print(f"{key}: {value:.4f}")

print("\n========== HIGH-VOLATILITY FILTER STRATEGY ==========")
for key, value in filtered_metrics.items():
    print(f"{key}: {value:.4f}")

# Save results for further analysis.
original_results.to_csv(
    "position_changes_original_results.csv", index=False
)

filtered_results.to_csv(
    "position_changes_filtered_results.csv", index=False
)

comparison = pd.DataFrame({
    "metric": list(original_metrics.keys()),
    "original_momentum": list(original_metrics.values()),
    "high_volatility_filter": list(filtered_metrics.values()),
})

comparison.to_csv("position_changes_comparison.csv", index=False)

print("\nSaved files:")
print("- position_changes_original_results.csv")
print("- position_changes_filtered_results.csv")
print("- position_changes_comparison.csv")
