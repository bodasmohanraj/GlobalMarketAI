
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

PRICE_FILE = BASE / "gold_data.csv"
TRADES_FILE = BASE / "risk_managed_v2_trades.csv"

DETAILS_FILE = BASE / "volatility_walkforward_details.csv"
SUMMARY_FILE = BASE / "volatility_walkforward_summary.csv"

ATR_PERIOD = 14


def load_prices():
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Missing {PRICE_FILE.name}")

    df = pd.read_csv(PRICE_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = {"datetime", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Simple rolling ATR, consistent with a common ATR implementation.
    df["atr"] = true_range.rolling(
        ATR_PERIOD, min_periods=ATR_PERIOD
    ).mean()

    df["atr_pct"] = df["atr"] / df["close"] * 100

    return df


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan

    return gains / losses


def stats(group):
    pnl = group["net_pnl"].to_numpy(dtype=float)

    return {
        "trades": len(group),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": (
            float((pnl > 0).mean() * 100) if len(pnl) else np.nan
        ),
        "net_pnl": float(pnl.sum()),
        "mean_net_pnl": float(pnl.mean()) if len(pnl) else np.nan,
        "profit_factor": float(profit_factor(pnl)),
    }


def main():
    prices = load_prices()

    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Missing {TRADES_FILE.name}")

    trades = pd.read_csv(TRADES_FILE)
    trades.columns = trades.columns.str.strip().str.lower()

    required = {
        "fold", "entry_time", "entry_price",
        "atr_at_signal", "net_pnl",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Missing trade columns: {sorted(missing)}")

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    )

    for col in ["entry_price", "atr_at_signal", "net_pnl"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades = trades.dropna(
        subset=[
            "fold", "entry_time", "entry_price",
            "atr_at_signal", "net_pnl",
        ]
    ).copy()

    # First half of the chronological price history is training data.
    # The remaining half is not used to set the thresholds.
    split_index = len(prices) // 2
    training_prices = prices.iloc[:split_index]

    training_volatility = training_prices["atr_pct"].dropna()

    if len(training_volatility) < 30:
        raise ValueError("Not enough valid training ATR observations.")

    low_cutoff = float(training_volatility.quantile(1 / 3))
    high_cutoff = float(training_volatility.quantile(2 / 3))
    training_end = training_prices["datetime"].max()

    print("\n=== TRAINING-ONLY THRESHOLDS ===")
    print(f"Training data ends: {training_end}")
    print(f"Low/medium boundary: {low_cutoff:.5f}%")
    print(f"Medium/high boundary: {high_cutoff:.5f}%")

    # Classify trades using thresholds from the training period only.
    trades["atr_pct"] = (
        trades["atr_at_signal"] / trades["entry_price"] * 100
    )

    trades["volatility"] = np.select(
        [
            trades["atr_pct"] < low_cutoff,
            trades["atr_pct"] < high_cutoff,
        ],
        ["low", "medium"],
        default="high",
    )

    # Confirm the tested trades occur after the training period.
    early_trades = trades[trades["entry_time"] <= training_end]

    if not early_trades.empty:
        print(
            "\nWARNING: Some trades are dated within the training period."
            "\nCheck fold definitions before interpreting the results."
        )
        print("Early trades:", len(early_trades))

    rows = []

    for (fold, bucket), group in trades.groupby(
        ["fold", "volatility"], observed=True
    ):
        row = {"fold": int(fold), "volatility": bucket}
        row.update(stats(group))
        rows.append(row)

    details = pd.DataFrame(rows)

    if details.empty:
        raise ValueError("No trades available for the report.")

    details = details.sort_values(["fold", "volatility"])
    details.to_csv(DETAILS_FILE, index=False)

    summary_rows = []

    for bucket, group in trades.groupby("volatility", observed=True):
        row = {"volatility": bucket}
        row.update(stats(group))

        bucket_folds = details[details["volatility"] == bucket]
        row["positive_folds"] = int(
            (bucket_folds["net_pnl"] > 0).sum()
        )
        row["folds_with_trades"] = len(bucket_folds)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n=== VOLATILITY PERFORMANCE BY FOLD ===")
    print(details.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== OVERALL VOLATILITY SUMMARY ===")
    print(summary.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\nCreated:", DETAILS_FILE.name)
    print("Created:", SUMMARY_FILE.name)

    print(
        "\nResearch only: thresholds use the first half of price history. "
        "Verify that the ATR calculation matches the original backtest "
        "before treating this as a comparable result."
    )


if __name__ == "__main__":
    main()
