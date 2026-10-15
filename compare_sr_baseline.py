
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gold_support_resistance.csv"
TRADES_FILE = BASE_DIR / "gold_sr_validation_trades.csv"
OUTPUT_FILE = BASE_DIR / "gold_sr_baseline_comparison.csv"

HOLDOUT_FRACTION = 0.20
HOLD_BARS = 4
MAX_ELAPSED_HOURS = 6


def main():
    df = pd.read_csv(DATA_FILE)
    trades = pd.read_csv(TRADES_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    trades["signal_time"] = pd.to_datetime(
        trades["signal_time"], utc=True
    )
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True
    )
    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"], utc=True
    )

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    split_idx = int(len(df) * (1 - HOLDOUT_FRACTION))
    holdout_start = df.loc[split_idx, "datetime"]

    df = df.loc[df["datetime"] >= holdout_start].reset_index(drop=True)

    # Map exact timestamps to actual bars.
    index_by_time = {
        timestamp: i for i, timestamp in enumerate(df["datetime"])
    }

    rows = []

    for _, trade in trades.iterrows():
        if trade["signal_time"] < holdout_start:
            continue

        signal_idx = index_by_time.get(trade["signal_time"])
        entry_idx = index_by_time.get(trade["entry_time"])
        exit_idx = index_by_time.get(trade["exit_time"])

        if signal_idx is None or entry_idx is None or exit_idx is None:
            continue

        if entry_idx != signal_idx + 1:
            continue

        if exit_idx != signal_idx + HOLD_BARS:
            continue

        elapsed = (
            df.loc[exit_idx, "datetime"] -
            df.loc[entry_idx, "datetime"]
        ).total_seconds() / 3600

        if elapsed <= 0 or elapsed > MAX_ELAPSED_HOURS:
            continue

        entry = float(df.loc[entry_idx, "open"])
        exit_price = float(df.loc[exit_idx, "close"])

        # Same-direction baseline: hold the same direction over the
        # same entry and exit window, without applying signal selection.
        direction = trade["direction"]
        baseline_return = (
            (exit_price - entry) / entry * 100
            if direction == "LONG"
            else (entry - exit_price) / entry * 100
        )

        strategy_return = float(trade["gross_return_pct"])

        rows.append({
            "direction": direction,
            "signal_time": trade["signal_time"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
            "strategy_gross_return_pct": strategy_return,
            "matched_window_return_pct": baseline_return,
            "difference_pct_points": strategy_return - baseline_return,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No matched trades found. Check timestamp formats and files."
        )

    result.to_csv(OUTPUT_FILE, index=False)

    print("=== MATCHED WINDOW COMPARISON ===")
    print("Matched records:", len(result))
    print("\nBy direction:")
    print(
        result.groupby("direction")[
            [
                "strategy_gross_return_pct",
                "matched_window_return_pct",
                "difference_pct_points",
            ]
        ].mean().round(5).to_string()
    )

    print("\nOverall averages:")
    print(
        result[
            [
                "strategy_gross_return_pct",
                "matched_window_return_pct",
                "difference_pct_points",
            ]
        ].mean().round(5).to_string()
    )

    print("\nSaved:", OUTPUT_FILE.name)
    print(
        "Note: the matched-window return uses the same entry and exit "
        "prices, so it is an execution consistency check, not an "
        "independent benchmark of signal quality."
    )


if __name__ == "__main__":
    main()
