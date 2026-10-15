
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gold_support_resistance.csv"
OUTPUT_FILE = BASE_DIR / "gold_sr_validation.csv"
TRADES_FILE = BASE_DIR / "gold_sr_validation_trades.csv"

HOLDOUT_FRACTION = 0.20
HOLD_BARS = 4
MAX_ELAPSED_HOURS = 6
COSTS_PCT = [0.00, 0.05, 0.10, 0.15, 0.20]

REQUIRED_COLUMNS = [
    "datetime",
    "open",
    "close",
    "nearest_support",
    "nearest_resistance",
    "tested_support_level",
    "tested_resistance_level",
    "possible_resistance_breakout",
    "possible_support_breakdown",
]


def as_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            "Run support_resistance.py first."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in [
        "open", "close",
        "tested_support_level", "tested_resistance_level",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "close"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    # Verify that stored flags agree with the recorded prior-known levels.
    prev_close = df["close"].shift(1)
    tested_r = df["tested_resistance_level"]
    tested_s = df["tested_support_level"]

    expected_long = (
        tested_r.notna()
        & prev_close.notna()
        & (prev_close <= tested_r)
        & (df["close"] > tested_r)
    )

    expected_short = (
        tested_s.notna()
        & prev_close.notna()
        & (prev_close >= tested_s)
        & (df["close"] < tested_s)
    )

    stored_long = as_bool(df["possible_resistance_breakout"])
    stored_short = as_bool(df["possible_support_breakdown"])

    long_mismatches = int((stored_long != expected_long).sum())
    short_mismatches = int((stored_short != expected_short).sum())

    print("=== SIGNAL AUDIT ===")
    print("Long flag mismatches:", long_mismatches)
    print("Short flag mismatches:", short_mismatches)

    if long_mismatches or short_mismatches:
        raise ValueError(
            "Stored flags do not match the recorded tested levels. "
            "Inspect support_resistance.py before backtesting."
        )

    # Chronological holdout; no random shuffling.
    split_idx = int(len(df) * (1 - HOLDOUT_FRACTION))
    if split_idx <= 0 or split_idx >= len(df) - HOLD_BARS:
        raise ValueError("Not enough data for this holdout.")

    print("\n=== HOLDOUT PERIOD ===")
    print("Rows:", len(df))
    print("History ends:", df.loc[split_idx - 1, "datetime"])
    print("Holdout starts:", df.loc[split_idx, "datetime"])
    print("Holdout rows:", len(df) - split_idx)

    print(
        "\nWARNING: This is a preliminary historical test. "
        "The rules were developed using this dataset, so the holdout "
        "is not fully independent. Assumed costs are not broker quotes."
    )

    trades = []

    for direction, flags, level_col in [
        ("LONG", stored_long, "tested_resistance_level"),
        ("SHORT", stored_short, "tested_support_level"),
    ]:
        events = np.flatnonzero(flags.to_numpy())
        events = events[events >= split_idx]

        last_exit_idx = -1

        for i in events:
            entry_idx = i + 1
            exit_idx = i + HOLD_BARS

            if exit_idx >= len(df):
                continue

            # Prevent overlapping trades within each direction.
            if entry_idx <= last_exit_idx:
                continue

            entry_time = df.loc[entry_idx, "datetime"]
            exit_time = df.loc[exit_idx, "datetime"]

            elapsed = (
                exit_time - entry_time
            ).total_seconds() / 3600

            # Reject trades spanning excessive data gaps.
            if elapsed <= 0 or elapsed > MAX_ELAPSED_HOURS:
                continue

            entry = float(df.loc[entry_idx, "open"])
            exit_price = float(df.loc[exit_idx, "close"])
            level = df.loc[i, level_col]

            if (
                not np.isfinite(entry)
                or not np.isfinite(exit_price)
                or entry <= 0
            ):
                continue

            gross_pct = (
                (exit_price - entry) / entry * 100
                if direction == "LONG"
                else (entry - exit_price) / entry * 100
            )

            trades.append({
                "direction": direction,
                "signal_time": df.loc[i, "datetime"],
                "signal_close": float(df.loc[i, "close"]),
                "tested_level": (
                    float(level) if pd.notna(level) else np.nan
                ),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry,
                "exit_price": exit_price,
                "gross_return_pct": gross_pct,
                "holding_hours": elapsed,
            })

            last_exit_idx = exit_idx

    trade_df = pd.DataFrame(trades)
    trade_df.to_csv(TRADES_FILE, index=False)

    results = []

    for direction in ["LONG", "SHORT"]:
        subset = trade_df.loc[
            trade_df["direction"] == direction
        ] if not trade_df.empty else pd.DataFrame()

        gross = (
            subset["gross_return_pct"].to_numpy(dtype=float)
            if not subset.empty
            else np.array([], dtype=float)
        )

        for cost_pct in COSTS_PCT:
            net = gross - cost_pct

            results.append({
                "direction": direction,
                "assumed_cost_pct": cost_pct,
                "trade_count": len(net),
                "win_rate_pct": (
                    float(np.mean(net > 0) * 100)
                    if len(net) else np.nan
                ),
                "average_net_return_pct": (
                    float(np.mean(net)) if len(net) else np.nan
                ),
                "median_net_return_pct": (
                    float(np.median(net)) if len(net) else np.nan
                ),
                "total_compounded_return_pct": (
                    float((np.prod(1 + net / 100) - 1) * 100)
                    if len(net) else np.nan
                ),
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== PRELIMINARY HOLDOUT RESULTS ===")
    print(result_df.round(4).to_string(index=False))
    print("\nSaved summary:", OUTPUT_FILE)
    print("Saved individual trades:", TRADES_FILE)


if __name__ == "__main__":
    main()
