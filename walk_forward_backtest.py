
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"
OUTPUT = BASE / "walk_forward_results.csv"

WINDOW = 20
HOLD_HOURS = 4
MAX_EXIT_DELAY_HOURS = 5
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]

# Keep this fixed before reviewing the results.
INITIAL_CAPITAL = 100_000.0
N_SPLITS = 5


def load_data():
    df = pd.read_csv(SOURCE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 500:
        raise ValueError("Not enough candles for walk-forward research.")

    return df


def build_signals(df):
    df = df.copy()

    # Shift levels so the current candle cannot use its own high/low
    # to define the prior 20-candle breakout boundary.
    resistance = (
        df["high"].rolling(WINDOW, min_periods=WINDOW).max().shift(1)
    )
    support = (
        df["low"].rolling(WINDOW, min_periods=WINDOW).min().shift(1)
    )

    prev_close = df["close"].shift(1)
    prev_resistance = resistance.shift(1)
    prev_support = support.shift(1)

    breakout_up = (
        (df["close"] > resistance)
        & (prev_close <= prev_resistance)
    )
    breakout_down = (
        (df["close"] < support)
        & (prev_close >= prev_support)
    )

    df["signal"] = np.select(
        [breakout_up, breakout_down], [1, -1], default=0
    )

    return df


def get_exit(df, entry_index):
    entry_time = df.loc[entry_index, "datetime"]
    target = entry_time + pd.Timedelta(hours=HOLD_HOURS)

    exit_index = int(
        df["datetime"].searchsorted(target, side="left")
    )

    if exit_index >= len(df):
        return None

    exit_time = df.loc[exit_index, "datetime"]
    elapsed = (exit_time - entry_time).total_seconds() / 3600

    if elapsed < HOLD_HOURS or elapsed > MAX_EXIT_DELAY_HOURS:
        return None

    return exit_index


def evaluate_test_period(df, test_start, test_end, fold):
    trades = []
    last_exit_time = None

    # A signal can use the candles before test_start for its lookback.
    # However, entries must occur inside this test period.
    for i in range(len(df)):
        signal_time = df.loc[i, "datetime"]

        if signal_time < test_start or signal_time >= test_end:
            continue

        side = int(df.loc[i, "signal"])
        if side == 0:
            continue

        entry_index = i + 1
        if entry_index >= len(df):
            continue

        entry_time = df.loc[entry_index, "datetime"]

        if entry_time < test_start or entry_time >= test_end:
            continue

        if last_exit_time is not None and entry_time < last_exit_time:
            continue

        exit_index = get_exit(df, entry_index)
        if exit_index is None:
            continue

        exit_time = df.loc[exit_index, "datetime"]

        # Do not use an exit beyond the end of this test fold.
        if exit_time >= test_end:
            continue

        entry_price = float(df.loc[entry_index, "open"])
        exit_price = float(df.loc[exit_index, "close"])

        if entry_price <= 0 or exit_price <= 0:
            continue

        raw_return = exit_price / entry_price - 1
        strategy_return = side * raw_return

        trades.append({
            "fold": fold,
            "signal_time": signal_time,
            "direction": "LONG" if side == 1 else "SHORT",
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "holding_hours": (
                exit_time - entry_time
            ).total_seconds() / 3600,
            "return_before_cost_pct": strategy_return * 100,
        })

        last_exit_time = exit_time

    return pd.DataFrame(trades)


def summarize_fold(trades, fold, start, end):
    returns = (
        trades["return_before_cost_pct"].to_numpy(dtype=float) / 100
        if not trades.empty else np.array([])
    )

    row = {
        "fold": fold,
        "test_start": start,
        "test_end_exclusive": end,
        "trades": len(returns),
        "win_rate_pct": (
            float((returns > 0).mean() * 100)
            if len(returns) else 0.0
        ),
        "mean_trade_return_pct": (
            float(returns.mean() * 100)
            if len(returns) else 0.0
        ),
    }

    for cost in COSTS:
        equity = INITIAL_CAPITAL
        peak = equity
        max_drawdown = 0.0

        for r in returns:
            equity *= max(0.0, 1 + r - cost)
            peak = max(peak, equity)
            dd = equity / peak - 1 if peak else -1
            max_drawdown = min(max_drawdown, dd)

        label = f"{cost * 100:.2f}%"
        row[f"return_after_cost_{label}"] = (
            (equity / INITIAL_CAPITAL - 1) * 100
        )
        row[f"max_drawdown_{label}"] = max_drawdown * 100

    return row


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing file: {SOURCE}")

    df = build_signals(load_data())

    # Split the timeline into an initial warm-up/training history and
    # five consecutive, non-overlapping evaluation periods.
    n = len(df)
    first_test = int(n * 0.50)
    boundaries = np.linspace(first_test, n, N_SPLITS + 1, dtype=int)

    summaries = []
    all_trades = []

    for fold in range(N_SPLITS):
        start_i = boundaries[fold]
        end_i = boundaries[fold + 1]

        test_start = df.loc[start_i, "datetime"]
        if end_i < n:
            test_end = df.loc[end_i, "datetime"]
        else:
            # Exclusive boundary after the final candle.
            test_end = df.loc[end_i - 1, "datetime"] + pd.Timedelta(
                nanoseconds=1
            )

        fold_trades = evaluate_test_period(
            df, test_start, test_end, fold + 1
        )

        if not fold_trades.empty:
            all_trades.append(fold_trades)

        summaries.append(
            summarize_fold(fold_trades, fold + 1, test_start, test_end)
        )

    result = pd.DataFrame(summaries)
    result.to_csv(OUTPUT, index=False)

    print("\n=== WALK-FORWARD BREAKOUT_20 RESEARCH ===")
    print(f"Candles: {n}")
    print(f"Warm-up/training history: first {first_test} candles")
    print(f"Evaluation folds: {N_SPLITS}")
    print("Entry: next candle open")
    print("Exit: first available candle at/after 4 hours, maximum 5 hours")
    print("No position carried across fold boundaries")
    print("\n=== RESULTS BY FOLD ===")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(BASE / "walk_forward_trades.csv", index=False)
        print("\nTotal evaluation trades:", len(combined))
        print("Saved: walk_forward_trades.csv")
    else:
        print("\nNo eligible trades in the evaluation folds.")

    print("Saved:", OUTPUT.name)
    print(
        "\nResearch only. Costs are assumptions, not broker-verified. "
        "This is a fixed-rule evaluation, not a guarantee of profitability."
    )


if __name__ == "__main__":
    main()
