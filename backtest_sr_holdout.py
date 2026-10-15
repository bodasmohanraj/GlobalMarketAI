from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"
TRADES_OUT = BASE / "sr_holdout_trades.csv"
RESULTS_OUT = BASE / "sr_holdout_results.csv"

INITIAL_CAPITAL = 100_000.0
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
HOLD_HOURS = 4
MAX_EXIT_DELAY_HOURS = 5
NEAR_LEVEL = 0.002
HOLDOUT_FRACTION = 0.20

STRATEGIES = [
    "bounce_20", "breakout_20", "combined_20",
    "bounce_50", "breakout_50", "combined_50"
]


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing file: {SOURCE}")

    df = pd.read_csv(SOURCE)
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 200:
        raise ValueError("Not enough valid OHLC candles.")
    return df


def generate_signals(df):
    df = df.copy()

    for window in [20, 50]:
        resistance = (
            df["high"].rolling(window, min_periods=window)
            .max().shift(1)
        )
        support = (
            df["low"].rolling(window, min_periods=window)
            .min().shift(1)
        )

        near_support = (
            (df["low"] - support).abs() / df["close"] <= NEAR_LEVEL
        )
        near_resistance = (
            (df["high"] - resistance).abs() / df["close"] <= NEAR_LEVEL
        )

        df[f"bounce_{window}"] = np.select(
            [
                near_support & (df["close"] > df["open"])
                & (df["close"] > support),
                near_resistance & (df["close"] < df["open"])
                & (df["close"] < resistance),
            ],
            [1, -1],
            default=0,
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

        df[f"breakout_{window}"] = np.select(
            [breakout_up, breakout_down], [1, -1], default=0
        )

        bounce = df[f"bounce_{window}"]
        breakout = df[f"breakout_{window}"]

        df[f"combined_{window}"] = np.where(
            bounce == 0,
            breakout,
            np.where(
                (breakout == 0) | (bounce == breakout),
                bounce,
                0,
            ),
        )

    return df


def find_exit(df, entry_index):
    entry_time = df.loc[entry_index, "datetime"]
    target_time = entry_time + pd.Timedelta(hours=HOLD_HOURS)

    exit_index = int(
        df["datetime"].searchsorted(target_time, side="left")
    )
    if exit_index >= len(df):
        return None, None

    elapsed = (
        df.loc[exit_index, "datetime"] - entry_time
    ).total_seconds() / 3600

    if elapsed < HOLD_HOURS or elapsed > MAX_EXIT_DELAY_HOURS:
        return None, None

    return exit_index, elapsed


def create_trades(df, holdout_start):
    records = []

    for strategy in STRATEGIES:
        last_exit_time = None
        signal_mask = (
            (df["datetime"] >= holdout_start) & (df[strategy] != 0)
        )

        for signal_index in np.flatnonzero(signal_mask.to_numpy()):
            side = int(df.loc[signal_index, strategy])
            entry_index = signal_index + 1

            if entry_index >= len(df):
                continue

            entry_time = df.loc[entry_index, "datetime"]
            if entry_time < holdout_start:
                continue
            if (
                last_exit_time is not None
                and entry_time < last_exit_time
            ):
                continue

            exit_index, elapsed = find_exit(df, entry_index)
            if exit_index is None:
                continue

            exit_time = df.loc[exit_index, "datetime"]
            if exit_time < holdout_start:
                continue

            entry_price = float(df.loc[entry_index, "open"])
            exit_price = float(df.loc[exit_index, "close"])
            if entry_price <= 0 or exit_price <= 0:
                continue

            raw_return = exit_price / entry_price - 1
            directional_return = side * raw_return

            records.append({
                "strategy": strategy,
                "signal_time": df.loc[signal_index, "datetime"],
                "direction": "LONG" if side == 1 else "SHORT",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "holding_hours": elapsed,
                "raw_return_pct": raw_return * 100,
                "strategy_return_pct": directional_return * 100,
            })

            last_exit_time = exit_time

    return pd.DataFrame(records)


def calculate_results(trades):
    summaries = []

    for strategy in STRATEGIES:
        t = trades[trades["strategy"] == strategy].sort_values("entry_time")
        returns = t["strategy_return_pct"].to_numpy(dtype=float) / 100
        directions = t["direction"] if len(t) else pd.Series(dtype=str)

        row = {
            "strategy": strategy,
            "trades": len(t),
            "long_trades": int((directions == "LONG").sum()),
            "short_trades": int((directions == "SHORT").sum()),
            "win_rate_pct": float((returns > 0).mean() * 100) if len(returns) else 0.0,
            "mean_trade_return_pct": float(returns.mean() * 100) if len(returns) else 0.0,
        }

        for cost in COSTS:
            equity = INITIAL_CAPITAL
            peak = equity
            max_drawdown = 0.0

            for trade_return in returns:
                equity *= max(0.0, 1 + trade_return - cost)
                peak = max(peak, equity)
                drawdown = equity / peak - 1 if peak else -1
                max_drawdown = min(max_drawdown, drawdown)

            label = f"{cost * 100:.2f}%"
            row[f"return_after_cost_{label}"] = (
                (equity / INITIAL_CAPITAL - 1) * 100
            )

            if abs(cost - 0.001) < 1e-12:
                row["max_drawdown_at_0.10pct_cost"] = max_drawdown * 100

        summaries.append(row)

    return pd.DataFrame(summaries)


def main():
    df = generate_signals(load_data())
    split_index = int(len(df) * (1 - HOLDOUT_FRACTION))
    holdout_start = df.loc[split_index, "datetime"]

    trades = create_trades(df, holdout_start)
    results = calculate_results(trades)

    trades.to_csv(TRADES_OUT, index=False)
    results.to_csv(RESULTS_OUT, index=False)

    print("\n=== SUPPORT / RESISTANCE HOLDOUT ===")
    print("Candles:", len(df))
    print("Holdout starts:", holdout_start)
    print("Data ends:", df["datetime"].max())
    print("Holdout size:", f"{HOLDOUT_FRACTION:.0%}")
    print("Entry: next candle open")
    print("Exit: first available candle at/after 4 hours")
    print("\n=== RESULTS ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nSaved:", TRADES_OUT.name, "and", RESULTS_OUT.name)
    print("Research only; assumed costs are not broker-verified.")


if __name__ == "__main__":
    main()
