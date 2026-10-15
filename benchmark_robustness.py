
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "gold_data.csv"
TRADES_FILE = BASE / "walk_forward_trades.csv"
OUTPUT_FILE = BASE / "benchmark_robustness_results.csv"

INITIAL_CAPITAL = 100_000.0
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
RANDOM_RUNS = 5000
SEED = 42
N_SPLITS = 5


def compound_return(returns, cost=0.0):
    equity = INITIAL_CAPITAL

    for r in returns:
        equity *= max(0.0, 1.0 + float(r) - cost)

    return (equity / INITIAL_CAPITAL - 1.0) * 100


def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["datetime", "open", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in gold_data.csv: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return (
        df.dropna(subset=required)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )


def main():
    if not DATA_FILE.exists() or not TRADES_FILE.exists():
        raise FileNotFoundError(
            "Required files: gold_data.csv and walk_forward_trades.csv"
        )

    candles = load_data()
    trades = pd.read_csv(TRADES_FILE)
    trades.columns = trades.columns.str.strip().str.lower()

    required = {
        "fold", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "return_before_cost_pct"
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(
            f"Missing columns in walk_forward_trades.csv: {sorted(missing)}"
        )

    for col in ["entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(
            trades[col], utc=True, errors="coerce"
        )

    for col in ["entry_price", "exit_price", "return_before_cost_pct"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades = trades.dropna(
        subset=[
            "fold", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "return_before_cost_pct"
        ]
    ).copy()

    trades["side"] = (
        trades["direction"].str.upper().map({"LONG": 1, "SHORT": -1})
    )
    trades = trades.dropna(subset=["side"])
    trades["side"] = trades["side"].astype(int)

    # Recover the underlying price return from the directional strategy return.
    # strategy_return = side * underlying_price_return
    trades["raw_return"] = (
        trades["return_before_cost_pct"] / 100
    ) * trades["side"]

    # Recreate the exact chronological fold boundaries from the
    # walk-forward script.
    n = len(candles)
    first_test = int(n * 0.50)
    boundaries = np.linspace(first_test, n, N_SPLITS + 1, dtype=int)

    rng = np.random.default_rng(SEED)
    results = []

    for fold in range(1, N_SPLITS + 1):
        start_i = boundaries[fold - 1]
        end_i = boundaries[fold]

        start = candles.loc[start_i, "datetime"]
        end = (
            candles.loc[end_i, "datetime"]
            if end_i < n
            else candles.loc[end_i - 1, "datetime"]
                 + pd.Timedelta(nanoseconds=1)
        )

        period = trades[trades["fold"] == fold].sort_values("entry_time")
        actual_returns = period["return_before_cost_pct"].to_numpy() / 100
        raw_returns = period["raw_return"].to_numpy()

        # Buy-and-hold approximation over the entire candle fold.
        fold_candles = candles[
            (candles["datetime"] >= start)
            & (candles["datetime"] < end)
        ]

        if len(fold_candles) >= 2:
            first_price = float(fold_candles.iloc[0]["open"])
            last_price = float(fold_candles.iloc[-1]["close"])

            buy_hold_return = (
                last_price / first_price - 1
                if first_price > 0 else np.nan
            )
        else:
            buy_hold_return = np.nan

        row = {
            "fold": fold,
            "test_start": start,
            "test_end_exclusive": end,
            "strategy_trades": len(period),
            "buy_hold_return_pct": (
                buy_hold_return * 100
                if pd.notna(buy_hold_return) else np.nan
            ),
            "strategy_return_before_cost_pct": compound_return(
                actual_returns, 0.0
            ),
        }

        # Apply the same assumed per-trade cost to the strategy.
        for cost in COSTS:
            label = f"{cost * 100:.2f}%"
            row[f"strategy_return_after_cost_{label}"] = compound_return(
                actual_returns, cost
            )

        # Random-direction benchmark: retain the strategy's trade times
        # and underlying price moves, but randomly choose long or short.
        # This tests directional choice, not signal timing.
        random_returns = []

        if len(raw_returns):
            for _ in range(RANDOM_RUNS):
                random_sides = rng.choice([-1, 1], size=len(raw_returns))
                random_trade_returns = random_sides * raw_returns
                random_returns.append(
                    compound_return(random_trade_returns, 0.0005)
                )

            random_returns = np.asarray(random_returns)

            row["random_direction_median_at_0.05pct_cost"] = float(
                np.median(random_returns)
            )
            row["random_direction_5th_pct_at_0.05pct_cost"] = float(
                np.percentile(random_returns, 5)
            )
            row["random_direction_95th_pct_at_0.05pct_cost"] = float(
                np.percentile(random_returns, 95)
            )
            row["random_direction_pct_below_strategy"] = float(
                (random_returns < row[
                    "strategy_return_after_cost_0.05%"
                ]).mean() * 100
            )
        else:
            row["random_direction_median_at_0.05pct_cost"] = np.nan
            row["random_direction_5th_pct_at_0.05pct_cost"] = np.nan
            row["random_direction_95th_pct_at_0.05pct_cost"] = np.nan
            row["random_direction_pct_below_strategy"] = np.nan

        results.append(row)

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n=== BENCHMARK & ROBUSTNESS RESULTS ===")
    print("Random simulation runs per fold:", RANDOM_RUNS)
    print("Random seed:", SEED)
    print("Random-direction cost: 0.05% per trade")
    print("\n=== RESULTS BY FOLD ===")
    print(result.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== OVERALL SUMMARY ===")
    total_trades = len(trades)
    overall_returns = (
        trades.sort_values(["fold", "entry_time"])[
            "return_before_cost_pct"
        ].to_numpy() / 100
    )

    print("Total evaluation trades:", total_trades)
    print(
        "Combined strategy return before costs:",
        f"{compound_return(overall_returns, 0.0):.3f}%"
    )
    print(
        "Combined strategy return at 0.05% cost/trade:",
        f"{compound_return(overall_returns, 0.0005):.3f}%"
    )

    print("\nSaved:", OUTPUT_FILE.name)
    print(
        "\nResearch only. Random directions use the strategy's existing "
        "trade times and are not an independent random-entry strategy. "
        "Buy-and-hold is a price-return comparison, not a fully matched "
        "investment simulation."
    )


if __name__ == "__main__":
    main()
