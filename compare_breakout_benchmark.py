from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "sr_holdout_trades.csv"
OUTPUT = BASE / "breakout_benchmark_results.csv"

INITIAL_CAPITAL = 100_000.0
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
RANDOM_RUNS = 5000
SEED = 42


def compound_returns(returns, cost):
    equity = INITIAL_CAPITAL

    for ret in returns:
        equity *= max(0.0, 1.0 + ret - cost)

    return (equity / INITIAL_CAPITAL - 1.0) * 100


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Missing sr_holdout_trades.csv. Run backtest_sr_holdout.py first."
        )

    df = pd.read_csv(SOURCE)
    df = df[df["strategy"] == "breakout_20"].copy()

    required = [
        "direction", "entry_time", "exit_time",
        "entry_price", "exit_price"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True, errors="coerce"
    )
    df["exit_time"] = pd.to_datetime(
        df["exit_time"], utc=True, errors="coerce"
    )
    for col in ["entry_price", "exit_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("entry_time")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No breakout_20 trades found in the input file.")

    if (df["entry_price"] <= 0).any() or (df["exit_price"] <= 0).any():
        raise ValueError("Invalid non-positive entry or exit prices.")

    if (df["exit_time"] < df["entry_time"]).any():
        raise ValueError("Found an exit before entry.")

    # Identical entry/exit prices for all three comparisons.
    price_returns = (
        df["exit_price"].to_numpy(dtype=float)
        / df["entry_price"].to_numpy(dtype=float)
        - 1.0
    )

    directions = np.where(
        df["direction"].str.upper().to_numpy() == "LONG",
        1.0,
        -1.0
    )

    strategy_returns = directions * price_returns
    long_returns = price_returns

    rng = np.random.default_rng(SEED)
    random_results = {}

    # Shuffle the existing direction labels, preserving the count
    # of long and short signals while randomizing their assignment.
    for cost in COSTS:
        simulations = np.empty(RANDOM_RUNS)

        for i in range(RANDOM_RUNS):
            random_directions = rng.permutation(directions)
            simulations[i] = compound_returns(
                random_directions * price_returns, cost
            )

        random_results[cost] = simulations

    rows = []
    for cost in COSTS:
        strategy_return = compound_returns(strategy_returns, cost)
        long_return = compound_returns(long_returns, cost)
        random_values = random_results[cost]

        rows.extend([
            {
                "cost_per_trade_pct": cost * 100,
                "method": "Breakout_20",
                "return_pct": strategy_return,
                "random_5th_percentile_pct": np.nan,
                "random_median_pct": np.nan,
                "random_95th_percentile_pct": np.nan,
            },
            {
                "cost_per_trade_pct": cost * 100,
                "method": "Always_Long",
                "return_pct": long_return,
                "random_5th_percentile_pct": np.nan,
                "random_median_pct": np.nan,
                "random_95th_percentile_pct": np.nan,
            },
            {
                "cost_per_trade_pct": cost * 100,
                "method": "Random_Direction_Distribution",
                "return_pct": np.nan,
                "random_5th_percentile_pct": np.percentile(random_values, 5),
                "random_median_pct": np.percentile(random_values, 50),
                "random_95th_percentile_pct": np.percentile(random_values, 95),
            },
        ])

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT, index=False)

    print("\n=== BREAKOUT_20 MATCHED BENCHMARK ===")
    print("Trades:", len(df))
    print("Long signals:", int((directions == 1).sum()))
    print("Short signals:", int((directions == -1).sum()))
    print("First entry:", df["entry_time"].min())
    print("Last exit:", df["exit_time"].max())
    print("Random simulations per cost:", RANDOM_RUNS)
    print("Random seed:", SEED)
    print("\nAll methods use the same trade intervals and prices.")
    print("Costs are assumed per completed trade, not broker-verified.")
    print("\n=== RESULTS ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nSaved:", OUTPUT.name)
    print("Research only; randomization is a comparison, not a trading strategy.")


if __name__ == "__main__":
    main()
