
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "sr_holdout_trades.csv"

BLOCK_SIZE = 5
SIMULATIONS = 10000
SEED = 42


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "sr_holdout_trades.csv not found. Run the holdout backtest first."
        )

    df = pd.read_csv(SOURCE)
    df = df[df["strategy"] == "breakout_20"].copy()
    df = df.sort_values("entry_time")

    returns = (
        pd.to_numeric(df["strategy_return_pct"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        / 100
    )

    n = len(returns)
    if n < BLOCK_SIZE * 2:
        raise ValueError("Not enough trades for block bootstrap.")

    observed_mean = returns.mean()
    rng = np.random.default_rng(SEED)
    bootstrap_means = np.empty(SIMULATIONS)

    # Circular moving-block bootstrap:
    # resample blocks of consecutive trades to preserve some dependence.
    possible_starts = np.arange(n)

    for sim in range(SIMULATIONS):
        sampled = []

        while len(sampled) < n:
            start = int(rng.choice(possible_starts))
            block = [
                returns[(start + offset) % n]
                for offset in range(BLOCK_SIZE)
            ]
            sampled.extend(block)

        sampled = np.asarray(sampled[:n])
        bootstrap_means[sim] = sampled.mean()

    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)

    # Bootstrap distribution centered under the zero-mean null.
    centered = returns - observed_mean
    null_means = np.empty(SIMULATIONS)

    for sim in range(SIMULATIONS):
        sampled = []

        while len(sampled) < n:
            start = int(rng.choice(possible_starts))
            block = [
                centered[(start + offset) % n]
                for offset in range(BLOCK_SIZE)
            ]
            sampled.extend(block)

        null_means[sim] = np.mean(sampled[:n])

    p_value = (
        1 + np.sum(np.abs(null_means) >= abs(observed_mean))
    ) / (SIMULATIONS + 1)

    print("\n=== BREAKOUT_20 BLOCK BOOTSTRAP ===")
    print("Trades:", n)
    print("Block size:", BLOCK_SIZE, "consecutive trades")
    print("Simulations:", SIMULATIONS)
    print("Observed mean return per trade:",
          f"{observed_mean * 100:.5f}%")
    print("Bootstrap 95% interval:",
          f"[{lower * 100:.5f}%, {upper * 100:.5f}%]")
    print("Approximate two-sided bootstrap p-value:",
          f"{p_value:.4f}")

    if lower <= 0 <= upper:
        print("Interpretation: the interval includes zero.")
    else:
        print("Interpretation: the interval excludes zero.")

    print(
        "\nCaution: this is an exploratory diagnostic. "
        "Results depend on block size and the tested period; "
        "it does not establish future profitability."
    )


if __name__ == "__main__":
    main()
