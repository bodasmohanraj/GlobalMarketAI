
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "reliable_backtest_predictions.csv"
OUTPUT_FILE = BASE_DIR / "quick_holdout_results.csv"

INITIAL_CAPITAL = 100_000.0
COST_PER_SIDE = 0.0005
ROUND_TRIP_COST = 2 * COST_PER_SIDE
THRESHOLD = 0.58
DEVELOPMENT_FRACTION = 0.70


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE.name}. Run reliable_backtest.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "datetime",
        "target_time",
        "close",
        "future_close_4h",
        "prob_up",
        "forward_return",
        "training_up_rate",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True, errors="coerce"
    )

    numeric = [
        "close",
        "future_close_4h",
        "prob_up",
        "forward_return",
        "training_up_rate",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise ValueError("Not enough valid predictions for a holdout test.")

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate prediction timestamps detected.")

    if not (
        df["target_time"]
        == df["datetime"] + pd.Timedelta(hours=4)
    ).all():
        raise ValueError("Invalid four-hour target timestamps.")

    if not df["prob_up"].between(0, 1).all():
        raise ValueError("AI probabilities must be between 0 and 1.")

    if not df["training_up_rate"].between(0, 1).all():
        raise ValueError("Baseline probabilities must be between 0 and 1.")

    calculated_return = (
        df["future_close_4h"] / df["close"]
    ) - 1

    if not np.allclose(
        calculated_return.to_numpy(),
        df["forward_return"].to_numpy(),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("Forward returns do not match the recorded prices.")

    if len(df) > 1:
        next_start = df["datetime"].iloc[1:].reset_index(drop=True)
        previous_end = df["target_time"].iloc[:-1].reset_index(drop=True)

        if not (next_start >= previous_end).all():
            raise ValueError("Holding periods overlap.")

    return df


def make_positions(probabilities):
    return np.where(
        probabilities >= THRESHOLD,
        1,
        np.where(probabilities <= 1 - THRESHOLD, -1, 0),
    ).astype(int)


def evaluate(name, data, positions):
    returns = data["forward_return"].to_numpy(dtype=float)
    positions = np.asarray(positions, dtype=int)

    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    total_costs = 0.0
    trade_returns = []

    for position, forward_return in zip(positions, returns):
        starting_equity = equity

        cost = (
            starting_equity * ROUND_TRIP_COST
            if position != 0
            else 0.0
        )

        gross_pnl = starting_equity * position * forward_return
        equity = starting_equity + gross_pnl - cost

        total_costs += cost
        equity_curve.append(equity)

        if position != 0:
            trade_returns.append(
                position * forward_return - ROUND_TRIP_COST
            )

    curve = np.asarray(equity_curve)
    drawdown = curve / np.maximum.accumulate(curve) - 1

    gross_period_returns = positions * returns
    gross_equity = INITIAL_CAPITAL * np.prod(1 + gross_period_returns)

    return {
        "strategy": name,
        "periods": len(data),
        "trades": int(np.count_nonzero(positions)),
        "exposure_pct": np.mean(positions != 0) * 100,
        "gross_return_pct": (gross_equity / INITIAL_CAPITAL - 1) * 100,
        "net_return_pct": (equity / INITIAL_CAPITAL - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "net_win_rate_pct": (
            np.mean(np.asarray(trade_returns) > 0) * 100
            if trade_returns else np.nan
        ),
        "total_costs_currency": total_costs,
        "final_equity": equity,
    }


def main():
    df = load_data()

    split = int(len(df) * DEVELOPMENT_FRACTION)
    development = df.iloc[:split].copy()
    holdout = df.iloc[split:].copy()

    if holdout.empty:
        raise ValueError("Holdout segment is empty.")

    # Fixed threshold: do not choose a new threshold from holdout results.
    ai_positions = make_positions(holdout["prob_up"].to_numpy())

    # Historical-rate baseline uses only each row's recorded past training rate.
    baseline_positions = make_positions(
        holdout["training_up_rate"].to_numpy()
    )

    flat_positions = np.zeros(len(holdout), dtype=int)

    results = [
        evaluate("AI threshold 0.58", holdout, ai_positions),
        evaluate("Historical-rate baseline 0.58", holdout, baseline_positions),
        evaluate("Always flat", holdout, flat_positions),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nQUICK CHRONOLOGICAL HOLDOUT")
    print("=" * 75)
    print(f"All available periods: {len(df)}")
    print(f"Development periods (first 70%): {len(development)}")
    print(f"Holdout periods (final 30%): {len(holdout)}")
    print(f"Development ends: {development['target_time'].iloc[-1]}")
    print(f"Holdout starts:   {holdout['datetime'].iloc[0]}")
    print(f"Holdout ends:     {holdout['target_time'].iloc[-1]}")
    print(f"Fixed AI threshold: {THRESHOLD}")
    print(f"Round-trip cost assumption: {ROUND_TRIP_COST:.3%}")

    print("\nHOLDOUT RESULTS")
    print("-" * 100)
    print(
        results_df.to_string(
            index=False,
            formatters={
                "exposure_pct": lambda x: f"{x:.2f}%",
                "gross_return_pct": lambda x: f"{x:.2f}%",
                "net_return_pct": lambda x: f"{x:.2f}%",
                "max_drawdown_pct": lambda x: f"{x:.2f}%",
                "net_win_rate_pct": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.2f}%"
                ),
                "total_costs_currency": lambda x: f"{x:,.2f}",
                "final_equity": lambda x: f"{x:,.2f}",
            },
        )
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")
    print(
        "\nCAUTION: This is only a preliminary holdout. "
        "The full-period results were already inspected, and the "
        "historical-rate baseline is not an independently fitted strategy."
    )


if __name__ == "__main__":
    main()
