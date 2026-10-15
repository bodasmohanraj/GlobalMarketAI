
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GLOBAL MARKET AI — PROBABILITY THRESHOLD EXPERIMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "reliable_backtest_predictions.csv"
RESULTS_FILE = BASE_DIR / "probability_threshold_results.csv"
DETAILS_FILE = BASE_DIR / "probability_threshold_periods.csv"

INITIAL_CAPITAL = 100_000.00
COST_PER_SIDE = 0.0005
ROUND_TRIP_COST = 2 * COST_PER_SIDE

# Pre-specified exploratory thresholds.
# These are not optimized or validated trading settings.
THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_predictions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}.\n"
            "Run reliable_backtest.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "datetime",
        "target_time",
        "close",
        "future_close_4h",
        "prob_up",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True, errors="coerce"
    )

    for col in ["close", "future_close_4h", "prob_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid predictions remain.")

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate prediction timestamps found.")

    if not (
        df["target_time"]
        == df["datetime"] + pd.Timedelta(hours=4)
    ).all():
        raise ValueError("Invalid four-hour target timestamps.")

    if not df["prob_up"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not (df["close"] > 0).all():
        raise ValueError("Close prices must be positive.")

    if not (df["future_close_4h"] > 0).all():
        raise ValueError("Future prices must be positive.")

    df["forward_return"] = (
        df["future_close_4h"] / df["close"]
    ) - 1.0

    # Check that periods do not overlap.
    if len(df) > 1:
        next_start = df["datetime"].iloc[1:].reset_index(drop=True)
        prior_end = df["target_time"].iloc[:-1].reset_index(drop=True)

        if not (next_start >= prior_end).all():
            raise ValueError(
                "Overlapping periods found. Use the non-overlapping "
                "predictions from reliable_backtest.py."
            )

    print("\nINPUT VALIDATION")
    print("-" * 60)
    print(f"Periods: {len(df)}")
    print(f"First prediction: {df['datetime'].iloc[0]}")
    print(f"Last target time: {df['target_time'].iloc[-1]}")
    print("Four-hour target alignment: PASS")
    print("Non-overlapping periods: PASS")

    return df


# ============================================================
# GENERATE POSITIONS
# ============================================================

def make_positions(probabilities, threshold):
    return np.where(
        probabilities >= threshold,
        1,
        np.where(
            probabilities <= 1 - threshold,
            -1,
            0,
        ),
    ).astype(int)


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(df, threshold):
    probabilities = df["prob_up"].to_numpy(dtype=float)
    forward_returns = df["forward_return"].to_numpy(dtype=float)

    positions = make_positions(probabilities, threshold)

    active = positions != 0
    trades = int(active.sum())

    # Gross strategy returns before transaction costs.
    gross_period_returns = positions * forward_returns

    # Net returns deduct the assumed round-trip cost per active period.
    costs = np.where(active, ROUND_TRIP_COST, 0.0)
    net_period_returns = gross_period_returns - costs

    # Compound equity using the same arithmetic convention as the
    # existing backtest: equity * (1 + position*return - cost).
    equity = float(INITIAL_CAPITAL)
    equity_curve = [equity]
    total_costs = 0.0

    for position, forward_return in zip(
        positions, forward_returns
    ):
        starting_equity = equity

        if position != 0:
            trade_cost = starting_equity * ROUND_TRIP_COST
        else:
            trade_cost = 0.0

        gross_pnl = starting_equity * position * forward_return

        equity = starting_equity + gross_pnl - trade_cost
        total_costs += trade_cost
        equity_curve.append(equity)

    equity_curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0

    gross_equity = INITIAL_CAPITAL * np.prod(
        1 + gross_period_returns
    )

    gross_return_pct = (
        gross_equity / INITIAL_CAPITAL - 1
    ) * 100

    net_return_pct = (
        equity / INITIAL_CAPITAL - 1
    ) * 100

    max_drawdown_pct = drawdowns.min() * 100

    active_gross = gross_period_returns[active]
    active_net = net_period_returns[active]

    gross_win_rate = (
        (active_gross > 0).mean() * 100
        if trades else np.nan
    )

    net_win_rate = (
        (active_net > 0).mean() * 100
        if trades else np.nan
    )

    long_mask = positions == 1
    short_mask = positions == -1

    avg_long_gross_pct = (
        forward_returns[long_mask].mean() * 100
        if long_mask.any() else np.nan
    )

    avg_short_gross_pct = (
        -forward_returns[short_mask].mean() * 100
        if short_mask.any() else np.nan
    )

    return {
        "threshold": threshold,
        "periods": len(df),
        "trades": trades,
        "long_trades": int(long_mask.sum()),
        "short_trades": int(short_mask.sum()),
        "exposure_pct": active.mean() * 100,
        "avg_probability_active": (
            probabilities[active].mean() if trades else np.nan
        ),
        "gross_return_pct": gross_return_pct,
        "net_return_pct": net_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "gross_win_rate_pct": gross_win_rate,
        "net_win_rate_pct": net_win_rate,
        "avg_long_gross_pct": avg_long_gross_pct,
        "avg_short_gross_pct": avg_short_gross_pct,
        "total_costs_currency": total_costs,
        "final_equity": equity,
    }, positions


# ============================================================
# RUN EXPERIMENT
# ============================================================

def main():
    df = load_predictions()

    all_results = []
    detail_frames = []

    probabilities = df["prob_up"].to_numpy(dtype=float)

    # Always-flat reference.
    # It is included for context and has no trading costs.
    flat_positions = np.zeros(len(df), dtype=int)

    flat_result, _ = evaluate_threshold(df, 1.01)
    flat_result["threshold"] = "Always flat"
    flat_result["trades"] = 0
    flat_result["long_trades"] = 0
    flat_result["short_trades"] = 0
    flat_result["exposure_pct"] = 0.0
    flat_result["avg_probability_active"] = np.nan
    flat_result["gross_return_pct"] = 0.0
    flat_result["net_return_pct"] = 0.0
    flat_result["max_drawdown_pct"] = 0.0
    flat_result["gross_win_rate_pct"] = np.nan
    flat_result["net_win_rate_pct"] = np.nan
    flat_result["avg_long_gross_pct"] = np.nan
    flat_result["avg_short_gross_pct"] = np.nan
    flat_result["total_costs_currency"] = 0.0
    flat_result["final_equity"] = INITIAL_CAPITAL

    all_results.append(flat_result)

    for threshold in THRESHOLDS:
        result, positions = evaluate_threshold(df, threshold)
        all_results.append(result)

        details = df[
            [
                "datetime",
                "target_time",
                "close",
                "future_close_4h",
                "forward_return",
                "prob_up",
            ]
        ].copy()

        details["threshold"] = threshold
        details["position"] = positions
        details["gross_period_return"] = (
            positions * details["forward_return"]
        )
        details["assumed_cost_fraction"] = np.where(
            positions != 0, ROUND_TRIP_COST, 0.0
        )
        details["net_period_return"] = (
            details["gross_period_return"]
            - details["assumed_cost_fraction"]
        )

        detail_frames.append(details)

    results = pd.DataFrame(all_results)

    print("\nPROBABILITY THRESHOLD EXPERIMENT")
    print("=" * 110)
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Cost per side: {COST_PER_SIDE:.3%}")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.3%}")
    print("Direction: long above threshold, short below its complement")
    print("Research analysis only; thresholds are exploratory.")

    print("\nRESULTS")
    print("-" * 110)

    formatters = {
        "exposure_pct": lambda x: f"{x:.2f}%",
        "gross_return_pct": lambda x: f"{x:.2f}%",
        "net_return_pct": lambda x: f"{x:.2f}%",
        "max_drawdown_pct": lambda x: f"{x:.2f}%",
        "gross_win_rate_pct": lambda x: (
            "N/A" if pd.isna(x) else f"{x:.2f}%"
        ),
        "net_win_rate_pct": lambda x: (
            "N/A" if pd.isna(x) else f"{x:.2f}%"
        ),
        "avg_long_gross_pct": lambda x: (
            "N/A" if pd.isna(x) else f"{x:.3f}%"
        ),
        "avg_short_gross_pct": lambda x: (
            "N/A" if pd.isna(x) else f"{x:.3f}%"
        ),
        "total_costs_currency": lambda x: f"{x:,.2f}",
        "final_equity": lambda x: f"{x:,.2f}",
        "avg_probability_active": lambda x: (
            "N/A" if pd.isna(x) else f"{x:.3f}"
        ),
    }

    print(
        results.to_string(
            index=False,
            formatters=formatters,
        )
    )

    results.to_csv(RESULTS_FILE, index=False)

    if detail_frames:
        all_details = pd.concat(
            detail_frames,
            ignore_index=True,
        )
    else:
        all_details = pd.DataFrame()

    all_details.to_csv(DETAILS_FILE, index=False)

    print("\nFILES SAVED")
    print("-" * 60)
    print(f"Summary: {RESULTS_FILE}")
    print(f"Period details: {DETAILS_FILE}")

    print("\nHOW TO READ THE RESULTS")
    print("-" * 60)
    print("Gross return excludes the assumed trading costs.")
    print("Net return includes the assumed round-trip cost per active period.")
    print("Maximum drawdown measures peak-to-trough equity decline.")
    print(
        "A threshold with fewer trades is not automatically better. "
        "Check trade count, gross return, net return, and drawdown together."
    )
    print(
        "Trying many thresholds on the same data can overfit. "
        "Do not treat the best result as proof of a trading edge."
    )


if __name__ == "__main__":
    main()
