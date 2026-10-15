
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GLOBAL MARKET AI — FAIR BASELINE COMPARISON
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "reliable_backtest_predictions.csv"
FEATURES_FILE = BASE_DIR / "gold_features_clean.csv"

RESULTS_FILE = BASE_DIR / "baseline_strategy_comparison.csv"
PERIODS_FILE = BASE_DIR / "baseline_comparison_periods.csv"

INITIAL_CAPITAL = 100_000.00

# Same illustrative costs as reliable_backtest.py
COST_PER_SIDE = 0.0005
ROUND_TRIP_COST = 2 * COST_PER_SIDE

SIGNAL_THRESHOLD = 0.55


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {PREDICTIONS_FILE.name}.\n"
            "Run reliable_backtest.py first."
        )

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {FEATURES_FILE.name}.\n"
            "Run build_clean_features.py first."
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)
    features = pd.read_csv(FEATURES_FILE)

    prediction_required = [
        "datetime",
        "target_time",
        "close",
        "future_close_4h",
        "forward_return",
        "prob_up",
    ]

    feature_required = [
        "datetime",
        "close",
        "future_close_4h",
        "return_4h",
        "trend_5_20",
    ]

    missing_predictions = [
        c for c in prediction_required
        if c not in predictions.columns
    ]

    missing_features = [
        c for c in feature_required
        if c not in features.columns
    ]

    if missing_predictions:
        raise ValueError(
            f"Prediction file is missing columns: "
            f"{missing_predictions}"
        )

    if missing_features:
        raise ValueError(
            f"Feature file is missing columns: {missing_features}"
        )

    for frame in [predictions, features]:
        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            utc=True,
            errors="coerce",
        )

    predictions["target_time"] = pd.to_datetime(
        predictions["target_time"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "close",
        "future_close_4h",
        "forward_return",
        "prob_up",
    ]

    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    for column in [
        "close",
        "future_close_4h",
        "return_4h",
        "trend_5_20",
    ]:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    predictions = predictions.dropna(
        subset=prediction_required
    ).copy()

    features = features.dropna(
        subset=feature_required
    ).copy()

    if predictions["datetime"].duplicated().any():
        raise ValueError(
            "Duplicate prediction timestamps found."
        )

    if features["datetime"].duplicated().any():
        raise ValueError(
            "Duplicate feature timestamps found."
        )

    if not (
        predictions["target_time"]
        == predictions["datetime"] + pd.Timedelta(hours=4)
    ).all():
        raise ValueError(
            "Some predictions do not have an exact four-hour horizon."
        )

    if not predictions["prob_up"].between(0, 1).all():
        raise ValueError(
            "prob_up must be between 0 and 1."
        )

    predictions = predictions.sort_values(
        "datetime"
    ).reset_index(drop=True)

    features = features.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # Confirm the supplied forward return agrees with the prices.
    calculated_return = (
        predictions["future_close_4h"]
        / predictions["close"]
    ) - 1

    if not np.allclose(
        calculated_return.to_numpy(),
        predictions["forward_return"].to_numpy(),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            "forward_return does not match the recorded prices."
        )

    # Merge only the features needed for the baseline strategies.
    feature_subset = features[
        [
            "datetime",
            "close",
            "future_close_4h",
            "return_4h",
            "trend_5_20",
        ]
    ].copy()

    feature_subset = feature_subset.rename(
        columns={
            "close": "feature_close",
            "future_close_4h": "feature_future_close",
        }
    )

    merged = predictions.merge(
        feature_subset,
        on="datetime",
        how="left",
        validate="one_to_one",
    )

    if merged["return_4h"].isna().any():
        raise ValueError(
            "Some prediction timestamps have no valid return_4h feature."
        )

    if merged["trend_5_20"].isna().any():
        raise ValueError(
            "Some prediction timestamps have no valid trend_5_20 feature."
        )

    price_match = np.isclose(
        merged["close"].to_numpy(),
        merged["feature_close"].to_numpy(),
        rtol=1e-7,
        atol=0.02,
    )

    future_match = np.isclose(
        merged["future_close_4h"].to_numpy(),
        merged["feature_future_close"].to_numpy(),
        rtol=1e-7,
        atol=0.02,
    )

    if not price_match.all() or not future_match.all():
        raise ValueError(
            "Prediction prices do not match the clean feature dataset."
        )

    # Reconfirm non-overlapping holding periods.
    if len(merged) > 1:
        next_starts = merged["datetime"].iloc[1:].reset_index(drop=True)
        previous_ends = merged["target_time"].iloc[:-1].reset_index(
            drop=True
        )

        if not (next_starts >= previous_ends).all():
            raise ValueError(
                "Prediction periods overlap. Rebuild the prediction file."
            )

    # Use the AI's recorded positions when available.
    if "position" in merged.columns:
        merged["ai_position"] = pd.to_numeric(
            merged["position"],
            errors="coerce",
        )

        if merged["ai_position"].isna().any():
            raise ValueError("AI positions contain invalid values.")

        if not merged["ai_position"].isin([-1, 0, 1]).all():
            raise ValueError(
                "AI positions must be -1, 0, or 1."
            )
    else:
        merged["ai_position"] = np.where(
            merged["prob_up"] >= SIGNAL_THRESHOLD,
            1,
            np.where(
                merged["prob_up"] <= 1 - SIGNAL_THRESHOLD,
                -1,
                0,
            ),
        )

    print("\nINPUT VALIDATION")
    print("-" * 60)
    print(f"Common out-of-sample periods: {len(merged)}")
    print(f"First period: {merged['datetime'].iloc[0]}")
    print(f"Last holding-period end: {merged['target_time'].iloc[-1]}")
    print("Prediction prices match feature data: PASS")
    print("Exact four-hour horizon: PASS")
    print("Non-overlapping periods: PASS")

    return merged


# ============================================================
# CREATE STRATEGY POSITIONS
# ============================================================

def create_positions(data):
    result = data.copy()

    # Strategy 1: Existing AI model.
    result["AI"] = result["ai_position"].astype(int)

    # Strategy 2: Four-hour momentum.
    # Positive prior four-hour return -> long.
    # Negative prior four-hour return -> short.
    result["Momentum"] = np.sign(
        result["return_4h"]
    ).astype(int)

    # Strategy 3: Moving-average trend.
    # Positive SMA5/SMA20 relationship -> long.
    # Negative relationship -> short.
    result["Trend"] = np.sign(
        result["trend_5_20"]
    ).astype(int)

    # Strategy 4: Always flat.
    result["Flat"] = 0

    return result


# ============================================================
# SIMULATE ONE STRATEGY
# ============================================================

def simulate_strategy(name, forward_returns, positions):
    equity = float(INITIAL_CAPITAL)
    equity_curve = [equity]

    total_costs = 0.0
    trade_count = 0
    active_net_returns = []

    for forward_return, position in zip(
        forward_returns,
        positions,
    ):
        starting_equity = equity

        if position != 0:
            trade_count += 1
            trade_cost = starting_equity * ROUND_TRIP_COST
        else:
            trade_cost = 0.0

        gross_pnl = (
            starting_equity * position * forward_return
        )

        equity = starting_equity + gross_pnl - trade_cost

        total_costs += trade_cost
        equity_curve.append(equity)

        if position != 0:
            active_net_returns.append(
                position * forward_return - ROUND_TRIP_COST
            )

    equity_curve = np.asarray(equity_curve, dtype=float)

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1

    total_return_pct = (
        equity / INITIAL_CAPITAL - 1
    ) * 100

    max_drawdown_pct = drawdowns.min() * 100

    win_rate_pct = (
        np.mean(np.asarray(active_net_returns) > 0) * 100
        if active_net_returns
        else np.nan
    )

    return {
        "strategy": name,
        "periods": len(forward_returns),
        "trades": trade_count,
        "exposure_pct": (
            np.count_nonzero(positions) / len(positions) * 100
        ),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "total_costs_currency": total_costs,
        "final_equity": equity,
    }


# ============================================================
# RUN THE COMPARISON
# ============================================================

def main():
    data = load_inputs()
    data = create_positions(data)

    forward_returns = data["forward_return"].to_numpy(
        dtype=float
    )

    strategies = {
        "AI threshold": data["AI"].to_numpy(dtype=int),
        "Four-hour momentum": data["Momentum"].to_numpy(dtype=int),
        "Moving-average trend": data["Trend"].to_numpy(dtype=int),
        "Always flat": data["Flat"].to_numpy(dtype=int),
    }

    results = []

    for name, positions in strategies.items():
        results.append(
            simulate_strategy(
                name,
                forward_returns,
                positions,
            )
        )

    results_df = pd.DataFrame(results)

    print("\nBASELINE STRATEGY COMPARISON")
    print("=" * 105)
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Cost per side: {COST_PER_SIDE:.3%}")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.3%}")
    print("Holding period: exact four-hour target")
    print("All strategies use the same out-of-sample timestamps.")

    print("\nRESULTS")
    print("-" * 105)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "exposure_pct": "{:.2f}%".format,
                "total_return_pct": "{:.2f}%".format,
                "max_drawdown_pct": "{:.2f}%".format,
                "win_rate_pct": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.2f}%"
                ),
                "total_costs_currency": "{:,.2f}".format,
                "final_equity": "{:,.2f}".format,
            },
        )
    )

    # Save summary and individual period-level positions.
    results_df.to_csv(RESULTS_FILE, index=False)

    period_output = data[
        [
            "datetime",
            "target_time",
            "close",
            "future_close_4h",
            "forward_return",
            "prob_up",
            "return_4h",
            "trend_5_20",
            "AI",
            "Momentum",
            "Trend",
            "Flat",
        ]
    ].copy()

    period_output.to_csv(PERIODS_FILE, index=False)

    print("\nFILES SAVED")
    print("-" * 60)
    print(f"Summary: {RESULTS_FILE}")
    print(f"Period-by-period details: {PERIODS_FILE}")

    print("\nINTERPRETATION")
    print("-" * 60)
    print(
        "Compare returns, drawdowns, trade counts, and exposure together. "
        "A strategy with higher returns may also have greater risk."
    )
    print(
        "These are historical simulations using illustrative costs. "
        "They do not establish future profitability or account for all "
        "real-world execution effects."
    )


if __name__ == "__main__":
    main()
