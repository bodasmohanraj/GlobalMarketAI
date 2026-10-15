
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "gold_features_clean.csv"
OUTPUT_FILE = BASE_DIR / "reliable_backtest_predictions.csv"

INITIAL_CAPITAL = 100_000.00

# Illustrative transaction cost per side.
# 0.0005 = 0.05% per side, or approximately 0.10% round trip.
COST_PER_SIDE = 0.0005

# A long signal requires probability >= 0.55.
# A short signal requires probability <= 0.45.
SIGNAL_THRESHOLD = 0.55

# Start predictions after the first 60% of the dataset.
INITIAL_TRAIN_FRACTION = 0.60

# Refit after another 250 eligible training observations.
REFIT_EVERY = 250

RANDOM_STATE = 42

FEATURES = [
    "return_1h",
    "return_4h",
    "return_12h",
    "sma_5_gap",
    "sma_20_gap",
    "trend_5_20",
    "volatility_10",
    "volatility_20",
    "rsi_14",
]


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_and_validate_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run build_clean_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "datetime",
        "close",
        "future_close_4h",
        "target_up",
        *FEATURES,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"], errors="coerce"
    )

    df["future_close_4h"] = pd.to_numeric(
        df["future_close_4h"], errors="coerce"
    )

    df["target_up"] = pd.to_numeric(
        df["target_up"], errors="coerce"
    )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "datetime",
            "close",
            "future_close_4h",
            "target_up",
            *FEATURES,
        ]
    ).copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain in the dataset.")

    if df["datetime"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. Resolve them before backtesting."
        )

    if not (df["close"] > 0).all():
        raise ValueError("Close prices must be positive.")

    if not (df["future_close_4h"] > 0).all():
        raise ValueError("Future close prices must be positive.")

    # Require exact four-hour target timestamps.
    df["target_time"] = df["datetime"] + pd.Timedelta(hours=4)

    # Validate the binary target.
    if not df["target_up"].isin([0, 1]).all():
        raise ValueError("target_up must contain only 0 and 1.")

    expected_target = (
        df["future_close_4h"] > df["close"]
    ).astype(int)

    if not np.array_equal(
        df["target_up"].astype(int).to_numpy(),
        expected_target.to_numpy(),
    ):
        raise ValueError(
            "target_up does not match future_close_4h > close."
        )

    # Validate target_time if the dataset already contains it.
    if "target_time" in df.columns:
        existing_target_time = pd.to_datetime(
            df["target_time"],
            utc=True,
            errors="coerce",
        )

        if not existing_target_time.equals(df["target_time"]):
            # Check timestamps directly after normalizing to UTC.
            if not (
                existing_target_time.reset_index(drop=True)
                .equals(df["target_time"].reset_index(drop=True))
            ):
                raise ValueError(
                    "Could not validate existing target_time column."
                )

    # Avoid silently training on invalid feature values.
    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Features contain infinite or invalid values.")

    df["target_up"] = df["target_up"].astype(int)

    if df["target_up"].nunique() < 2:
        raise ValueError(
            "The dataset must contain both target classes."
        )

    print("\nDATA VALIDATION")
    print("-" * 55)
    print(f"Rows: {len(df)}")
    print(f"First timestamp: {df['datetime'].iloc[0]}")
    print(f"Last timestamp:  {df['datetime'].iloc[-1]}")
    print(
        "Target horizon: exactly 4 hours after each prediction timestamp"
    )
    print("Features validated:", len(FEATURES))

    return df


# ============================================================
# MODEL
# ============================================================

def create_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# CHRONOLOGICAL WALK-FORWARD PREDICTIONS
# ============================================================

def walk_forward_predict(df):
    n_rows = len(df)

    start_index = int(
        n_rows * INITIAL_TRAIN_FRACTION
    )

    if start_index < 100:
        raise ValueError(
            "Not enough rows for the initial training period."
        )

    predictions = []

    model = None
    last_fit_end = 0

    print("\nWALK-FORWARD MODEL")
    print("-" * 55)
    print(f"Initial training fraction: {INITIAL_TRAIN_FRACTION:.0%}")
    print(f"First prediction row: {start_index}")
    print(f"Refit interval: {REFIT_EVERY} eligible observations")

    for i in range(start_index, n_rows):
        prediction_time = df.loc[i, "datetime"]

        # Only use training rows whose labels were already available
        # before the current prediction timestamp.
        earlier_rows = df.iloc[:i]

        eligible_mask = (
            earlier_rows["target_time"] < prediction_time
        )

        train = earlier_rows.loc[eligible_mask]

        eligible_end = len(train)

        if eligible_end < 100:
            continue

        should_refit = (
            model is None
            or eligible_end - last_fit_end >= REFIT_EVERY
        )

        if should_refit:
            X_train = train[FEATURES]
            y_train = train["target_up"]

            if y_train.nunique() < 2:
                warnings.warn(
                    f"Skipping prediction at {prediction_time}: "
                    "training data contains only one class."
                )
                model = None
                continue

            model = create_model()
            model.fit(X_train, y_train)

            last_fit_end = eligible_end

        X_current = df.loc[[i], FEATURES]

        probability_up = float(
            model.predict_proba(X_current)[0][
                list(model.named_steps["classifier"].classes_).index(1)
            ]
        )

        row = df.loc[i]

        forward_return = (
            row["future_close_4h"] / row["close"]
        ) - 1.0

        predictions.append(
            {
                "datetime": row["datetime"],
                "target_time": row["target_time"],
                "close": row["close"],
                "future_close_4h": row["future_close_4h"],
                "target_up": int(row["target_up"]),
                "prob_up": probability_up,
                "training_up_rate": float(
                    train["target_up"].mean()
                ),
                "forward_return": float(forward_return),
            }
        )

    result = pd.DataFrame(predictions)

    if result.empty:
        raise ValueError(
            "No out-of-sample predictions were generated."
        )

    result = result.sort_values(
        "datetime"
    ).reset_index(drop=True)

    if not (
        result["target_time"]
        == result["datetime"] + pd.Timedelta(hours=4)
    ).all():
        raise ValueError(
            "Prediction output contains invalid target horizons."
        )

    print(f"Out-of-sample predictions: {len(result)}")
    print(f"Test start: {result['datetime'].iloc[0]}")
    print(f"Test end:   {result['target_time'].iloc[-1]}")

    return result


# ============================================================
# SELECT NON-OVERLAPPING FOUR-HOUR PERIODS
# ============================================================

def select_non_overlapping(predictions):
    predictions = predictions.sort_values(
        "datetime"
    ).reset_index(drop=True)

    selected_rows = []
    next_allowed_time = None

    for _, row in predictions.iterrows():
        if (
            next_allowed_time is None
            or row["datetime"] >= next_allowed_time
        ):
            selected_rows.append(row)
            next_allowed_time = row["target_time"]

    result = pd.DataFrame(selected_rows).reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "No non-overlapping holding periods were selected."
        )

    # Confirm selected holding periods do not overlap.
    if len(result) > 1:
        previous_end = result["target_time"].iloc[:-1].reset_index(
            drop=True
        )
        next_start = result["datetime"].iloc[1:].reset_index(
            drop=True
        )

        if not (next_start >= previous_end).all():
            raise ValueError(
                "Overlapping holding periods were detected."
            )

    return result


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def print_classification_metrics(test):
    y_true = test["target_up"].to_numpy(dtype=int)

    model_probability = np.clip(
        test["prob_up"].to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )

    baseline_probability = np.clip(
        test["training_up_rate"].to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )

    model_prediction = (
        model_probability >= 0.5
    ).astype(int)

    baseline_prediction = (
        baseline_probability >= 0.5
    ).astype(int)

    metrics = {
        "Model accuracy": accuracy_score(
            y_true, model_prediction
        ),
        "Historical-rate baseline accuracy": accuracy_score(
            y_true, baseline_prediction
        ),
        "Model Brier score": brier_score_loss(
            y_true, model_probability
        ),
        "Baseline Brier score": brier_score_loss(
            y_true, baseline_probability
        ),
        "Model log loss": log_loss(
            y_true,
            model_probability,
            labels=[0, 1],
        ),
        "Baseline log loss": log_loss(
            y_true,
            baseline_probability,
            labels=[0, 1],
        ),
    }

    print("\nCLASSIFICATION METRICS")
    print("-" * 55)
    print(
        f"Model accuracy:          "
        f"{metrics['Model accuracy']:.6f}"
    )
    print(
        f"Baseline accuracy:       "
        f"{metrics['Historical-rate baseline accuracy']:.6f}"
    )
    print(
        f"Model Brier score:       "
        f"{metrics['Model Brier score']:.6f}"
    )
    print(
        f"Baseline Brier score:    "
        f"{metrics['Baseline Brier score']:.6f}"
    )
    print(
        f"Model log loss:          "
        f"{metrics['Model log loss']:.6f}"
    )
    print(
        f"Baseline log loss:       "
        f"{metrics['Baseline log loss']:.6f}"
    )

    return metrics


# ============================================================
# STRATEGY SIMULATION
# ============================================================

def simulate_strategy(name, test, positions):
    positions = np.asarray(positions, dtype=float)
    returns = test["forward_return"].to_numpy(dtype=float)

    if len(positions) != len(returns):
        raise ValueError(
            "Position count must match the number of test periods."
        )

    equity = float(INITIAL_CAPITAL)
    equity_curve = [equity]
    total_costs_currency = 0.0
    trade_count = 0

    round_trip_cost = 2 * COST_PER_SIDE

    for position, forward_return in zip(positions, returns):
        starting_equity = equity

        if position != 0:
            trade_count += 1
            trade_cost = starting_equity * round_trip_cost
        else:
            trade_cost = 0.0

        gross_pnl = (
            starting_equity * position * forward_return
        )

        # This cash-flow calculation is consistent with the
        # net return formula: position * return - transaction cost.
        equity = starting_equity + gross_pnl - trade_cost

        total_costs_currency += trade_cost
        equity_curve.append(equity)

    equity_curve = np.asarray(equity_curve, dtype=float)

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0

    total_return_pct = (
        equity / INITIAL_CAPITAL - 1.0
    ) * 100.0

    max_drawdown_pct = (
        drawdowns.min() * 100.0
    )

    # Measure win rate on active periods after costs.
    active_returns = []

    for position, forward_return in zip(positions, returns):
        if position != 0:
            net_return = (
                position * forward_return - round_trip_cost
            )
            active_returns.append(net_return)

    win_rate_pct = (
        np.mean(np.asarray(active_returns) > 0) * 100.0
        if active_returns
        else float("nan")
    )

    return {
        "strategy": name,
        "periods": len(test),
        "trades": trade_count,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "total_costs_currency": total_costs_currency,
        "final_equity": equity,
    }


# ============================================================
# RUN BACKTEST
# ============================================================

def main():
    df = load_and_validate_data()

    # Generate chronological out-of-sample predictions.
    predictions = walk_forward_predict(df)

    # Use non-overlapping four-hour holding periods.
    test = select_non_overlapping(predictions)

    # Signal positions:
    # +1 = long, -1 = short, 0 = flat.
    probability_up = test["prob_up"].to_numpy(dtype=float)

    positions = np.where(
        probability_up >= SIGNAL_THRESHOLD,
        1,
        np.where(
            probability_up <= (1 - SIGNAL_THRESHOLD),
            -1,
            0,
        ),
    )

    test["position"] = positions

    print("\nBACKTEST SETTINGS")
    print("-" * 55)
    print("Mode: research only")
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Cost per side: {COST_PER_SIDE:.3%}")
    print(f"Round-trip cost: {2 * COST_PER_SIDE:.3%}")
    print(f"Signal threshold: {SIGNAL_THRESHOLD:.2f}")
    print(f"Non-overlapping periods: {len(test)}")
    print(f"Active trades: {int(np.count_nonzero(positions))}")

    print_classification_metrics(test)

    # Compare the model strategy with simple benchmarks.
    strategy_results = [
        simulate_strategy(
            "AI threshold strategy",
            test,
            positions,
        ),
        simulate_strategy(
            "Always long",
            test,
            np.ones(len(test)),
        ),
        simulate_strategy(
            "Always short",
            test,
            -np.ones(len(test)),
        ),
        simulate_strategy(
            "Always flat",
            test,
            np.zeros(len(test)),
        ),
    ]

    results_df = pd.DataFrame(strategy_results)

    print("\nSTRATEGY METRICS")
    print("-" * 100)

    print(
        results_df.to_string(
            index=False,
            formatters={
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

    print("\nPOSITION COUNTS")
    print("-" * 55)
    print(
        pd.Series(positions)
        .value_counts()
        .sort_index()
        .rename(index={-1: "Short (-1)", 0: "Flat (0)", 1: "Long (+1)"})
        .to_string()
    )

    # Save the non-overlapping out-of-sample predictions.
    test.to_csv(OUTPUT_FILE, index=False)

    print("\nOUTPUT")
    print("-" * 55)
    print(f"Predictions saved to: {OUTPUT_FILE}")

    print("\nIMPORTANT")
    print(
        "This is a historical simulation, not a live trading signal. "
        "Transaction costs are illustrative, and results do not establish "
        "a reliable trading edge."
    )


if __name__ == "__main__":
    main()
