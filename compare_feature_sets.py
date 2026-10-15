
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gold_features_clean.csv"

PREDICTIONS_FILE = BASE_DIR / "feature_set_predictions.csv"
SUMMARY_FILE = BASE_DIR / "feature_set_comparison.csv"

INITIAL_TRAIN_FRACTION = 0.60
REFIT_EVERY = 250

# Keep this threshold fixed for every model.
PROBABILITY_THRESHOLD = 0.55

# Illustrative cost: 0.05% per side, 0.10% per completed trade.
ROUND_TRIP_COST = 0.001

INITIAL_CAPITAL = 100_000.0

FEATURE_SETS = {
    "Full model": [
        "return_1h",
        "return_4h",
        "return_12h",
        "sma_5_gap",
        "sma_20_gap",
        "trend_5_20",
        "volatility_10",
        "volatility_20",
        "rsi_14",
    ],
    "Trend and momentum": [
        "return_1h",
        "return_4h",
        "return_12h",
        "sma_5_gap",
        "sma_20_gap",
        "trend_5_20",
    ],
    "Volatility and RSI": [
        "volatility_10",
        "volatility_20",
        "rsi_14",
    ],
}


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}.\n"
            f"Expected location: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "datetime",
        "target_time",
        "close",
        "future_close_4h",
        "target_up",
    }

    for features in FEATURE_SETS.values():
        required.update(features)

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True, errors="coerce"
    )

    numeric_cols = list(
        set().union(*FEATURE_SETS.values())
    ) + ["close", "future_close_4h"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target_up"] = pd.to_numeric(
        df["target_up"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "datetime",
            "target_time",
            "close",
            "future_close_4h",
            "target_up",
        ]
    ).copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    # Ensure the target refers to the exact four-hour horizon.
    expected_target_time = df["datetime"] + pd.Timedelta(hours=4)

    if not df["target_time"].eq(expected_target_time).all():
        raise ValueError(
            "Some target_time values are not exactly four hours "
            "after datetime. Check your feature-building script."
        )

    # Ensure target labels agree with future prices.
    expected_target = (
        df["future_close_4h"] > df["close"]
    ).astype(int)

    actual_target = df["target_up"].astype(int)

    if not expected_target.equals(actual_target):
        raise ValueError(
            "Some target_up labels disagree with the price data."
        )

    # Use one common dataset for every feature set.
    all_features = list(
        set().union(*FEATURE_SETS.values())
    )

    df = df.dropna(subset=all_features).reset_index(drop=True)

    if not df["target_up"].isin([0, 1]).all():
        raise ValueError("target_up must contain only 0 and 1.")

    if len(df) < 1000:
        raise ValueError(
            f"Only {len(df)} usable rows. Check the input dataset."
        )

    df["target_up"] = df["target_up"].astype(int)

    # Return over the four-hour forecast horizon.
    df["forward_return"] = (
        df["future_close_4h"] / df["close"]
    ) - 1.0

    return df


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=42,
            ),
        ),
    ])


def walk_forward_predictions(df, model_name, features):
    print(f"\nTraining feature set: {model_name}")
    print(f"Number of features: {len(features)}")

    initial_train_end = int(
        len(df) * INITIAL_TRAIN_FRACTION
    )

    if initial_train_end >= len(df):
        raise ValueError("Not enough data for walk-forward testing.")

    model = None
    last_fit_train_count = 0
    records = []

    for i in range(initial_train_end, len(df)):
        prediction_time = df.loc[i, "datetime"]

        # Only use labels that would already be known
        # at the time of the prediction.
        train = df.iloc[:i].copy()

        train = train[
            train["target_time"] < prediction_time
        ]

        if len(train) < 100:
            continue

        if train["target_up"].nunique() < 2:
            continue

        should_refit = (
            model is None
            or len(train) - last_fit_train_count >= REFIT_EVERY
        )

        if should_refit:
            model = create_model()

            model.fit(
                train[features],
                train["target_up"],
            )

            last_fit_train_count = len(train)

        current_row = df.iloc[[i]]

        probability_up = float(
            model.predict_proba(current_row[features])[0][
                list(model.named_steps["classifier"].classes_).index(1)
            ]
        )

        records.append({
            "model": model_name,
            "datetime": prediction_time,
            "target_time": df.loc[i, "target_time"],
            "close": df.loc[i, "close"],
            "future_close_4h": df.loc[i, "future_close_4h"],
            "forward_return": df.loc[i, "forward_return"],
            "target_up": int(df.loc[i, "target_up"]),
            "probability_up": probability_up,
            "training_up_rate": float(
                train["target_up"].mean()
            ),
            "training_rows": len(train),
        })

        if (i - initial_train_end + 1) % 1000 == 0:
            print(
                f"  Processed "
                f"{i - initial_train_end + 1} prediction rows..."
            )

    predictions = pd.DataFrame(records)

    if predictions.empty:
        raise ValueError(
            f"No predictions were generated for {model_name}."
        )

    print(f"Generated predictions: {len(predictions)}")

    return predictions


# ============================================================
# SELECT NON-OVERLAPPING FOUR-HOUR PERIODS
# ============================================================

def select_non_overlapping(predictions):
    predictions = predictions.sort_values(
        "datetime"
    ).reset_index(drop=True)

    selected = []
    last_target_time = None

    for _, row in predictions.iterrows():
        if (
            last_target_time is None
            or row["datetime"] >= last_target_time
        ):
            selected.append(row)
            last_target_time = row["target_time"]

    return pd.DataFrame(selected).reset_index(drop=True)


# ============================================================
# EVALUATION METRICS
# ============================================================

def classification_metrics(y_true, probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    y_true = np.asarray(y_true, dtype=int)
    predicted_class = (probabilities >= 0.5).astype(int)

    return {
        "accuracy_pct": accuracy_score(
            y_true, predicted_class
        ) * 100,
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
    }


def trading_metrics(predictions):
    probabilities = predictions[
        "probability_up"
    ].to_numpy()

    forward_returns = predictions[
        "forward_return"
    ].to_numpy()

    # Long when probability >= threshold.
    # Short when probability <= 1 - threshold.
    # Otherwise remain flat.
    positions = np.where(
        probabilities >= PROBABILITY_THRESHOLD,
        1,
        np.where(
            probabilities <= 1 - PROBABILITY_THRESHOLD,
            -1,
            0,
        ),
    )

    active = positions != 0

    gross_period_returns = positions * forward_returns

    net_period_returns = (
        gross_period_returns
        - active.astype(float) * ROUND_TRIP_COST
    )

    # Compound equity through each selected period.
    gross_equity = INITIAL_CAPITAL * np.cumprod(
        1 + gross_period_returns
    )

    net_equity = INITIAL_CAPITAL * np.cumprod(
        1 + net_period_returns
    )

    equity_curve = np.concatenate(
        ([INITIAL_CAPITAL], net_equity)
    )

    running_peak = np.maximum.accumulate(equity_curve)

    drawdowns = equity_curve / running_peak - 1.0

    active_returns = net_period_returns[active]

    if active_returns.size:
        wins = int((active_returns > 0).sum())
        win_rate = wins / active_returns.size * 100
    else:
        win_rate = 0.0

    return {
        "periods": len(predictions),
        "trades": int(active.sum()),
        "exposure_pct": float(active.mean() * 100),
        "gross_return_pct": float(
            (gross_equity[-1] / INITIAL_CAPITAL - 1) * 100
        ),
        "net_return_pct": float(
            (net_equity[-1] / INITIAL_CAPITAL - 1) * 100
        ),
        "max_drawdown_pct": float(drawdowns.min() * 100),
        "trade_win_rate_pct": win_rate,
        "estimated_costs": float(
            INITIAL_CAPITAL
            * ROUND_TRIP_COST
            * sum(
                net_equity[j - 1] / INITIAL_CAPITAL
                if j > 0 else 1.0
                for j in range(len(active))
                if active[j]
            )
        ),
        "final_equity": float(net_equity[-1]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("GOLD AI - FEATURE SET COMPARISON")
    print("=" * 65)

    df = load_data()

    print(f"\nUsable dataset rows: {len(df)}")
    print(f"First candle: {df['datetime'].min()}")
    print(f"Last candle:  {df['datetime'].max()}")
    print(
        f"Initial training fraction: "
        f"{INITIAL_TRAIN_FRACTION:.0%}"
    )
    print(f"Refit interval: {REFIT_EVERY} training rows")
    print(
        f"Fixed probability threshold: "
        f"{PROBABILITY_THRESHOLD:.2f}"
    )

    all_predictions = []

    for model_name, features in FEATURE_SETS.items():
        predictions = walk_forward_predictions(
            df,
            model_name,
            features,
        )

        selected = select_non_overlapping(predictions)
        all_predictions.append(selected)

        print(
            f"Non-overlapping periods for {model_name}: "
            f"{len(selected)}"
        )

    # Confirm that every model is evaluated on identical timestamps.
    reference_times = all_predictions[0]["datetime"].tolist()

    for predictions in all_predictions[1:]:
        if predictions["datetime"].tolist() != reference_times:
            raise ValueError(
                "Feature sets produced different evaluation timestamps. "
                "Comparison stopped to avoid unfair results."
            )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    results = []

    for model_name, predictions in zip(
        FEATURE_SETS.keys(),
        all_predictions,
    ):
        metrics = classification_metrics(
            predictions["target_up"],
            predictions["probability_up"],
        )

        baseline_metrics = classification_metrics(
            predictions["target_up"],
            predictions["training_up_rate"],
        )

        strategy = trading_metrics(predictions)

        results.append({
            "model": model_name,
            "features": len(FEATURE_SETS[model_name]),
            **metrics,
            "baseline_accuracy_pct": (
                baseline_metrics["accuracy_pct"]
            ),
            "baseline_brier_score": (
                baseline_metrics["brier_score"]
            ),
            "baseline_log_loss": (
                baseline_metrics["log_loss"]
            ),
            **strategy,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    display_columns = [
        "model",
        "features",
        "accuracy_pct",
        "brier_score",
        "log_loss",
        "baseline_accuracy_pct",
        "baseline_brier_score",
        "baseline_log_loss",
        "trades",
        "net_return_pct",
        "max_drawdown_pct",
        "final_equity",
    ]

    print("\n" + "=" * 110)
    print("FEATURE SET COMPARISON RESULTS")
    print("=" * 110)

    print(
        results_df[display_columns].round(4).to_string(
            index=False
        )
    )

    print("\nFiles saved:")
    print(f"  Predictions: {PREDICTIONS_FILE}")
    print(f"  Summary:     {SUMMARY_FILE}")

    print("\nHow to read the results:")
    print("- Higher accuracy is not always better.")
    print("- Lower Brier score is better.")
    print("- Lower log loss is better.")
    print("- Compare each model against its baseline.")
    print("- Trading returns are illustrative, not live results.")
    print(
        "- The 0.55 threshold is fixed for comparison; "
        "it has not been optimized here."
    )
    print(
        "\nIMPORTANT: These historical results are exploratory. "
        "The same historical period has already been examined, "
        "so this is not an independent confirmation of profitability."
    )


if __name__ == "__main__":
    main()
