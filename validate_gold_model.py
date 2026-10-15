from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gold_features_clean.csv"
PREDICTIONS_FILE = BASE_DIR / "walk_forward_predictions.csv"
METRICS_FILE = BASE_DIR / "walk_forward_metrics.csv"

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

TARGET = "target_up"
INITIAL_TRAIN_FRACTION = 0.60
N_TEST_FOLDS = 6
TARGET_HORIZON_HOURS = 4


def safe_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, probabilities)


def calculate_metrics(y_true, predictions, probabilities=None):
    result = {
        "rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_up": precision_score(y_true, predictions, zero_division=0),
        "recall_up": recall_score(y_true, predictions, zero_division=0),
        "f1_up": f1_score(y_true, predictions, zero_division=0),
    }
    if probabilities is not None:
        result["roc_auc"] = safe_auc(y_true, probabilities)
        result["brier_score"] = brier_score_loss(y_true, probabilities)
    return result


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run build_clean_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = ["datetime", TARGET, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Required columns are missing from gold_features_clean.csv: "
            + ", ".join(missing)
        )

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["datetime", TARGET, *FEATURES])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    # This is a binary direction task. Refuse unexpected target values.
    target_values = set(df[TARGET].astype(int).unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain only 0 and 1; found {sorted(target_values)}"
        )
    df[TARGET] = df[TARGET].astype(int)

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable rows; not enough for this validation.")

    first_test_position = int(len(df) * INITIAL_TRAIN_FRACTION)
    remaining_positions = np.arange(first_test_position, len(df))
    test_chunks = [
        chunk for chunk in np.array_split(remaining_positions, N_TEST_FOLDS)
        if len(chunk) > 0
    ]

    all_predictions = []
    fold_metrics = []

    print("\nGLOBAL MARKET AI — WALK-FORWARD VALIDATION")
    print("Research only. No automatic trading.")
    print(f"Usable rows: {len(df):,}")
    print(f"Features: {len(FEATURES)}")
    print(f"Test folds: {len(test_chunks)}")
    print(f"Target horizon: {TARGET_HORIZON_HOURS} elapsed hours")
    print("\nEach fold trains only on earlier observations and purges the final")
    print("four hours before its test window to reduce target-overlap leakage.")

    for fold_number, test_positions in enumerate(test_chunks, start=1):
        test = df.iloc[test_positions].copy()
        test_start = test["datetime"].min()

        # Purge any training label whose four-hour target may reach the test window.
        cutoff = test_start - pd.Timedelta(hours=TARGET_HORIZON_HOURS)
        train = df[df["datetime"] < cutoff].copy()

        if train.empty or train[TARGET].nunique() < 2:
            warnings.warn(f"Skipping fold {fold_number}: insufficient training classes.")
            continue

        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                )),
            ]
        )
        model.fit(X_train, y_train)

        model_probability = model.predict_proba(X_test)[:, 1]
        model_prediction = (model_probability >= 0.5).astype(int)

        # Fair simple baseline: majority class determined from this fold's training data.
        majority_class = int(y_train.mode().iloc[0])
        majority_prediction = np.full(len(test), majority_class, dtype=int)
        training_up_rate = float(y_train.mean())
        majority_probability = np.full(len(test), training_up_rate, dtype=float)

        # Simple momentum baseline: predict up if the trailing four-hour return is positive.
        momentum_prediction = (test["return_4h"].to_numpy() > 0).astype(int)

        fold_record = {
            "fold": fold_number,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start_utc": train["datetime"].min().isoformat(),
            "train_end_utc": train["datetime"].max().isoformat(),
            "test_start_utc": test_start.isoformat(),
            "test_end_utc": test["datetime"].max().isoformat(),
            "purge_cutoff_utc": cutoff.isoformat(),
        }

        for name, prediction, probability in [
            ("model", model_prediction, model_probability),
            ("majority_baseline", majority_prediction, majority_probability),
            ("momentum_baseline", momentum_prediction, None),
        ]:
            metrics = calculate_metrics(y_test.to_numpy(), prediction, probability)
            for metric_name, value in metrics.items():
                fold_record[f"{name}_{metric_name}"] = value

        fold_metrics.append(fold_record)

        fold_output = pd.DataFrame({
            "datetime": test["datetime"].to_numpy(),
            "fold": fold_number,
            "actual_target_up": y_test.to_numpy(),
            "model_prediction": model_prediction,
            "model_probability_up": model_probability,
            "majority_baseline_prediction": majority_prediction,
            "momentum_baseline_prediction": momentum_prediction,
        })
        all_predictions.append(fold_output)

        print(
            f"Fold {fold_number}: train={len(train):,}, test={len(test):,}, "
            f"model accuracy={accuracy_score(y_test, model_prediction):.3f}, "
            f"majority accuracy={accuracy_score(y_test, majority_prediction):.3f}, "
            f"momentum accuracy={accuracy_score(y_test, momentum_prediction):.3f}"
        )

    if not all_predictions:
        raise RuntimeError("No validation folds completed. Check the dataset and target.")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    # Overall metrics use only predictions made on unseen, later observations.
    actual = predictions_df["actual_target_up"].to_numpy()
    aggregate = []

    for name, pred_col, prob_col in [
        ("model", "model_prediction", "model_probability_up"),
        ("majority_baseline", "majority_baseline_prediction", None),
        ("momentum_baseline", "momentum_baseline_prediction", None),
    ]:
        probabilities = (
            predictions_df[prob_col].to_numpy() if prob_col is not None else None
        )
        metrics = calculate_metrics(actual, predictions_df[pred_col].to_numpy(), probabilities)
        metrics["method"] = name
        aggregate.append(metrics)

    aggregate_df = pd.DataFrame(aggregate)[
        ["method", "rows", "accuracy", "balanced_accuracy", "precision_up",
         "recall_up", "f1_up", "roc_auc", "brier_score"]
    ]

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    metrics_df.to_csv(METRICS_FILE, index=False)

    print("\n--- AGGREGATED OUT-OF-SAMPLE RESULTS ---")
    print(aggregate_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved predictions: {PREDICTIONS_FILE}")
    print(f"Saved fold metrics: {METRICS_FILE}")

    model_row = aggregate_df.loc[aggregate_df["method"] == "model"].iloc[0]
    majority_row = aggregate_df.loc[
        aggregate_df["method"] == "majority_baseline"
    ].iloc[0]
    momentum_row = aggregate_df.loc[
        aggregate_df["method"] == "momentum_baseline"
    ].iloc[0]

    print("\n--- INTERPRETATION ---")
    if (
        model_row["accuracy"] <= majority_row["accuracy"]
        and model_row["accuracy"] <= momentum_row["accuracy"]
    ):
        print("The model did not beat either simple baseline on aggregate accuracy.")
    else:
        print("The model beat at least one baseline on aggregate accuracy.")
    print("This is a diagnostic, not proof of a durable edge or profitability.")
    print("Do not select a model from accuracy alone; inspect balanced accuracy,")
    print("ROC-AUC, Brier score, fold stability, and realistic transaction costs.")


if __name__ == "__main__":
    main()
