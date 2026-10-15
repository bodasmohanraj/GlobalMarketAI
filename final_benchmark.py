from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_features_clean.csv"
OUT = BASE / "final_benchmark_results.csv"

FEATURES = [
    "return_1h", "return_4h", "return_12h",
    "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14",
]
TARGET = "target_up"
TIME = "datetime"
TARGET_TIME = "target_time"
HORIZON = pd.Timedelta(hours=4)


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def metrics(y, prediction, probability):
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "log_loss": log_loss(y, probability, labels=[0, 1]),
    }


def main():
    df = pd.read_csv(DATA)
    df[TIME] = pd.to_datetime(df[TIME], utc=True, errors="coerce")
    df[TARGET_TIME] = pd.to_datetime(
        df[TARGET_TIME], utc=True, errors="coerce"
    )

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=[TIME, TARGET_TIME, TARGET, *FEATURES])
        .sort_values(TIME)
        .drop_duplicates(TIME, keep="last")
        .reset_index(drop=True)
    )
    df[TARGET] = df[TARGET].astype(int)

    # Latest 20% is the final test; never use it to select a threshold.
    final_start_position = int(len(df) * 0.80)
    final_test_start = df.iloc[final_start_position][TIME]

    earlier = df[df[TIME] < final_test_start].copy()
    test = df[df[TIME] >= final_test_start].copy()

    # Purge training rows whose future target reaches the test period.
    earlier = earlier[earlier[TARGET_TIME] < final_test_start].copy()

    # Split earlier data chronologically into model-fit and validation sets.
    validation_start_position = int(len(earlier) * 0.80)
    fit = earlier.iloc[:validation_start_position].copy()
    validation = earlier.iloc[validation_start_position:].copy()

    validation_start = validation[TIME].min()
    # Also prevent fit labels from overlapping the validation period.
    fit = fit[fit[TARGET_TIME] < validation_start].copy()

    if (
        fit.empty or validation.empty or test.empty
        or fit[TARGET].nunique() < 2
        or validation[TARGET].nunique() < 2
        or test[TARGET].nunique() < 2
    ):
        raise ValueError("A split has insufficient rows or target classes.")

    print("\nFINAL BENCHMARK — RESEARCH ONLY")
    print(f"Fit rows: {len(fit):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Final test rows: {len(test):,}")
    print(f"Final test begins: {final_test_start}")
    print(f"Final test ends: {test[TIME].max()}")
    print("The final test was not used for threshold selection.")

    # Choose threshold using validation balanced accuracy only.
    validation_model = make_model()
    validation_model.fit(fit[FEATURES], fit[TARGET])
    val_prob = validation_model.predict_proba(validation[FEATURES])[:, 1]

    candidates = np.arange(0.35, 0.651, 0.01)
    scored = []
    for threshold in candidates:
        val_pred = (val_prob >= threshold).astype(int)
        score = balanced_accuracy_score(validation[TARGET], val_pred)
        scored.append((score, -abs(threshold - 0.5), threshold))

    chosen_threshold = max(scored)[2]
    print(f"Threshold selected on validation: {chosen_threshold:.2f}")
    print(
        "Validation balanced accuracy at selected threshold:",
        round(max(scored)[0], 4)
    )

    # Refit on all data available before the final test, with target-time purge.
    final_train = df[
        (df[TIME] < final_test_start)
        & (df[TARGET_TIME] < final_test_start)
    ].copy()

    final_model = make_model()
    final_model.fit(final_train[FEATURES], final_train[TARGET])
    probability = final_model.predict_proba(test[FEATURES])[:, 1]
    prediction = (probability >= chosen_threshold).astype(int)
    y = test[TARGET].to_numpy()

    results = []

    model_metrics = metrics(y, prediction, probability)
    model_metrics.update({
        "method": "logistic_regression_selected_threshold",
        "threshold": chosen_threshold,
        "predicted_up_rate": prediction.mean(),
        "actual_up_rate": y.mean(),
    })
    results.append(model_metrics)

    # Constant 50% probability reference for proper scoring rules.
    neutral_prob = np.full(len(test), 0.5)
    neutral_pred = np.ones(len(test), dtype=int)
    neutral_metrics = metrics(y, neutral_pred, neutral_prob)
    neutral_metrics.update({
        "method": "constant_50pct_reference",
        "threshold": 0.5,
        "predicted_up_rate": 1.0,
        "actual_up_rate": y.mean(),
    })
    results.append(neutral_metrics)

    # Majority-class baseline learned from final training data only.
    majority_class = int(final_train[TARGET].mode().iloc[0])
    majority_pred = np.full(len(test), majority_class, dtype=int)
    training_up_rate = float(final_train[TARGET].mean())
    majority_prob = np.full(len(test), training_up_rate)
    majority_metrics = metrics(y, majority_pred, majority_prob)
    majority_metrics.update({
        "method": "majority_baseline",
        "threshold": 0.5,
        "predicted_up_rate": majority_pred.mean(),
        "actual_up_rate": y.mean(),
    })
    results.append(majority_metrics)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT, index=False)

    print("\n--- FINAL TEST RESULTS ---")
    print(result_df[
        ["method", "accuracy", "balanced_accuracy", "roc_auc",
         "brier_score", "log_loss", "predicted_up_rate", "actual_up_rate"]
    ].round(4).to_string(index=False))

    print("\nModel confusion matrix (rows=actual 0/1, columns=predicted 0/1):")
    print(confusion_matrix(y, prediction, labels=[0, 1]))
    print(f"\nSaved results to: {OUT}")
    print("A single final test is informative but has sampling uncertainty.")
    print("No result here establishes trading profitability.")


if __name__ == "__main__":
    main()
