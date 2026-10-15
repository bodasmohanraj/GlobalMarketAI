from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    brier_score_loss, log_loss
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_features_clean.csv"
PRED_OUT = BASE / "model_comparison_predictions.csv"
FOLD_OUT = BASE / "model_comparison_by_fold.csv"
SUMMARY_OUT = BASE / "model_comparison_summary.csv"

FEATURES = [
    "return_1h", "return_4h", "return_12h",
    "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14"
]
HORIZON = pd.Timedelta(hours=4)
N_FOLDS = 6


def build_models():
    return {
        "logistic_C_0.1": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=42))
        ]),
        "logistic_C_1.0": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=30,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42
            ))
        ])
    }


def score_predictions(y, pred, prob):
    result = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "predicted_up_rate": float(np.mean(pred)),
        "actual_up_rate": float(np.mean(y))
    }
    if prob is not None and len(np.unique(y)) == 2:
        result.update({
            "roc_auc": roc_auc_score(y, prob),
            "brier_score": brier_score_loss(y, prob),
            "log_loss": log_loss(y, prob, labels=[0, 1])
        })
    else:
        result.update({
            "roc_auc": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan
        })
    return result


def main():
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True, errors="coerce"
    )

    for col in FEATURES + ["target_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "target_time", "target_up", *FEATURES])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )
    df["target_up"] = df["target_up"].astype(int)

    if len(df) < 1000:
        raise ValueError("Not enough usable rows for this comparison.")

    # Keep the earliest 60% for the initial training history.
    # The remaining observations are split into six sequential test folds.
    first_test = int(len(df) * 0.60)
    test_indices = np.arange(first_test, len(df))
    folds = np.array_split(test_indices, N_FOLDS)

    all_predictions = []
    fold_results = []

    print("\nCHRONOLOGICAL MODEL COMPARISON")
    print("Research only — no automatic trading")
    print("Rows:", len(df))
    print("Features:", len(FEATURES))
    print("Walk-forward folds:", N_FOLDS)
    print("Each test fold is later in time than its training data.")

    for fold_num, indices in enumerate(folds, start=1):
        if len(indices) == 0:
            continue

        test = df.iloc[indices].copy()
        test_start = test["datetime"].min()

        # Purge training examples whose four-hour target reaches
        # or crosses the first timestamp of the test fold.
        train = df[
            (df["datetime"] < test_start) &
            (df["target_time"] < test_start)
        ].copy()

        if len(train) < 100 or test["target_up"].nunique() < 2:
            print(f"Skipping fold {fold_num}: insufficient data/classes.")
            continue

        y = test["target_up"].to_numpy()

        print(
            f"\nFold {fold_num}: train={len(train)}, test={len(test)}, "
            f"test_start={test_start}"
        )

        # Simple baseline 1: predict the training majority class.
        majority = int(train["target_up"].mode().iloc[0])
        majority_pred = np.full(len(test), majority, dtype=int)
        majority_prob = np.full(len(test), train["target_up"].mean())

        # Simple baseline 2: positive 4-hour return means UP.
        momentum_pred = (test["return_4h"].to_numpy() > 0).astype(int)

        baseline_items = [
            ("majority_baseline", majority_pred, majority_prob),
            ("momentum_baseline", momentum_pred, None)
        ]

        for name, pred, prob in baseline_items:
            row = {
                "fold": fold_num,
                "method": name,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_start": test_start,
                "test_end": test["datetime"].max()
            }
            row.update(score_predictions(y, pred, prob))
            fold_results.append(row)

            for i, (_, source) in enumerate(test.iterrows()):
                all_predictions.append({
                    "fold": fold_num,
                    "method": name,
                    "datetime": source["datetime"],
                    "target_time": source["target_time"],
                    "actual": int(y[i]),
                    "predicted": int(pred[i]),
                    "probability_up": (
                        float(prob[i]) if prob is not None else np.nan
                    )
                })

        # Fit each candidate only on data available before this fold.
        for name, model in build_models().items():
            model.fit(train[FEATURES], train["target_up"])
            prob = model.predict_proba(test[FEATURES])[:, 1]
            pred = (prob >= 0.5).astype(int)

            row = {
                "fold": fold_num,
                "method": name,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_start": test_start,
                "test_end": test["datetime"].max()
            }
            row.update(score_predictions(y, pred, prob))
            fold_results.append(row)

            for i, (_, source) in enumerate(test.iterrows()):
                all_predictions.append({
                    "fold": fold_num,
                    "method": name,
                    "datetime": source["datetime"],
                    "target_time": source["target_time"],
                    "actual": int(y[i]),
                    "predicted": int(pred[i]),
                    "probability_up": float(prob[i])
                })

            print(
                f"  {name}: accuracy={accuracy_score(y, pred):.3f}, "
                f"balanced_accuracy={balanced_accuracy_score(y, pred):.3f}"
            )

    predictions = pd.DataFrame(all_predictions)
    fold_df = pd.DataFrame(fold_results)

    if predictions.empty:
        raise ValueError("No predictions were generated.")

    predictions.to_csv(PRED_OUT, index=False)
    fold_df.to_csv(FOLD_OUT, index=False)

    summaries = []
    for name, group in predictions.groupby("method"):
        y = group["actual"].to_numpy()
        pred = group["predicted"].to_numpy()
        prob = group["probability_up"].to_numpy()

        usable_prob = np.isfinite(prob)
        result = {"method": name, "rows": len(group)}
        result.update(score_predictions(
            y, pred, prob if usable_prob.all() else None
        ))
        summaries.append(result)

    summary = pd.DataFrame(summaries).sort_values(
        "balanced_accuracy", ascending=False
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n\nAGGREGATE OUT-OF-SAMPLE RESULTS")
    print(summary.round(4).to_string(index=False))

    print("\nFOLD-BY-FOLD RESULTS")
    cols = [
        "fold", "method", "accuracy", "balanced_accuracy",
        "roc_auc", "brier_score", "log_loss"
    ]
    print(fold_df[cols].round(4).to_string(index=False))

    print("\nSaved files:")
    print(PRED_OUT)
    print(FOLD_OUT)
    print(SUMMARY_OUT)
    print("\nUse the results as research, not as proof of trading profitability.")


if __name__ == "__main__":
    main()
