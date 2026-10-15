from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_features_clean.csv"
OUTPUT = BASE / "threshold_experiment_results.csv"

FEATURES = [
    "return_1h", "return_4h", "return_12h",
    "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14"
]
TARGET = "target_up"
HORIZON = pd.Timedelta(hours=4)


def make_model(balanced=False):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if balanced else None,
            random_state=42
        ))
    ])


def main():
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True, errors="coerce"
    )

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "target_time", TARGET, *FEATURES])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )
    df[TARGET] = df[TARGET].astype(int)

    # Six chronological test windows, with an earlier validation
    # segment carved out of each fold's training history.
    test_start_position = int(len(df) * 0.60)
    test_chunks = [
        c for c in np.array_split(
            np.arange(test_start_position, len(df)), 6
        ) if len(c)
    ]

    records = []

    print("\nTHRESHOLD EXPERIMENT — RESEARCH ONLY")
    print("Thresholds are selected on earlier validation data.")
    print("Each training set is purged by target timestamp.")

    for fold, positions in enumerate(test_chunks, 1):
        test = df.iloc[positions].copy()
        test_start = test["datetime"].min()

        eligible = df[
            (df["datetime"] < test_start)
            & (df["target_time"] < test_start)
        ].copy()

        if len(eligible) < 200:
            print(f"Fold {fold}: skipped; insufficient history.")
            continue

        # Latest 20% of eligible history is the threshold-validation set.
        split = int(len(eligible) * 0.80)
        fit_df = eligible.iloc[:split].copy()
        val_df = eligible.iloc[split:].copy()

        if (
            fit_df.empty or val_df.empty
            or fit_df[TARGET].nunique() < 2
            or val_df[TARGET].nunique() < 2
        ):
            print(f"Fold {fold}: skipped; insufficient class coverage.")
            continue

        test_features = test[FEATURES]
        y_test = test[TARGET].to_numpy()

        for model_name, balanced in [
            ("logistic_default", False),
            ("logistic_balanced", True)
        ]:
            model = make_model(balanced)
            model.fit(fit_df[FEATURES], fit_df[TARGET])

            val_prob = model.predict_proba(val_df[FEATURES])[:, 1]
            y_val = val_df[TARGET].to_numpy()

            # Pick the threshold with best validation balanced accuracy.
            # Ties prefer the threshold nearest 0.50.
            candidates = np.arange(0.35, 0.651, 0.01)
            scores = []
            for threshold in candidates:
                pred_val = (val_prob >= threshold).astype(int)
                scores.append(
                    (balanced_accuracy_score(y_val, pred_val),
                     -abs(threshold - 0.50), threshold)
                )
            chosen_threshold = max(scores)[2]

            # Refit on all eligible history; freeze threshold before test.
            final_model = make_model(balanced)
            final_model.fit(eligible[FEATURES], eligible[TARGET])
            test_prob = final_model.predict_proba(test_features)[:, 1]
            test_pred = (test_prob >= chosen_threshold).astype(int)

            records.append({
                "fold": fold,
                "model": model_name,
                "threshold_selected_on_validation": chosen_threshold,
                "validation_rows": len(val_df),
                "test_rows": len(test),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_balanced_accuracy": balanced_accuracy_score(
                    y_test, test_pred
                ),
                "test_predicted_up_rate": test_pred.mean(),
                "test_actual_up_rate": y_test.mean()
            })

            print(
                f"Fold {fold} | {model_name} | "
                f"threshold={chosen_threshold:.2f} | "
                f"accuracy={accuracy_score(y_test, test_pred):.3f} | "
                f"balanced_accuracy="
                f"{balanced_accuracy_score(y_test, test_pred):.3f} | "
                f"predicted_up={test_pred.mean():.3f}"
            )

    if not records:
        raise RuntimeError("No folds completed.")

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT, index=False)

    print("\n--- SUMMARY BY MODEL ---")
    print(
        result.groupby("model")[
            ["test_accuracy", "test_balanced_accuracy",
             "test_predicted_up_rate", "test_actual_up_rate"]
        ].mean().round(4).to_string()
    )
    print(f"\nSaved results to: {OUTPUT}")
    print("Threshold selection is still noisy with limited validation data.")
    print("Do not interpret these metrics as evidence of trading profitability.")


if __name__ == "__main__":
    main()
