
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DATA_FILE = "gold_features.csv"

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
GAP = 4


def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()

    required = FEATURES + [TARGET, "datetime"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    splitter = TimeSeriesSplit(n_splits=5, gap=GAP)
    results = []

    print("WALK-FORWARD VALIDATION — 9 FEATURES\n")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        baseline_class = y_train.mode().iloc[0]
        baseline_predictions = [baseline_class] * len(y_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(y_test, predictions)
        baseline = accuracy_score(y_test, baseline_predictions)

        results.append({
            "Fold": fold,
            "Test rows": len(test_idx),
            "Model accuracy": accuracy,
            "Balanced accuracy": balanced,
            "Baseline accuracy": baseline,
        })

    results_df = pd.DataFrame(results)

    print(results_df.to_string(
        index=False,
        formatters={
            "Model accuracy": "{:.4f}".format,
            "Balanced accuracy": "{:.4f}".format,
            "Baseline accuracy": "{:.4f}".format,
        },
    ))

    print("\nAverage model accuracy:",
          f"{results_df['Model accuracy'].mean():.4f}")
    print("Average balanced accuracy:",
          f"{results_df['Balanced accuracy'].mean():.4f}")
    print("Average baseline accuracy:",
          f"{results_df['Baseline accuracy'].mean():.4f}")

    results_df.to_csv("walk_forward_results.csv", index=False)
    print("\nSaved results to walk_forward_results.csv")


if __name__ == "__main__":
    main()
