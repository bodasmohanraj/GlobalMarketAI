
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

FILE = "gold_features_clean.csv"

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

df = pd.read_csv(FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

X = df[FEATURES]
y = df["target_up"].astype(int)

blocks = np.array_split(np.arange(len(df)), 5)
results = {"Logistic Regression": [], "Random Forest": [], "Baseline": []}

print("\n--- MODEL COMPARISON ---")
print("Dataset rows:", len(df))

for fold in range(1, 5):
    test_idx = blocks[fold]
    test_start = df.loc[test_idx[0], "datetime"]

    train_mask = df["target_time"] < test_start
    train_idx = np.flatnonzero(train_mask.to_numpy())

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if y_train.nunique() < 2:
        print(f"Fold {fold}: skipped; only one training class.")
        continue

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    majority_class = y_train.value_counts().idxmax()
    baseline_predictions = np.full(len(test_idx), majority_class)

    predictions_by_model = {
        "Baseline": baseline_predictions
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions_by_model[name] = model.predict(X_test)

    print(f"\nFold {fold}")
    print("Training rows:", len(train_idx))
    print("Test rows:", len(test_idx))

    for name, predictions in predictions_by_model.items():
        acc = accuracy_score(y_test, predictions)
        bal = balanced_accuracy_score(y_test, predictions)

        results[name].append((acc, bal))

        print(
            f"{name}: accuracy={acc:.4f}, "
            f"balanced_accuracy={bal:.4f}"
        )

print("\n--- MEAN RESULTS ACROSS FOLDS ---")

for name, fold_results in results.items():
    if not fold_results:
        continue

    values = np.array(fold_results)
    print(
        f"{name}: accuracy={values[:, 0].mean():.4f}, "
        f"balanced_accuracy={values[:, 1].mean():.4f}"
    )

print("\nResearch only. No trading signals generated.")

