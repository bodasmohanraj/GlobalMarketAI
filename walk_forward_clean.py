
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

df = pd.read_csv("gold_model_ready.csv")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

features = [
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

X = df[features]
y = df["target_up"].astype(int)

splitter = TimeSeriesSplit(n_splits=5, gap=4)

results = []

print("\n--- CLEAN WALK-FORWARD EVALUATION ---")

for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    majority_class = y_train.value_counts().idxmax()
    baseline_predictions = np.full(len(y_test), majority_class)

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    baseline = accuracy_score(y_test, baseline_predictions)

    results.append((accuracy, balanced, baseline))

    print(f"\nFold {fold}")
    print(f"Train rows: {len(train_idx)}")
    print(f"Test rows: {len(test_idx)}")
    print(f"Test period: {df['datetime'].iloc[test_idx[0]]} to "
          f"{df['datetime'].iloc[test_idx[-1]]}")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"Training-majority baseline: {baseline:.4f}")

results = np.array(results)

print("\n--- MEAN RESULTS ---")
print(f"Mean model accuracy: {results[:, 0].mean():.4f}")
print(f"Mean balanced accuracy: {results[:, 1].mean():.4f}")
print(f"Mean baseline accuracy: {results[:, 2].mean():.4f}")

print("\nReminder: classification accuracy is not trading profitability.")
