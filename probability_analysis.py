
import pandas as pd
import numpy as np

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
all_predictions = []

print("\n--- RANDOM FOREST PROBABILITY ANALYSIS ---")

for fold in range(1, 5):
    test_idx = blocks[fold]
    test_start = df.loc[test_idx[0], "datetime"]

    train_idx = np.flatnonzero(
        (df["target_time"] < test_start).to_numpy()
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\nFold {fold}")
    print("Test rows:", len(test_idx))
    print(f"Accuracy at 0.50: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy at 0.50:",
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(f"Mean predicted probability: {probabilities.mean():.4f}")
    print(f"Probability standard deviation: {probabilities.std():.4f}")

    all_predictions.append(pd.DataFrame({
        "datetime": df.loc[test_idx, "datetime"].to_numpy(),
        "target": y_test.to_numpy(),
        "probability_up": probabilities,
    }))

out = pd.concat(all_predictions, ignore_index=True)
out.to_csv("rf_oof_probabilities.csv", index=False)

print("\nSaved out-of-fold probabilities to rf_oof_probabilities.csv")
print("Only later, held-out fold predictions are saved.")
