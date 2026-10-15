
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Four expanding chronological test folds.
blocks = np.array_split(np.arange(len(df)), 5)
results = []

print("\n--- CLEAN DATA WALK-FORWARD EVALUATION ---")
print("Dataset rows:", len(df))

for fold in range(1, 5):
    test_idx = blocks[fold]
    test_start = df.loc[test_idx[0], "datetime"]

    # Purge training labels whose outcome reaches or crosses
    # the start of the test period.
    train_mask = df["target_time"] < test_start
    train_idx = np.flatnonzero(train_mask.to_numpy())

    if len(train_idx) == 0 or y.iloc[train_idx].nunique() < 2:
        print(f"Fold {fold}: insufficient training data")
        continue

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        ))
    ])

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    majority = y.iloc[train_idx].value_counts().idxmax()
    baseline_pred = np.full(len(test_idx), majority)

    acc = accuracy_score(y.iloc[test_idx], pred)
    bal = balanced_accuracy_score(y.iloc[test_idx], pred)
    base = accuracy_score(y.iloc[test_idx], baseline_pred)

    results.append((acc, bal, base))

    print(f"\nFold {fold}")
    print("Training rows:", len(train_idx))
    print("Test rows:", len(test_idx))
    print("Test period:",
          df.loc[test_idx[0], "datetime"], "to",
          df.loc[test_idx[-1], "datetime"])
    print(f"Model accuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal:.4f}")
    print(f"Majority baseline: {base:.4f}")

if results:
    r = np.array(results)
    print("\n--- AVERAGE RESULTS ---")
    print(f"Model accuracy: {r[:, 0].mean():.4f}")
    print(f"Balanced accuracy: {r[:, 1].mean():.4f}")
    print(f"Baseline accuracy: {r[:, 2].mean():.4f}")

print("\nResearch evaluation only; not a profitability test.")
