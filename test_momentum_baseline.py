
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("gold_features_clean.csv")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

df = df.dropna(
    subset=["return_4h", "target_up", "target_time"]
).sort_values("datetime").reset_index(drop=True)

df["target_up"] = df["target_up"].astype(int)

splitter = TimeSeriesSplit(n_splits=4)
scores = []

print("\n--- 4-HOUR MOMENTUM BASELINE ---")

for fold, (train_idx, test_idx) in enumerate(splitter.split(df), 1):
    test_start = df.iloc[test_idx]["datetime"].min()

    # Purge training rows whose target reaches the test period.
    train_idx = [
        i for i in train_idx
        if df.iloc[i]["target_time"] < test_start
    ]

    test = df.iloc[test_idx].copy()

    # Positive prior 4-hour return predicts up; otherwise down.
    predictions = (test["return_4h"] > 0).astype(int)

    acc = accuracy_score(test["target_up"], predictions)
    bal_acc = balanced_accuracy_score(
        test["target_up"], predictions
    )

    scores.append(acc)

    print(
        f"Fold {fold}: "
        f"Accuracy={acc:.4f}, "
        f"Balanced accuracy={bal_acc:.4f}, "
        f"Test rows={len(test)}"
    )

print("\nMean momentum accuracy:", round(sum(scores) / len(scores), 4))
