
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

FILE = "gold_model_ready.csv"
N_SPLITS = 5
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
df["future_time"] = df["datetime"] + pd.Timedelta(hours=4)
df = df.sort_values("datetime").reset_index(drop=True)

X = df[FEATURES]
y = df["target_up"].astype(int)

# Use expanding chronological training windows.
# Each test fold is a consecutive block of later observations.
blocks = np.array_split(np.arange(len(df)), N_SPLITS)
results = []

print("\n--- PURGED WALK-FORWARD EVALUATION ---")

for fold in range(1, N_SPLITS):
    test_idx = blocks[fold]
    test_start = df.loc[test_idx[0], "datetime"]

    # Train only on observations whose four-hour outcome ends
    # strictly before the test period begins.
    train_mask = df["future_time"] < test_start
    train_idx = np.flatnonzero(train_mask.to_numpy())

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if len(train_idx) == 0 or y_train.nunique() < 2:
        print(f"Fold {fold}: skipped (insufficient training classes)")
        continue

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Majority-class baseline using training labels only.
    majority = y_train.value_counts().idxmax()
    baseline = np.full(len(test_idx), majority)

    acc = accuracy_score(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    base_acc = accuracy_score(y_test, baseline)

    results.append((acc, bal_acc, base_acc))

    print(f"\nFold {fold}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Test rows: {len(test_idx)}")
    print(f"Test period: {df.loc[test_idx[0], 'datetime']} to "
          f"{df.loc[test_idx[-1], 'datetime']}")
    print(f"Model accuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Majority baseline: {base_acc:.4f}")

if results:
    results = np.array(results)
    print("\n--- MEAN RESULTS ---")
    print(f"Mean model accuracy: {results[:, 0].mean():.4f}")
    print(f"Mean balanced accuracy: {results[:, 1].mean():.4f}")
    print(f"Mean baseline accuracy: {results[:, 2].mean():.4f}")

print("\nThis is a classification evaluation, not a profitability test.")
