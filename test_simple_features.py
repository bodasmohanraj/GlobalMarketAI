
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("gold_features_clean.csv")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

features = [
    "return_1h",
    "return_4h",
    "return_12h",
    "trend_5_20",
]

df = df.dropna(subset=features + ["target_up", "target_time"]).copy()
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

X = df[features]
y = df["target_up"].astype(int)

splitter = TimeSeriesSplit(n_splits=4)
model_scores = []
baseline_scores = []

print("\n--- SIMPLE FEATURE WALK-FORWARD TEST ---")

for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    test_start = df.iloc[test_idx]["datetime"].min()

    train_idx = [
        i for i in train_idx
        if df.iloc[i]["target_time"] < test_start
    ]

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
    predictions = model.predict(X_test)

    majority_prediction = int(y_train.mean() >= 0.5)
    baseline = [majority_prediction] * len(y_test)

    acc = accuracy_score(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    base_acc = accuracy_score(y_test, baseline)

    model_scores.append(acc)
    baseline_scores.append(base_acc)

    print(
        f"Fold {fold}: "
        f"Model accuracy={acc:.4f}, "
        f"Balanced accuracy={bal_acc:.4f}, "
        f"Baseline accuracy={base_acc:.4f}"
    )

print("\nMean model accuracy:", round(sum(model_scores) / len(model_scores), 4))
print("Mean baseline accuracy:", round(sum(baseline_scores) / len(baseline_scores), 4))
