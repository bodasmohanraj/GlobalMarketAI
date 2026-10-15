
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FILE = "gold_features_clean.csv"

df = pd.read_csv(FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

df = df.sort_values("datetime").reset_index(drop=True)

all_features = [
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

simple_features = [
    "return_1h",
    "return_4h",
    "return_12h",
    "trend_5_20",
]

df = df.dropna(
    subset=all_features + ["target_up", "target_time"]
).reset_index(drop=True)

y = df["target_up"].astype(int)

splitter = TimeSeriesSplit(n_splits=4)
results = []

print("\n--- SAME-PERIOD MODEL COMPARISON ---")
print("Dataset rows:", len(df))

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(df), start=1
):
    test_start = df.iloc[test_idx]["datetime"].min()

    # Remove training labels whose outcome reaches the test period.
    train_idx = [
        i for i in train_idx
        if df.iloc[i]["target_time"] < test_start
    ]

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    X_train_all = train[all_features]
    X_test_all = test[all_features]

    X_train_simple = train[simple_features]
    X_test_simple = test[simple_features]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    majority_class = int(y_train.mean() >= 0.5)
    baseline_predictions = [majority_class] * len(test)

    models = {
        "Random Forest - All": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest - Simple": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression - All": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    }

    model_features = {
        "Random Forest - All": (X_train_all, X_test_all),
        "Random Forest - Simple": (
            X_train_simple, X_test_simple
        ),
        "Logistic Regression - All": (
            X_train_all, X_test_all
        ),
    }

    print(f"\nFold {fold} | Test rows: {len(test)}")

    baseline_acc = accuracy_score(
        y_test, baseline_predictions
    )
    baseline_bal = balanced_accuracy_score(
        y_test, baseline_predictions
    )

    print(
        f"Majority baseline: accuracy={baseline_acc:.4f}, "
        f"balanced={baseline_bal:.4f}"
    )

    results.append({
        "fold": fold,
        "model": "Majority baseline",
        "accuracy": baseline_acc,
        "balanced_accuracy": baseline_bal,
        "test_rows": len(test),
    })

    # Momentum prediction is evaluated on these same test rows.
    momentum_predictions = (
        test["return_4h"] > 0
    ).astype(int)

    momentum_acc = accuracy_score(
        y_test, momentum_predictions
    )
    momentum_bal = balanced_accuracy_score(
        y_test, momentum_predictions
    )

    print(
        f"4-hour momentum: accuracy={momentum_acc:.4f}, "
        f"balanced={momentum_bal:.4f}"
    )

    results.append({
        "fold": fold,
        "model": "4-hour momentum",
        "accuracy": momentum_acc,
        "balanced_accuracy": momentum_bal,
        "test_rows": len(test),
    })

    for name, model in models.items():
        Xtr, Xte = model_features[name]

        model.fit(Xtr, y_train)
        predictions = model.predict(Xte)

        acc = accuracy_score(y_test, predictions)
        bal = balanced_accuracy_score(y_test, predictions)

        print(
            f"{name}: accuracy={acc:.4f}, "
            f"balanced={bal:.4f}"
        )

        results.append({
            "fold": fold,
            "model": name,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "test_rows": len(test),
        })

result_df = pd.DataFrame(results)

print("\n--- MEAN RESULTS ACROSS SAME TEST PERIODS ---")
summary = result_df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"),
    mean_balanced_accuracy=("balanced_accuracy", "mean"),
)

print(summary.sort_values("mean_accuracy", ascending=False).round(4))

result_df.to_csv("same_period_comparison_results.csv", index=False)
print("\nSaved: same_period_comparison_results.csv")
