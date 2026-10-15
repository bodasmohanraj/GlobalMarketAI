from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    brier_score_loss, log_loss
)

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "gold_features_clean.csv")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

groups = {
    "momentum_only": ["return_1h", "return_4h", "return_12h"],
    "trend_only": ["sma_5_gap", "sma_20_gap", "trend_5_20"],
    "volatility_rsi_only": ["volatility_10", "volatility_20", "rsi_14"],
    "all_features": [
        "return_1h", "return_4h", "return_12h", "sma_5_gap",
        "sma_20_gap", "trend_5_20", "volatility_10",
        "volatility_20", "rsi_14"
    ]
}

features = sorted(set(c for group in groups.values() for c in group))
df = df.dropna(subset=["datetime", "target_time", "target_up"] + features)
df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
df["target_up"] = df["target_up"].astype(int)

folds = np.array_split(np.arange(int(len(df) * 0.60), len(df)), 6)
fold_rows, prediction_rows = [], []

for fold_no, indices in enumerate(folds, 1):
    test = df.iloc[indices].copy()
    if test.empty or test["target_up"].nunique() < 2:
        continue

    start = test["datetime"].min()
    train = df[
        (df["datetime"] < start) & (df["target_time"] < start)
    ].copy()
    y = test["target_up"].to_numpy()

    for name, cols in groups.items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1, max_iter=2000, random_state=42
            ))
        ])
        model.fit(train[cols], train["target_up"])
        prob = model.predict_proba(test[cols])[:, 1]
        pred = (prob >= 0.5).astype(int)

        row = {
            "fold": fold_no, "feature_group": name,
            "train_rows": len(train), "test_rows": len(test),
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "brier_score": brier_score_loss(y, prob),
            "log_loss": log_loss(y, prob, labels=[0, 1]),
            "predicted_up_rate": pred.mean(),
            "actual_up_rate": y.mean()
        }
        fold_rows.append(row)

        for i, (_, source) in enumerate(test.iterrows()):
            prediction_rows.append({
                "fold": fold_no, "feature_group": name,
                "datetime": source["datetime"],
                "actual": int(y[i]), "predicted": int(pred[i]),
                "probability_up": float(prob[i])
            })

    print(f"Completed fold {fold_no}: {start}")

predictions = pd.DataFrame(prediction_rows)
fold_results = pd.DataFrame(fold_rows)

if predictions.empty:
    raise ValueError("No predictions produced. Check the input dataset.")

summary_rows = []
for name, g in predictions.groupby("feature_group"):
    y, pred, prob = (
        g["actual"].to_numpy(),
        g["predicted"].to_numpy(),
        g["probability_up"].to_numpy()
    )
    summary_rows.append({
        "feature_group": name,
        "rows": len(g),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "brier_score": brier_score_loss(y, prob),
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "predicted_up_rate": pred.mean(),
        "actual_up_rate": y.mean()
    })

summary = pd.DataFrame(summary_rows).sort_values(
    "balanced_accuracy", ascending=False
)
summary.to_csv(BASE / "feature_ablation_summary.csv", index=False)
fold_results.to_csv(BASE / "feature_ablation_by_fold.csv", index=False)
predictions.to_csv(BASE / "feature_ablation_predictions.csv", index=False)

print("\nAGGREGATE RESULTS")
print(summary.round(4).to_string(index=False))
print("\nRESULTS BY FOLD")
print(fold_results[
    ["fold", "feature_group", "accuracy", "balanced_accuracy",
     "roc_auc", "brier_score", "log_loss"]
].round(4).to_string(index=False))
print("\nSaved three feature_ablation CSV files.")
