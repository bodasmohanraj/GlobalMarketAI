from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

BASE = Path(__file__).resolve().parent
INPUT = BASE / "validation_audit_oos_predictions.csv"
OUTPUT = BASE / "validation_audit_corrected_summary.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"Missing {INPUT}. Run audit_multi_horizon.py first.")

df = pd.read_csv(INPUT)

required = [
    "horizon", "fold", "actual_up", "actual_change",
    "p_up_model", "p_up_momentum", "pred_change_model",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

results = []

def add_class_result(horizon, method, y, decisions, probabilities):
    y = np.asarray(y, dtype=int)
    decisions = np.asarray(decisions, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)

    results.append({
        "horizon": horizon,
        "task": "classification",
        "method": method,
        "rows": len(y),
        "accuracy": accuracy_score(y, decisions),
        "balanced_accuracy": balanced_accuracy_score(y, decisions),
        "roc_auc": (
            roc_auc_score(y, probabilities)
            if len(np.unique(y)) == 2 else np.nan
        ),
        "brier": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "predicted_up_rate": decisions.mean(),
        "actual_up_rate": y.mean(),
    })

def add_reg_result(horizon, method, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    results.append({
        "horizon": horizon,
        "task": "regression",
        "method": method,
        "rows": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "mean_error": np.mean(predicted - actual),
    })

for horizon, all_h in df.groupby("horizon", sort=False):
    all_h = all_h.sort_values(["fold", "datetime"]).copy()
    y_all = all_h["actual_up"].astype(int).to_numpy()

    # Model's own probabilities and 0.5 decision threshold.
    p_model = all_h["p_up_model"].to_numpy(dtype=float)
    add_class_result(
        horizon, "model_logistic", y_all,
        (p_model >= 0.5).astype(int), p_model
    )

    # Momentum: explicit direction decision; use smoothed probability
    # for proper scoring, not a raw 0/1 probability.
    momentum_decision = (all_h["p_up_momentum"].to_numpy() > 0).astype(int)
    prevalence = float(y_all.mean())
    # Jeffreys-smoothed overall prevalence for descriptive pooled scoring.
    prevalence_smoothed = (y_all.sum() + 0.5) / (len(y_all) + 1.0)
    momentum_p = np.where(momentum_decision == 1, 0.75, 0.25)
    add_class_result(
        horizon, "baseline_momentum_15m", y_all,
        momentum_decision, momentum_p
    )

    # Fair probability-only reference: constant 50% probability.
    p50 = np.full(len(y_all), 0.5)
    add_class_result(
        horizon, "baseline_probability_50pct", y_all,
        np.zeros(len(y_all), dtype=int), p50
    )

    # A prevalence baseline is scored probabilistically. Its decision
    # is the training-only prevalence compared against 0.5 per fold.
    p_prev = np.empty(len(all_h), dtype=float)
    decision_prev = np.empty(len(all_h), dtype=int)
    for fold, fold_rows in all_h.groupby("fold", sort=True):
        idx = fold_rows.index
        earlier = all_h[
            (all_h["datetime"] < fold_rows["datetime"].min())
        ]
        if earlier.empty:
            # The first fold's true training prevalence isn't included in
            # the OOS file; do not pretend it is known.
            p = prevalence_smoothed
        else:
            p = (earlier["actual_up"].sum() + 0.5) / (len(earlier) + 1.0)
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        p_prev[all_h.index.get_indexer(idx)] = p
        decision_prev[all_h.index.get_indexer(idx)] = int(p >= 0.5)

    add_class_result(
        horizon, "baseline_historical_prevalence", y_all,
        decision_prev, p_prev
    )

    actual = all_h["actual_change"].to_numpy(dtype=float)
    pred_model = all_h["pred_change_model"].to_numpy(dtype=float)
    pred_zero = np.zeros(len(all_h), dtype=float)

    add_reg_result(horizon, "model_ridge", actual, pred_model)
    add_reg_result(horizon, "baseline_zero_change", actual, pred_zero)

    # Training-mean predictions must be supplied by the audit script's
    # training-only baseline column; use it when present.
    if "pred_change_train_mean" in all_h.columns:
        pred_mean = all_h["pred_change_train_mean"].to_numpy(dtype=float)
        add_reg_result(
            horizon, "baseline_training_mean_change", actual, pred_mean
        )

out = pd.DataFrame(results)
out.to_csv(OUTPUT, index=False)

print("CORRECTED BASELINE REPORT — RESEARCH ONLY")
print(out.round(4).to_string(index=False))
print(f"\nSaved: {OUTPUT}")
print(
    "\nImportant: the pooled historical-prevalence baseline is descriptive "
    "because the first fold's original training labels are not stored in "
    "the OOS predictions file. Use fold metrics for strictly fold-specific "
    "comparisons. Momentum probabilities are illustrative (0.75/0.25), "
    "not calibrated estimates."
)
