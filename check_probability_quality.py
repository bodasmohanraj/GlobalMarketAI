
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)

FILE = "rf_oof_probabilities.csv"

df = pd.read_csv(FILE)

df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)
df["probability_up"] = pd.to_numeric(
    df["probability_up"], errors="raise"
)

df = df.dropna(subset=["target", "probability_up"])

y = df["target"].to_numpy()
p = df["probability_up"].clip(1e-6, 1 - 1e-6).to_numpy()
pred = (p >= 0.5).astype(int)

print("\n--- PROBABILITY QUALITY CHECK ---")
print("Held-out predictions:", len(df))
print(f"Observed upward rate: {y.mean():.4f}")
print(f"Mean predicted upward probability: {p.mean():.4f}")
print(f"Accuracy at 0.50: {accuracy_score(y, pred):.4f}")
print(f"Brier score: {brier_score_loss(y, p):.4f}")
print(f"Log loss: {log_loss(y, p):.4f}")

# A constant probability equal to the observed rate is a reference
# benchmark here, not an independently trained forecasting model.
constant_p = np.full(len(y), y.mean())

print(
    "Constant-rate reference Brier score:",
    f"{brier_score_loss(y, constant_p):.4f}"
)

df["probability_bucket"] = pd.cut(
    df["probability_up"],
    bins=[0, 0.45, 0.48, 0.50, 0.52, 0.55, 1.0],
    include_lowest=True,
)

summary = df.groupby(
    "probability_bucket",
    observed=False
).agg(
    count=("target", "size"),
    mean_predicted_probability=("probability_up", "mean"),
    observed_up_rate=("target", "mean"),
)

print("\nProbability buckets:")
print(summary.round(4).to_string())

print("\nSaved predictions were out-of-fold; no model was retrained.")
