from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)

INPUT = Path("threshold_oos_predictions.csv")
OUT = Path("calibration_analysis")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)
required = [
    "datetime", "target_time", "target_up",
    "model", "fold", "probability_up"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
df["target_up"] = pd.to_numeric(df["target_up"], errors="coerce")
df["probability_up"] = pd.to_numeric(
    df["probability_up"], errors="coerce"
)
df = df.dropna(subset=required).copy()
df = df[df["target_up"].isin([0, 1])]
df = df[df["probability_up"].between(0, 1)]
df = df.sort_values("datetime")

# Each row is a forecast of the four-hour outcome.
# Overlapping forecasts are reported separately from the
# every-four-hours diagnostic below.
def metrics(g):
    y = g["target_up"].astype(int).to_numpy()
    p = g["probability_up"].clip(1e-6, 1 - 1e-6).to_numpy()
    pred = (p >= 0.5).astype(int)
    prevalence = y.mean()
    baseline_brier = np.mean((y - prevalence) ** 2)
    brier = brier_score_loss(y, p)
    return {
        "rows": len(g),
        "actual_up_rate": prevalence,
        "mean_predicted_up_probability": p.mean(),
        "probability_bias": p.mean() - prevalence,
        "brier_score": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill_score": (
            1 - brier / baseline_brier
            if baseline_brier > 0 else np.nan
        ),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "directional_accuracy_at_0.5": accuracy_score(y, pred),
    }

overall = []
for name, g in df.groupby("model"):
    overall.append({"model": name, **metrics(g)})
overall_df = pd.DataFrame(overall)
overall_df.to_csv(OUT / "overall_metrics.csv", index=False)

# Calibration bins: predicted probability versus actual up rate.
edges = [0, .4, .45, .5, .55, .6, .65, .7, 1.000001]
df["probability_bin"] = pd.cut(
    df["probability_up"], bins=edges, right=False
)
cal = (
    df.groupby(["model", "probability_bin"], observed=False)
      .agg(
          forecasts=("target_up", "size"),
          mean_predicted_probability=("probability_up", "mean"),
          actual_up_rate=("target_up", "mean"),
      )
      .reset_index()
)
cal.to_csv(OUT / "calibration_bins.csv", index=False)

# Walk-forward fold stability.
fold_rows = []
for (name, fold), g in df.groupby(["model", "fold"]):
    fold_rows.append({
        "model": name,
        "fold": fold,
        **metrics(g),
    })
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT / "fold_metrics.csv", index=False)

# Reduce overlap: keep one forecast every four hours per model.
# This is a sensitivity check, not a full trading simulation.
sampled = []
for name, g in df.groupby("model"):
    g = g.sort_values("datetime").copy()
    g = g.drop_duplicates("datetime")
    g = g.set_index("datetime")
    # Choose the first available forecast at or after each 4-hour slot.
    slots = pd.date_range(
        g.index.min().ceil("4h"),
        g.index.max(),
        freq="4h",
    )
    chosen = []
    for slot in slots:
        candidates = g.loc[g.index >= slot]
        if len(candidates):
            row = candidates.iloc[0]
            chosen.append(row)
            g = g.loc[g.index > row.name]
    if chosen:
        s = pd.DataFrame(chosen)
        sampled.append({"model": name, **metrics(s)})

sampled_df = pd.DataFrame(sampled)
sampled_df.to_csv(OUT / "four_hour_sample_metrics.csv", index=False)

print("\n=== OVERALL METRICS (overlapping forecasts) ===")
print(overall_df.round(4).to_string(index=False))
print("\n=== CALIBRATION BINS ===")
print(cal.round(4).to_string(index=False))
print("\n=== WALK-FORWARD FOLD METRICS ===")
print(fold_df.round(4).to_string(index=False))
print("\n=== FOUR-HOUR-SAMPLED SENSITIVITY CHECK ===")
print(sampled_df.round(4).to_string(index=False))
print(f"\nSaved reports to: {OUT.resolve()}")
