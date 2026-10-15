import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("validation_audit_oos_predictions_v2.csv")
OUTPUT = Path("paired_bootstrap_1h_report.csv")
BLOCK_LENGTH = 12
REPLICATES = 5000
SEED = 42

df = pd.read_csv(INPUT)
df = df[df["horizon"].astype(str).eq("1hour")].copy()

required = [
    "fold", "datetime", "actual_up",
    "p_up_model", "p_up_prevalence"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
for col in ["actual_up", "p_up_model", "p_up_prevalence"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()
df = df.sort_values(["fold", "datetime"]).reset_index(drop=True)

if df.duplicated(["fold", "datetime"]).any():
    raise ValueError("Duplicate fold/timestamp rows found.")

df["model_loss"] = (
    df["p_up_model"].clip(0, 1) - df["actual_up"]
) ** 2
df["baseline_loss"] = (
    df["p_up_prevalence"].clip(0, 1) - df["actual_up"]
) ** 2
df["loss_difference"] = df["model_loss"] - df["baseline_loss"]

# Negative difference favors the logistic model.
# Resample contiguous blocks separately within each fold.
groups = {
    str(fold): g.sort_values("datetime").reset_index(drop=True)
    for fold, g in df.groupby("fold", sort=True)
}
for name, g in groups.items():
    if len(g) < BLOCK_LENGTH:
        raise ValueError(f"Fold {name} is too short for block length.")

print("=" * 68)
print("PAIRED 1-HOUR BOOTSTRAP — RESEARCH ONLY")
print("=" * 68)
print("Rows:", len(df))
print("Block length:", BLOCK_LENGTH, "observations per block")
print("Replicates:", REPLICATES)
print("Negative difference favors logistic model.")
print()

summary = []
for name, g in groups.items():
    d = g["loss_difference"].to_numpy()
    summary.append({
        "fold": name,
        "rows": len(g),
        "model_brier": g["model_loss"].mean(),
        "baseline_brier": g["baseline_loss"].mean(),
        "difference_model_minus_baseline": d.mean(),
    })

fold_summary = pd.DataFrame(summary)
print("OBSERVED RESULTS BY FOLD")
print(fold_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Bootstrap each fold independently, then pool by its row count.
rng = np.random.default_rng(SEED)
boot_by_fold = {name: np.empty(REPLICATES) for name in groups}
pooled_boot = np.empty(REPLICATES)

for b in range(REPLICATES):
    fold_means = {}
    for name, g in groups.items():
        d = g["loss_difference"].to_numpy()
        n = len(d)
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n - BLOCK_LENGTH + 1))
            sampled.extend(d[start:start + BLOCK_LENGTH])
        sample = np.asarray(sampled[:n])
        fold_means[name] = sample.mean()
        boot_by_fold[name][b] = sample.mean()

    pooled_boot[b] = sum(
        len(groups[name]) * fold_means[name] for name in groups
    ) / len(df)

lo, hi = np.quantile(pooled_boot, [0.025, 0.975])
observed = df["loss_difference"].mean()

print()
print("POOLED PAIRED RESULT")
print(f"Model Brier:                 {df['model_loss'].mean():.6f}")
print(f"Training-prevalence Brier:   {df['baseline_loss'].mean():.6f}")
print(f"Difference (model - base):   {observed:.6f}")
print(f"95% block-bootstrap interval: [{lo:.6f}, {hi:.6f}]")

if hi < 0:
    verdict = "Model Brier advantage supported by this diagnostic interval."
elif lo > 0:
    verdict = "Training-prevalence baseline has lower Brier score in this interval."
else:
    verdict = "Interval includes zero; no clear Brier-score advantage."
print("Interpretation:", verdict)

for name in groups:
    flo, fhi = np.quantile(boot_by_fold[name], [0.025, 0.975])
    print(f"Fold {name} difference interval: [{flo:.6f}, {fhi:.6f}]")

fold_summary.to_csv(OUTPUT, index=False)
print("\nSaved fold summary:", OUTPUT.resolve())
print(
    "Caution: diagnostic moving-block bootstrap, not a complete "
    "trading test. The block length is a choice, and conclusions "
    "may change with other reasonable lengths."
)
