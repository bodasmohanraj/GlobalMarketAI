import numpy as np
import pandas as pd

FILE = "validation_audit_oos_predictions_v2.csv"
HORIZON = "1hour"
BLOCK = 12
REPS = 2000
SEED = 42

df = pd.read_csv(FILE)
df = df[df["horizon"].eq(HORIZON)].copy()
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

required = ["actual_up", "p_up_model", "p_up_coinflip"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required)
y = df["actual_up"].astype(int).to_numpy()
p = df["p_up_model"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
p0 = df["p_up_coinflip"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()

# Proper probability metrics; lower is better.
loss_model = (p - y) ** 2
loss_ref = (p0 - y) ** 2
diff = loss_model - loss_ref

print("=" * 64)
print("1-HOUR PAIRED PROBABILITY AUDIT — RESEARCH ONLY")
print("=" * 64)
print("Rows:", len(df))
print("Mean model Brier:", round(loss_model.mean(), 6))
print("Mean 50% reference Brier:", round(loss_ref.mean(), 6))
print("Brier difference (model - reference):", round(diff.mean(), 6))
print("Negative difference favors model; positive favors reference.")
print("Note: overlapping labels require dependence-aware uncertainty.")

# Moving-block bootstrap of paired loss differences.
rng = np.random.default_rng(SEED)
n = len(diff)
blocks_needed = int(np.ceil(n / BLOCK))
starts = np.arange(max(1, n - BLOCK + 1))
boot_means = np.empty(REPS)

for b in range(REPS):
    indices = []
    for _ in range(blocks_needed):
        s = rng.choice(starts)
        indices.extend(range(s, min(s + BLOCK, n)))
    indices = np.asarray(indices[:n])
    boot_means[b] = diff[indices].mean()

lo, hi = np.quantile(boot_means, [0.025, 0.975])
print(f"12-row moving-block bootstrap 95% interval: [{lo:.6f}, {hi:.6f}]")
if hi < 0:
    print("Result: model Brier score is better in this bootstrap interval.")
elif lo > 0:
    print("Result: 50% reference Brier score is better in this bootstrap interval.")
else:
    print("Result: interval includes zero; no clear Brier-score advantage.")

print("Caution: this is a diagnostic interval, not a complete trading test.")
