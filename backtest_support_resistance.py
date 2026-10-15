from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_features_clean.csv"
OUTPUT = BASE / "support_resistance_predictions.csv"
SUMMARY = BASE / "support_resistance_results.csv"

INITIAL_CAPITAL = 100_000.0
INITIAL_TRAIN_FRACTION = 0.60
REFIT_EVERY = 250
THRESHOLD = 0.55
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]

df = pd.read_csv(SOURCE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

required = [
    "datetime", "target_time", "close", "high", "low",
    "future_close_4h", "target_up", "return_1h", "return_4h",
    "return_12h", "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if not (df["target_time"] == df["datetime"] + pd.Timedelta(hours=4)).all():
    raise ValueError("Target horizon is not exactly four hours.")

# Use only candles preceding the prediction timestamp.
for window in (20, 50):
    df[f"resistance_{window}"] = (
        df["high"].rolling(window, min_periods=window).max().shift(1)
    )
    df[f"support_{window}"] = (
        df["low"].rolling(window, min_periods=window).min().shift(1)
    )
    df[f"distance_resistance_{window}"] = (
        df[f"resistance_{window}"] - df["close"]
    ) / df["close"]
    df[f"distance_support_{window}"] = (
        df["close"] - df[f"support_{window}"]
    ) / df["close"]

df["range_position_20"] = (
    (df["close"] - df["support_20"]) /
    (df["resistance_20"] - df["support_20"]).replace(0, np.nan)
)
df["range_position_50"] = (
    (df["close"] - df["support_50"]) /
    (df["resistance_50"] - df["support_50"]).replace(0, np.nan)
)
df["above_resistance_20"] = (
    df["close"] > df["resistance_20"]
).astype(float)
df["below_support_20"] = (
    df["close"] < df["support_20"]
).astype(float)

base_features = [
    "return_1h", "return_4h", "return_12h", "sma_5_gap",
    "sma_20_gap", "trend_5_20", "volatility_10",
    "volatility_20", "rsi_14"
]
sr_features = base_features + [
    "distance_resistance_20", "distance_support_20",
    "distance_resistance_50", "distance_support_50",
    "range_position_20", "range_position_50",
    "above_resistance_20", "below_support_20"
]

df["target_up"] = pd.to_numeric(df["target_up"], errors="coerce")
df["forward_return"] = df["future_close_4h"] / df["close"] - 1
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=base_features + sr_features + [
    "target_up", "forward_return", "future_close_4h"
]).reset_index(drop=True)

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

start = int(len(df) * INITIAL_TRAIN_FRACTION)
if start >= len(df) - 1:
    raise ValueError("Not enough rows for the test period.")

records = []
models = {"baseline_features": None, "support_resistance": None}
last_fit = -REFIT_EVERY

print("Running walk-forward predictions...")
print("Rows after feature preparation:", len(df))
print("Initial training fraction:", INITIAL_TRAIN_FRACTION)
print("Refit interval:", REFIT_EVERY, "test predictions")

for i in range(start, len(df)):
    now = df.loc[i, "datetime"]
    eligible = df.iloc[:i]
    eligible = eligible[eligible["target_time"] < now]
    eligible = eligible.dropna(subset=["target_up"])

    if len(eligible) < 100 or eligible["target_up"].nunique() < 2:
        continue

    if i == start or i - last_fit >= REFIT_EVERY:
        for name, features in [
            ("baseline_features", base_features),
            ("support_resistance", sr_features)
        ]:
            model = make_model()
            model.fit(eligible[features], eligible["target_up"].astype(int))
            models[name] = model
        last_fit = i

    row = df.iloc[[i]]
    rec = {
        "datetime": now,
        "target_time": df.loc[i, "target_time"],
        "close": df.loc[i, "close"],
        "future_close_4h": df.loc[i, "future_close_4h"],
        "target_up": int(df.loc[i, "target_up"]),
        "forward_return": df.loc[i, "forward_return"],
        "training_rows": len(eligible),
        "training_up_rate": eligible["target_up"].mean()
    }
    for name, features in [
        ("baseline_features", base_features),
        ("support_resistance", sr_features)
    ]:
        rec["prob_up_" + name] = models[name].predict_proba(row[features])[0, 1]
    records.append(rec)

pred = pd.DataFrame(records)
if pred.empty:
    raise RuntimeError("No predictions generated.")

# Same non-overlapping four-hour entry schedule for both models.
pred = pred.sort_values("datetime").reset_index(drop=True)
selected = []
next_allowed = None
for _, row in pred.iterrows():
    if next_allowed is None or row["datetime"] >= next_allowed:
        selected.append(row)
        next_allowed = row["target_time"]
test = pd.DataFrame(selected).reset_index(drop=True)

if test.empty:
    raise RuntimeError("No non-overlapping test rows available.")

if not (
    test["datetime"].iloc[1:].reset_index(drop=True)
    >= test["target_time"].iloc[:-1].reset_index(drop=True)
).all():
    raise RuntimeError("Overlapping holding periods detected.")

def metrics(actual, probabilities, baseline_rate):
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    y = np.asarray(actual).astype(int)
    b = np.clip(np.asarray(baseline_rate), 1e-6, 1 - 1e-6)
    return {
        "accuracy_at_0.50_pct": accuracy_score(y, p >= 0.5) * 100,
        "baseline_accuracy_at_0.50_pct": accuracy_score(y, b >= 0.5) * 100,
        "brier_score": brier_score_loss(y, p),
        "baseline_brier_score": brier_score_loss(y, b),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "baseline_log_loss": log_loss(y, b, labels=[0, 1])
    }

summaries = []
for name in ["baseline_features", "support_resistance"]:
    prob = test["prob_up_" + name].to_numpy()
    y = test["target_up"].to_numpy().astype(int)
    m = metrics(y, prob, test["training_up_rate"])
    positions = np.where(prob >= THRESHOLD, 1,
                         np.where(prob <= 1 - THRESHOLD, -1, 0))
    m["experiment"] = name
    m["periods"] = len(test)
    m["active_trades"] = int((positions != 0).sum())

    for cost in COSTS:
        equity = INITIAL_CAPITAL
        peak = equity
        max_dd = 0.0
        for pos, ret in zip(positions, test["forward_return"].to_numpy()):
            start_equity = equity
            equity = start_equity * (1 + pos * ret)
            if pos != 0:
                equity -= start_equity * cost
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1)
        m[f"return_cost_{cost*100:.2f}pct"] = (
            equity / INITIAL_CAPITAL - 1
        ) * 100
        if abs(cost - 0.001) < 1e-10:
            m["max_drawdown_at_0.10pct_cost_pct"] = max_dd * 100
            m["final_equity_at_0.10pct_cost"] = equity
    summaries.append(m)

test.to_csv(OUTPUT, index=False)
result = pd.DataFrame(summaries)
result.to_csv(SUMMARY, index=False)

print("\n=== SUPPORT/RESISTANCE WALK-FORWARD TEST ===")
print("Test start:", test["datetime"].iloc[0])
print("Last target:", test["target_time"].iloc[-1])
print("Non-overlapping periods:", len(test))
print("Threshold: long >= 0.55, short <= 0.45, otherwise flat")
print("\n=== MODEL COMPARISON ===")
print(result[[
    "experiment", "periods", "active_trades",
    "accuracy_at_0.50_pct", "baseline_accuracy_at_0.50_pct",
    "brier_score", "baseline_brier_score",
    "log_loss", "baseline_log_loss",
    "return_cost_0.00pct", "return_cost_0.05pct",
    "return_cost_0.10pct", "max_drawdown_at_0.10pct_cost_pct"
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved predictions:", OUTPUT.name)
print("Saved comparison:", SUMMARY.name)
print("Research only. This is a historical test, not a live signal.")
