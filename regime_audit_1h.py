import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss

DATA_FILE = Path("gold_5m_data.csv")
PRED_FILE = Path("validation_audit_oos_predictions_v2.csv")
OUT_FILE = Path("regime_audit_1h.csv")

# Load five-minute gold candles.
prices = pd.read_csv(DATA_FILE)
prices.columns = [c.strip().lower().replace(" ", "_") for c in prices.columns]

if "datetime" not in prices.columns or "close" not in prices.columns:
    raise ValueError(f"Expected datetime and close columns. Found: {prices.columns.tolist()}")

prices["datetime"] = pd.to_datetime(prices["datetime"], utc=True)
prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
prices = (
    prices.dropna(subset=["datetime", "close"])
    .sort_values("datetime")
    .drop_duplicates("datetime")
    .reset_index(drop=True)
)

# Calculate trailing 1-hour trend from four observed 5-minute candles.
# Shift by one candle so the regime uses information available before
# the prediction timestamp, not the current candle's final close.
prices["trend_1h_pct"] = prices["close"].pct_change(12).shift(1) * 100

# Use a fixed, transparent threshold. This is a diagnostic grouping,
# not an optimized trading threshold.
THRESHOLD_PCT = 0.10

prices["regime"] = np.select(
    [
        prices["trend_1h_pct"] > THRESHOLD_PCT,
        prices["trend_1h_pct"] < -THRESHOLD_PCT,
    ],
    ["trending_up", "trending_down"],
    default="sideways",
)

# Load out-of-sample predictions.
pred = pd.read_csv(PRED_FILE)
pred = pred[pred["horizon"].astype(str).eq("1hour")].copy()
pred["datetime"] = pd.to_datetime(pred["datetime"], utc=True)

for col in ["actual_up", "p_up_model", "decision_model"]:
    pred[col] = pd.to_numeric(pred[col], errors="coerce")

# Merge only the past-derived regime onto each prediction timestamp.
df = pred.merge(
    prices[["datetime", "trend_1h_pct", "regime"]],
    on="datetime",
    how="left",
    validate="many_to_one",
)

df = df.dropna(subset=["actual_up", "p_up_model", "regime"]).copy()
df["actual_up"] = df["actual_up"].astype(int)
df["p_up_model"] = df["p_up_model"].clip(0, 1)
df["decision_model"] = df["decision_model"].astype(int)

rows = []
for regime, g in df.groupby("regime", sort=True):
    y = g["actual_up"].to_numpy()
    p = g["p_up_model"].to_numpy()
    decision = g["decision_model"].to_numpy()

    rows.append({
        "horizon": "1hour",
        "regime": regime,
        "rows": len(g),
        "actual_up_rate": y.mean(),
        "predicted_up_rate": decision.mean(),
        "balanced_accuracy": balanced_accuracy_score(y, decision)
            if len(np.unique(y)) == 2 else np.nan,
        "roc_auc": roc_auc_score(y, p)
            if len(np.unique(y)) == 2 else np.nan,
        "brier": brier_score_loss(y, p),
        "mean_trailing_trend_pct": g["trend_1h_pct"].mean(),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_FILE, index=False)

print("=" * 68)
print("1-HOUR MARKET-REGIME AUDIT — RESEARCH ONLY")
print("=" * 68)
print("Matched prediction rows:", len(df))
print("Rows without a matching regime:", len(pred) - len(df))
print("Trend threshold:", THRESHOLD_PCT, "%")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:", OUT_FILE.resolve())
print("Important: regimes are diagnostic groups, not proven trading signals.")
