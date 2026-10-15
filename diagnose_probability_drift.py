import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "gold_features_clean.csv")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

FEATURES = [
    "return_1h", "return_4h", "return_12h",
    "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14"
]

df = df.dropna(subset=FEATURES + ["target_up", "datetime", "target_time"])
df = df.sort_values("datetime").reset_index(drop=True)

split = int(len(df) * 0.80)
test_start = df.iloc[split]["datetime"]

train = df[
    (df["datetime"] < test_start) &
    (df["target_time"] < test_start)
].copy()
test = df[df["datetime"] >= test_start].copy()

validation_split = int(len(train) * 0.80)
fit = train.iloc[:validation_split].copy()
validation = train.iloc[validation_split:].copy()

validation_start = validation["datetime"].min()
fit = fit[fit["target_time"] < validation_start].copy()

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

model = make_model()
model.fit(fit[FEATURES], fit["target_up"])
val_prob = model.predict_proba(validation[FEATURES])[:, 1]

model.fit(train[FEATURES], train["target_up"])
test_prob = model.predict_proba(test[FEATURES])[:, 1]

print("\nPROBABILITY DRIFT DIAGNOSTIC")
print("Validation period:", validation["datetime"].min(), "to", validation["datetime"].max())
print("Test period:", test["datetime"].min(), "to", test["datetime"].max())

for name, frame, probs in [
    ("FIT", fit, model.predict_proba(fit[FEATURES])[:, 1]),
    ("VALIDATION", validation, val_prob),
    ("FINAL TEST", test, test_prob)
]:
    print(f"\n{name}")
    print("Rows:", len(frame))
    print("Actual UP rate:", round(frame["target_up"].mean(), 4))
    print("Mean predicted UP probability:", round(float(np.mean(probs)), 4))
    print("Probability percentiles:",
          np.round(np.percentile(probs, [5, 25, 50, 75, 95]), 4))

print("\nFEATURE DISTRIBUTION SHIFT")
for col in FEATURES:
    a = validation[col].dropna()
    b = test[col].dropna()
    print(
        f"{col:15s}",
        "validation median =", round(a.median(), 6),
        "| test median =", round(b.median(), 6),
        "| validation std =", round(a.std(), 6),
        "| test std =", round(b.std(), 6)
    )

print("\nMONTHLY TEST RESULTS")
test = test.copy()
test["predicted_probability_up"] = test_prob
test["month"] = test["datetime"].dt.strftime("%Y-%m")
monthly = test.groupby("month").agg(
    rows=("target_up", "size"),
    actual_up_rate=("target_up", "mean"),
    mean_predicted_up_probability=("predicted_probability_up", "mean")
)
print(monthly.round(4).to_string())

print("\nDiagnostic complete. No files were modified.")
