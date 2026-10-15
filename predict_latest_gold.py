from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "gold_features_clean.csv"
LATEST_FILE = BASE_DIR / "gold_latest_features.csv"
OUTPUT_FILE = BASE_DIR / "latest_gold_prediction.csv"

from build_clean_features import FEATURES

MIN_PROBABILITY = 0.55


def main():
    print("=" * 60)
    print("GLOBAL MARKET AI - GOLD FORECAST")
    print("RESEARCH ONLY - NO AUTOMATIC TRADING")
    print("=" * 60)

    for path in [TRAIN_FILE, LATEST_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    training = pd.read_csv(TRAIN_FILE)
    latest = pd.read_csv(LATEST_FILE)

    required = FEATURES + ["target_up", "datetime"]
    missing = [c for c in required if c not in training.columns]
    if missing:
        raise ValueError(f"Training columns missing: {missing}")

    if latest.empty:
        raise ValueError("Latest feature file is empty.")

    missing = [c for c in FEATURES if c not in latest.columns]
    if missing:
        raise ValueError(f"Latest feature columns missing: {missing}")

    training = training.dropna(subset=FEATURES + ["target_up"]).copy()
    training["target_up"] = pd.to_numeric(
        training["target_up"], errors="coerce"
    )
    training = training.dropna(subset=["target_up"])
    training = training[training["target_up"].isin([0, 1])].copy()
    training = training.sort_values("datetime")

    if training["target_up"].nunique() != 2:
        raise ValueError("Training requires both target classes.")

    # Chronological holdout diagnostic: train on earlier rows,
    # evaluate on later rows. This is a diagnostic, not a guarantee.
    split = int(len(training) * 0.8)
    if split < 1 or len(training) - split < 1:
        raise ValueError("Not enough training rows for a holdout.")

    def make_model():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000, random_state=42
            )),
        ])

    check_model = make_model()
    check_model.fit(
        training.iloc[:split][FEATURES],
        training.iloc[:split]["target_up"].astype(int),
    )
    holdout = training.iloc[split:]
    holdout_probs = check_model.predict_proba(holdout[FEATURES])
    classes = list(check_model.named_steps["classifier"].classes_)
    up_idx = classes.index(1)
    holdout_pred = (holdout_probs[:, up_idx] >= 0.5).astype(int)
    baseline = max(
        holdout["target_up"].mean(),
        1 - holdout["target_up"].mean()
    )
    accuracy = (holdout_pred == holdout["target_up"].astype(int)).mean()

    print(f"\nTraining rows: {len(training)}")
    print(f"Chronological holdout rows: {len(holdout)}")
    print(f"Holdout accuracy: {accuracy:.3f}")
    print(f"Majority-class baseline: {baseline:.3f}")

    # Fit final model on all currently labeled historical rows.
    model = make_model()
    model.fit(training[FEATURES], training["target_up"].astype(int))

    latest["datetime"] = pd.to_datetime(
        latest["datetime"], utc=True, errors="coerce"
    )
    row = latest.iloc[[-1]].copy()
    missing_features = [
        f for f in FEATURES
        if pd.isna(row.iloc[0][f])
        or not np.isfinite(float(row.iloc[0][f]))
    ]

    result = {
        "datetime": row.iloc[0]["datetime"],
        "close": row.iloc[0].get("close", np.nan),
        "prob_up": np.nan,
        "prob_down": np.nan,
        "model_signal": "UNAVAILABLE",
        "holdout_accuracy": accuracy,
        "majority_baseline": baseline,
        "note": "Research only; forecasting edge not established.",
    }

    if missing_features:
        result["note"] = "No forecast: missing features: " + ", ".join(
            missing_features
        )
        print("\nNo forecast:", result["note"])
    else:
        probs = model.predict_proba(row[FEATURES])[0]
        classes = list(model.named_steps["classifier"].classes_)
        prob_up = float(probs[classes.index(1)])
        prob_down = 1.0 - prob_up

        if prob_up >= MIN_PROBABILITY:
            signal = "UPWARD BIAS"
        elif prob_up <= 1 - MIN_PROBABILITY:
            signal = "DOWNWARD BIAS"
        else:
            signal = "NO CLEAR SIGNAL"

        result.update({
            "prob_up": prob_up,
            "prob_down": prob_down,
            "model_signal": signal,
        })

        print("\nLATEST FEATURE TIME (UTC):", result["datetime"])
        print("Latest close:", result["close"])
        print(f"Model probability up:   {prob_up:.4f}")
        print(f"Model probability down: {prob_down:.4f}")
        print("Model signal:", signal)
        print("Probabilities may be uncalibrated; this is not a trade recommendation.")

    pd.DataFrame([result]).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved prediction to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
