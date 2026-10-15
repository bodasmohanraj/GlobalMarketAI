
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "gold_data.csv"
TRAIN_FILE = BASE_DIR / "gold_features_clean.csv"
OUTPUT_FILE = BASE_DIR / "latest_gold_prediction.csv"

FEATURES = [
    "return_1h",
    "return_4h",
    "return_12h",
    "sma_5_gap",
    "sma_20_gap",
    "trend_5_20",
    "volatility_10",
    "volatility_20",
    "rsi_14",
]

MIN_PROBABILITY = 0.55


# ============================================================
# FEATURE ENGINEERING
# Keep these definitions consistent with training.
# ============================================================

def calculate_features(raw):
    df = raw.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    df = (
        df.dropna(subset=["datetime", "close"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Detect genuinely consecutive hourly candles.
    consecutive_hour = (
        df["datetime"].diff() == pd.Timedelta(hours=1)
    )

    # One-hour return is valid only when candles are consecutive.
    return_1h = df["close"].pct_change()
    df["return_1h"] = return_1h.where(consecutive_hour)

    # Exact elapsed-time returns, not row shifts.
    close_by_time = pd.Series(
        df["close"].to_numpy(),
        index=df["datetime"],
    )

    for hours in [4, 12]:
        previous_time = df["datetime"] - pd.Timedelta(hours=hours)
        previous_close = previous_time.map(close_by_time)

        df[f"return_{hours}h"] = (
            df["close"] / previous_close - 1
        )

    # Moving averages use the latest observations.
    df["sma_5"] = df["close"].rolling(
        window=5, min_periods=5
    ).mean()

    df["sma_20"] = df["close"].rolling(
        window=20, min_periods=20
    ).mean()

    df["sma_5_gap"] = df["close"] / df["sma_5"] - 1
    df["sma_20_gap"] = df["close"] / df["sma_20"] - 1

    df["trend_5_20"] = (
        df["sma_5"] / df["sma_20"] - 1
    )

    # Volatility: rolling windows on valid one-hour returns.
    df["volatility_10"] = df["return_1h"].rolling(
        window=10, min_periods=8
    ).std()

    df["volatility_20"] = df["return_1h"].rolling(
        window=20, min_periods=12
    ).std()

    # RSI-14: only use changes between consecutive hourly candles.
    delta = df["close"].diff().where(consecutive_hour)

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=14, min_periods=10
    ).mean()

    avg_loss = loss.rolling(
        window=14, min_periods=10
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Handle edge cases for RSI.
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    no_losses = (avg_gain > 0) & (avg_loss == 0)

    df.loc[both_zero, "rsi_14"] = 50
    df.loc[no_losses, "rsi_14"] = 100

    return df


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def main():
    print("=" * 65)
    print("LATEST GOLD MODEL OUTPUT — RESEARCH ONLY")
    print("=" * 65)

    raw = load_csv(RAW_FILE)
    training = load_csv(TRAIN_FILE)

    required_training_columns = FEATURES + ["target_up", "datetime"]

    missing_columns = [
        col for col in required_training_columns
        if col not in training.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Training file is missing columns: {missing_columns}"
        )

    raw_features = calculate_features(raw)

    # Train only on historical rows with complete, known labels.
    training = training.dropna(
        subset=FEATURES + ["target_up"]
    ).copy()

    training["target_up"] = pd.to_numeric(
        training["target_up"], errors="coerce"
    )

    training = training.dropna(subset=["target_up"])
    training = training[
        training["target_up"].isin([0, 1])
    ]

    if training["target_up"].nunique() < 2:
        raise ValueError(
            "Training data must contain both target classes: 0 and 1."
        )

    # Same feature list; preprocessing is fitted on training data only.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=42,
        )),
    ])

    model.fit(
        training[FEATURES],
        training["target_up"].astype(int),
    )

    # Show the latest raw candles and feature availability.
    latest_raw = raw_features.tail(15)

    print("\nLATEST RAW CANDLES (UTC)")
    print(
        latest_raw[["datetime", "close"]]
        .tail(10)
        .to_string(index=False)
    )

    latest = raw_features.iloc[[-1]].copy()
    latest_time = latest.iloc[0]["datetime"]
    latest_close = float(latest.iloc[0]["close"])

    missing_features = [
        feature for feature in FEATURES
        if pd.isna(latest.iloc[0][feature])
        or not np.isfinite(float(latest.iloc[0][feature]))
    ]

    result = {
        "datetime": latest_time,
        "close": latest_close,
        "prob_up": np.nan,
        "prob_down": np.nan,
        "model_signal": "UNAVAILABLE",
        "missing_features": ", ".join(missing_features),
        "note": (
            "Research only; forecasting edge not established."
        ),
    }

    print("\nLATEST CANDLE FEATURE CHECK")
    print(f"Timestamp: {latest_time}")
    print(f"Close: {latest_close:.2f}")
    print(f"Missing required features: {missing_features or 'None'}")

    if missing_features:
        result["note"] = (
            "No prediction: required features are missing. "
            "Wait for sufficient valid hourly observations."
        )

        print("\nNO PREDICTION")
        print(result["note"])

    else:
        probabilities = model.predict_proba(
            latest[FEATURES]
        )[0]

        classes = list(model.named_steps["classifier"].classes_)
        up_index = classes.index(1)

        prob_up = float(probabilities[up_index])
        prob_down = 1.0 - prob_up

        if prob_up >= MIN_PROBABILITY:
            signal = "UPWARD BIAS"
        elif prob_up <= (1 - MIN_PROBABILITY):
            signal = "DOWNWARD BIAS"
        else:
            signal = "NO CLEAR SIGNAL"

        result.update({
            "prob_up": prob_up,
            "prob_down": prob_down,
            "model_signal": signal,
        })

        print("\nMODEL OUTPUT")
        print(f"Probability up:   {prob_up:.4f}")
        print(f"Probability down: {prob_down:.4f}")
        print(f"Model signal:     {signal}")
        print(
            "These probabilities are not necessarily calibrated "
            "and do not establish a profitable trading edge."
        )

    pd.DataFrame([result]).to_csv(
        OUTPUT_FILE, index=False
    )

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
