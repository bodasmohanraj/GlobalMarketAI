from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "gold_data.csv"
CLEAN_FILE = BASE_DIR / "gold_features_clean.csv"
LOG_FILE = BASE_DIR / "paper_trading_log.csv"

THRESHOLD = 0.55
MIN_TRAIN_ROWS = 500

MODEL_CONFIG = {
    "algorithm": "StandardScaler + LogisticRegression",
    "max_iter": 2000,
    "random_state": 42,
}

FEATURE_SETS = {
    "Full model": [
        "return_1h", "return_4h", "return_12h",
        "sma_5_gap", "sma_20_gap", "trend_5_20",
        "volatility_10", "volatility_20", "rsi_14",
    ],
    "Trend and momentum": [
        "return_1h", "return_4h", "return_12h",
        "sma_5_gap", "sma_20_gap", "trend_5_20",
    ],
    "Volatility and RSI": [
        "volatility_10", "volatility_20", "rsi_14",
    ],
}

LOG_COLUMNS = [
    "model", "prediction_time", "target_time",
    "prediction_close", "probability_up", "probability_down",
    "baseline_up_probability", "signal",
    "actual_close", "actual_up", "forward_return_pct",
    "resolved",
    "model_version", "training_cutoff_time", "training_rows",
    "feature_set", "prediction_created_at",
]


def read_raw():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError("gold_data.csv needs datetime and close columns.")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["datetime", "close"])
    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime", keep="last")
    return df.reset_index(drop=True)


def exact_timestamp_return(times, close, hours):
    lookup = pd.Series(close.to_numpy(), index=pd.DatetimeIndex(times))
    prior_times = times - pd.Timedelta(hours=hours)
    prior_close = prior_times.map(lookup)
    return (close / prior_close - 1).where(prior_close.notna())


def build_features(raw):
    df = raw.copy()
    times = df["datetime"]
    close = df["close"]
    gap_hours = times.diff().dt.total_seconds() / 3600

    df["return_1h"] = close.pct_change().where(gap_hours == 1)
    df["return_4h"] = exact_timestamp_return(times, close, 4)
    df["return_12h"] = exact_timestamp_return(times, close, 12)

    sma5 = close.rolling(5, min_periods=5).mean()
    sma20 = close.rolling(20, min_periods=20).mean()

    df["sma_5_gap"] = close / sma5 - 1
    df["sma_20_gap"] = close / sma20 - 1
    df["trend_5_20"] = sma5 / sma20 - 1

    df["volatility_10"] = df["return_1h"].rolling(
        10, min_periods=8
    ).std()
    df["volatility_20"] = df["return_1h"].rolling(
        20, min_periods=12
    ).std()

    consecutive = gap_hours == 1
    delta = close.diff().where(consecutive)
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(14, min_periods=10).mean()
    avg_loss = losses.rolling(14, min_periods=10).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    df["rsi_14"] = rsi
    return df


def validate_features(features, clean):
    feature_names = sorted(set().union(*FEATURE_SETS.values()))

    clean = clean.copy()
    clean["datetime"] = pd.to_datetime(
        clean["datetime"], utc=True, errors="coerce"
    )

    for col in feature_names:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    calculated = features[["datetime"] + feature_names]
    merged = clean[["datetime"] + feature_names].merge(
        calculated,
        on="datetime",
        how="inner",
        suffixes=("_clean", "_tracker"),
    )

    if merged.empty:
        raise ValueError("No overlapping timestamps for feature validation.")

    mismatches = []
    for col in feature_names:
        left = merged[f"{col}_clean"].to_numpy(dtype=float)
        right = merged[f"{col}_tracker"].to_numpy(dtype=float)

        equal = np.isclose(
            left, right, rtol=1e-7, atol=1e-9, equal_nan=True
        )
        if not equal.all():
            mismatches.append(f"{col}: {(~equal).sum()} mismatches")

    print(f"\nFeature validation: {len(merged)} overlapping timestamps.")

    if mismatches:
        raise ValueError(
            "Feature validation failed. No prediction will be added:\n"
            + "\n".join(mismatches)
        )

    print("PASS: tracker features match the saved clean features.")


def model_version(features):
    """Hash the model configuration and feature definition."""
    payload = {
        "config": MODEL_CONFIG,
        "features": features,
        "sklearn_version": sklearn.__version__,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=MODEL_CONFIG["max_iter"],
            random_state=MODEL_CONFIG["random_state"],
        )),
    ])


def load_log():
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    log = pd.read_csv(LOG_FILE)

    # Add new columns in memory while preserving old records.
    for col in LOG_COLUMNS:
        if col not in log.columns:
            log[col] = np.nan

    if log.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    log["prediction_time"] = pd.to_datetime(
        log["prediction_time"], utc=True, errors="coerce"
    )
    log["target_time"] = pd.to_datetime(
        log["target_time"], utc=True, errors="coerce"
    )

    return log[LOG_COLUMNS]


def resolve_predictions(log, raw):
    """Resolve only when the exact target timestamp is available."""
    if log.empty:
        return log

    price_lookup = dict(zip(raw["datetime"], raw["close"]))

    for i, row in log.iterrows():
        resolved = str(row["resolved"]).strip().lower() in {
            "true", "1", "yes"
        }
        if resolved:
            continue

        prediction_time = row["prediction_time"]
        target_time = row["target_time"]

        if pd.isna(prediction_time) or pd.isna(target_time):
            continue
        if target_time <= prediction_time:
            continue
        if target_time not in price_lookup:
            continue

        actual_close = float(price_lookup[target_time])
        original_close = float(row["prediction_close"])

        log.at[i, "actual_close"] = actual_close
        log.at[i, "actual_up"] = int(actual_close > original_close)
        log.at[i, "forward_return_pct"] = (
            actual_close / original_close - 1
        ) * 100
        log.at[i, "resolved"] = True

    return log


def generate_predictions(features, clean, log):
    latest = features.iloc[-1]
    prediction_time = latest["datetime"]

    # Normalize timestamps so comparisons work even if the CSV used strings.
    if not log.empty:
        log_times = pd.to_datetime(
            log["prediction_time"], utc=True, errors="coerce"
        )
        same_candle = log_times == prediction_time
        existing_models = set(
            log.loc[same_candle, "model"].dropna().astype(str).tolist()
        )
    else:
        existing_models = set()

    expected_models = set(FEATURE_SETS.keys())
    missing_models = expected_models - existing_models

    # If all expected models already exist for this candle, do not duplicate.
    if not missing_models:
        print(
            f"\nPredictions already exist for {prediction_time}. "
            "No duplicates added."
        )
        return log

    if existing_models:
        print(
            f"\nSome predictions already exist for {prediction_time}. "
            f"Only missing models will be added: {sorted(missing_models)}"
        )

    all_features = sorted(set().union(*FEATURE_SETS.values()))

    clean = clean.copy()
    clean["datetime"] = pd.to_datetime(
        clean["datetime"], utc=True, errors="coerce"
    )
    clean["target_time"] = pd.to_datetime(
        clean["target_time"], utc=True, errors="coerce"
    )

    for col in all_features + ["target_up"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # Only use training labels whose four-hour outcomes were known
    # strictly before the new prediction timestamp.
    train = clean[
        clean["target_time"] < prediction_time
    ].dropna(
        subset=["target_time", "target_up"] + all_features
    ).copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough eligible training rows: {len(train)}")

    if train["target_up"].nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    training_cutoff = train["target_time"].max()
    created_at = datetime.now(timezone.utc).isoformat()
    new_records = []

    for model_name, columns in FEATURE_SETS.items():
        if model_name not in missing_models:
            continue

        if latest[columns].isna().any():
            missing = latest[columns][latest[columns].isna()].index.tolist()
            raise ValueError(
                f"Latest candle has missing features for {model_name}: "
                f"{missing}"
            )

        model = make_model()
        model.fit(train[columns], train["target_up"])

        classifier = model.named_steps["classifier"]
        positive_index = list(classifier.classes_).index(1)

        probability_up = float(
            model.predict_proba(latest[columns].to_frame().T)[0][
                positive_index
            ]
        )

        if probability_up >= THRESHOLD:
            signal = "PAPER_LONG"
        elif probability_up <= 1 - THRESHOLD:
            signal = "PAPER_SHORT"
        else:
            signal = "NO_CLEAR_SIGNAL"

        new_records.append({
            "model": model_name,
            "prediction_time": prediction_time,
            "target_time": prediction_time + pd.Timedelta(hours=4),
            "prediction_close": float(latest["close"]),
            "probability_up": probability_up,
            "probability_down": 1 - probability_up,
            "baseline_up_probability": float(train["target_up"].mean()),
            "signal": signal,
            "actual_close": np.nan,
            "actual_up": np.nan,
            "forward_return_pct": np.nan,
            "resolved": False,
            "model_version": model_version(columns),
            "training_cutoff_time": training_cutoff,
            "training_rows": len(train),
            "feature_set": ",".join(columns),
            "prediction_created_at": created_at,
        })

    if not new_records:
        return log

    return pd.concat(
        [log, pd.DataFrame(new_records)],
        ignore_index=True,
    )[LOG_COLUMNS]


def show_report(log):
    if log.empty:
        print("No paper predictions logged.")
        return

    resolved_flags = log["resolved"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    latest_time = log["prediction_time"].max()
    latest_rows = log[log["prediction_time"] == latest_time]

    print("\nLATEST PREDICTIONS")
    print("-" * 78)

    for _, row in latest_rows.iterrows():
        print(
            f"{row['model']:<24} "
            f"Up {float(row['probability_up']):.1%} | "
            f"Down {float(row['probability_down']):.1%} | "
            f"{row['signal']}"
        )

    print("\nTRACKING STATUS")
    print("-" * 78)
    print(f"Total model predictions: {len(log)}")
    print(f"Resolved: {int(resolved_flags.sum())}")
    print(f"Pending: {int((~resolved_flags).sum())}")

    print("\nAUDIT FIELDS")
    print("-" * 78)
    print("New records include model_version, training_cutoff_time,")
    print("training_rows, feature_set, and prediction_created_at.")

    old_rows = log["model_version"].isna().sum()
    if old_rows:
        print(
            f"{old_rows} older records have no audit metadata. "
            "They were preserved, not reconstructed."
        )


def main():
    print("=" * 78)
    print("GOLD AI - AUDITABLE PAPER TRACKER")
    print("Research only - no real orders")
    print("=" * 78)

    raw = read_raw()
    features = build_features(raw)

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CLEAN_FILE}")

    clean = pd.read_csv(CLEAN_FILE)
    validate_features(features, clean)

    log = load_log()
    log = resolve_predictions(log, raw)
    log = generate_predictions(features, clean, log)
    log = resolve_predictions(log, raw)

    log = log.sort_values(
        ["prediction_time", "model"]
    ).reset_index(drop=True)

    log.to_csv(LOG_FILE, index=False)

    print(f"\nLatest candle: {raw['datetime'].iloc[-1]}")
    print(f"Latest close: {raw['close'].iloc[-1]:.2f}")
    print(f"Saved log: {LOG_FILE}")

    show_report(log)


if __name__ == "__main__":
    main()
