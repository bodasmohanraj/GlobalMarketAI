from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_5m_data.csv"
PRED_OUT = BASE / "multi_horizon_predictions.csv"
SUMMARY_OUT = BASE / "multi_horizon_summary.csv"
FOLD_OUT = BASE / "multi_horizon_by_fold.csv"

HORIZONS = {
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "4hour": 240
}

FEATURES = [
    "ret_5m", "ret_15m", "ret_1h", "ret_4h",
    "sma_3_gap", "sma_12_gap", "sma_48_gap",
    "volatility_12", "rsi_14"
]


def make_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=2000))
    ])


def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ])


def build_data():
    df = pd.read_csv(DATA)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    required = ["datetime", "close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (
        df.dropna(subset=["datetime", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

    # Use timestamp lookups for returns, rather than assuming every
    # adjacent row is exactly five minutes apart.
    indexed = df.set_index("datetime")
    close = indexed["close"]

    for name, minutes in [
        ("ret_5m", 5), ("ret_15m", 15),
        ("ret_1h", 60), ("ret_4h", 240)
    ]:
        prior = close.reindex(indexed.index - pd.Timedelta(minutes=minutes))
        prior.index = indexed.index
        indexed[name] = close / prior - 1

    # Rolling indicators use recent observed candles.
    indexed["sma_3_gap"] = close / close.rolling(3, min_periods=3).mean() - 1
    indexed["sma_12_gap"] = close / close.rolling(12, min_periods=12).mean() - 1
    indexed["sma_48_gap"] = close / close.rolling(48, min_periods=48).mean() - 1

    five_minute_returns = indexed["ret_5m"]
    indexed["volatility_12"] = five_minute_returns.rolling(
        12, min_periods=12
    ).std()

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    indexed["rsi_14"] = 100 - (100 / (1 + rs))

    # Avoid treating rolling observations across long gaps as continuous.
    times = pd.Series(indexed.index, index=indexed.index)
    gaps = times.diff()
    for col in ["sma_3_gap", "sma_12_gap", "sma_48_gap", "volatility_12", "rsi_14"]:
        indexed.loc[gaps > pd.Timedelta(minutes=10), col] = np.nan

    indexed = indexed.reset_index()
    indexed = indexed.rename(columns={"index": "datetime"})

    # Exact future timestamp labels: missing target candles are not fabricated.
    lookup = indexed.set_index("datetime")["close"]
    for name, minutes in HORIZONS.items():
        future_times = indexed["datetime"] + pd.Timedelta(minutes=minutes)
        future_close = future_times.map(lookup)
        indexed[f"future_close_{name}"] = future_close
        indexed[f"target_time_{name}"] = future_times
        indexed[f"change_{name}"] = future_close - indexed["close"]
        indexed[f"return_{name}"] = future_close / indexed["close"] - 1
        indexed[f"direction_{name}"] = np.where(
            future_close.notna(),
            (future_close > indexed["close"]).astype(float),
            np.nan
        )

    return indexed


def main():
    df = build_data()
    print("\nMULTI-HORIZON GOLD FORECAST — RESEARCH ONLY")
    print("Raw candles:", len(df))
    print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
    print("Only exact future timestamps are used for labels.")
    print("The dataset is short; results are exploratory.")

    all_predictions = []
    fold_rows = []

    # Keep the earliest 50% as initial history; test on later periods.
    for horizon_name, minutes in HORIZONS.items():
        direction_col = f"direction_{horizon_name}"
        target_time_col = f"target_time_{horizon_name}"
        change_col = f"change_{horizon_name}"

        usable = df.dropna(
            subset=FEATURES + [direction_col, change_col, target_time_col]
        ).copy()
        usable[direction_col] = usable[direction_col].astype(int)

        print(f"\n{horizon_name}: usable labeled rows = {len(usable)}")
        if len(usable) < 100:
            print("Skipping: too few usable rows for a meaningful comparison.")
            continue

        first_test = int(len(usable) * 0.50)
        folds = np.array_split(np.arange(first_test, len(usable)), 3)

        for fold_num, indices in enumerate(folds, start=1):
            if len(indices) == 0:
                continue

            test = usable.iloc[indices].copy()
            test_start = test["datetime"].min()

            train = usable[
                (usable["datetime"] < test_start) &
                (usable[target_time_col] < test_start)
            ].copy()

            if len(train) < 50 or test[direction_col].nunique() < 2:
                print(f"  Fold {fold_num}: skipped (insufficient data/classes)")
                continue

            X_train = train[FEATURES]
            X_test = test[FEATURES]
            y_train = train[direction_col].to_numpy()
            y_test = test[direction_col].to_numpy()
            actual_change = test[change_col].to_numpy()

            classifier = make_classifier()
            classifier.fit(X_train, y_train)
            prob = classifier.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)

            regressor = make_regressor()
            regressor.fit(X_train, train[change_col])
            change_pred = regressor.predict(X_test)

            # Simple benchmarks: training-majority direction and zero price change.
            majority = int(pd.Series(y_train).mode().iloc[0])
            base_direction = np.full(len(test), majority)
            zero_change = np.zeros(len(test))

            for method, p, probability in [
                ("logistic", pred, prob),
                ("majority_baseline", base_direction, None)
            ]:
                row = {
                    "horizon": horizon_name, "fold": fold_num,
                    "method": method, "train_rows": len(train),
                    "test_rows": len(test), "test_start": test_start,
                    "test_end": test["datetime"].max(),
                    "accuracy": accuracy_score(y_test, p),
                    "balanced_accuracy": balanced_accuracy_score(y_test, p),
                    "predicted_up_rate": np.mean(p),
                    "actual_up_rate": np.mean(y_test)
                }
                if probability is not None:
                    row.update({
                        "roc_auc": roc_auc_score(y_test, probability),
                        "brier_score": brier_score_loss(y_test, probability),
                        "log_loss": log_loss(y_test, probability, labels=[0, 1])
                    })
                else:
                    row.update({
                        "roc_auc": np.nan,
                        "brier_score": np.nan,
                        "log_loss": np.nan
                    })
                fold_rows.append(row)

            for method, prediction in [
                ("ridge_regression", change_pred),
                ("zero_change_baseline", zero_change)
            ]:
                fold_rows.append({
                    "horizon": horizon_name, "fold": fold_num,
                    "method": method, "train_rows": len(train),
                    "test_rows": len(test), "test_start": test_start,
                    "test_end": test["datetime"].max(),
                    "mae": mean_absolute_error(actual_change, prediction),
                    "rmse": np.sqrt(mean_squared_error(actual_change, prediction))
                })

            for i, (_, source) in enumerate(test.iterrows()):
                all_predictions.append({
                    "horizon": horizon_name, "fold": fold_num,
                    "datetime": source["datetime"],
                    "target_time": source[target_time_col],
                    "actual_direction": int(y_test[i]),
                    "predicted_direction": int(pred[i]),
                    "probability_up": float(prob[i]),
                    "actual_price_change": float(actual_change[i]),
                    "predicted_price_change": float(change_pred[i]),
                    "zero_change_prediction": 0.0
                })

            print(
                f"  Fold {fold_num}: train={len(train)}, test={len(test)}, "
                f"classifier balanced accuracy="
                f"{balanced_accuracy_score(y_test, pred):.3f}, "
                f"regression MAE={mean_absolute_error(actual_change, change_pred):.4f}"
            )

    predictions = pd.DataFrame(all_predictions)
    folds_df = pd.DataFrame(fold_rows)

    if predictions.empty:
        raise RuntimeError(
            "No predictions generated. Check exact future timestamp coverage."
        )

    predictions.to_csv(PRED_OUT, index=False)
    folds_df.to_csv(FOLD_OUT, index=False)

    print("\nCLASSIFICATION RESULTS BY HORIZON")
    for horizon_name, group in predictions.groupby("horizon"):
        y = group["actual_direction"].to_numpy()
        p = group["predicted_direction"].to_numpy()
        prob = group["probability_up"].to_numpy()
        print(
            f"{horizon_name:8s} rows={len(group):4d} "
            f"accuracy={accuracy_score(y,p):.3f} "
            f"balanced_accuracy={balanced_accuracy_score(y,p):.3f} "
            f"ROC-AUC={roc_auc_score(y,prob):.3f} "
            f"Brier={brier_score_loss(y,prob):.4f}"
        )

    print("\nREGRESSION RESULTS BY HORIZON")
    for horizon_name, group in predictions.groupby("horizon"):
        actual = group["actual_price_change"].to_numpy()
        predicted = group["predicted_price_change"].to_numpy()
        zero = np.zeros(len(group))
        print(
            f"{horizon_name:8s} rows={len(group):4d} "
            f"model_MAE={mean_absolute_error(actual,predicted):.4f} "
            f"zero_baseline_MAE={mean_absolute_error(actual,zero):.4f} "
            f"model_RMSE={np.sqrt(mean_squared_error(actual,predicted)):.4f} "
            f"zero_baseline_RMSE={np.sqrt(mean_squared_error(actual,zero)):.4f}"
        )

    print("\nSaved:")
    print(PRED_OUT)
    print(FOLD_OUT)
    print("Note: price changes are in the quoted gold futures price units.")
    print("Do not interpret these short-history experiments as trading signals.")


if __name__ == "__main__":
    main()
