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
    brier_score_loss, log_loss, mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent
INPUT = BASE / "gold_5m_data.csv"
HORIZONS = {"15min": 15, "30min": 30, "1hour": 60, "4hour": 240}
FEATURES = [
    "ret_5m", "ret_15m", "ret_1h", "ret_4h",
    "sma_3_gap", "sma_12_gap", "sma_48_gap",
    "volatility_12", "rsi_14",
]
FOLDS = 3
INITIAL_TRAIN_FRACTION = 0.50
MIN_TRAIN_ROWS = 500


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    if "datetime" not in df.columns:
        raise ValueError("Missing datetime column.")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    price_col = next(
        (c for c in ["close", "Close", "Adj Close", "adj close"] if c in df),
        None,
    )
    if price_col is None:
        raise ValueError(f"No close-price column found: {df.columns.tolist()}")

    df["close"] = pd.to_numeric(df[price_col], errors="coerce")
    df = (
        df.dropna(subset=["datetime", "close"])
        [["datetime", "close"]]
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )
    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")
    return df


def make_features(df):
    x = df.set_index("datetime").copy()
    close = x["close"]

    # A new segment starts after a gap greater than 10 minutes.
    gaps = x.index.to_series().diff()
    segment = (gaps.isna() | (gaps > pd.Timedelta(minutes=10))).cumsum()

    # Exact past timestamp lookups; missing candles remain missing.
    for name, minutes in [
        ("ret_5m", 5), ("ret_15m", 15),
        ("ret_1h", 60), ("ret_4h", 240),
    ]:
        prior_times = x.index - pd.Timedelta(minutes=minutes)
        prior = close.reindex(prior_times)
        prior.index = x.index
        x[name] = close / prior - 1.0

    # Reset rolling features at segment boundaries.
    x["sma_3_gap"] = np.nan
    x["sma_12_gap"] = np.nan
    x["sma_48_gap"] = np.nan
    x["volatility_12"] = np.nan
    x["rsi_14"] = np.nan

    for _, positions in pd.Series(range(len(x)), index=x.index).groupby(segment):
        idx = positions.to_numpy()
        c = close.iloc[idx]

        for window in [3, 12, 48]:
            avg = c.rolling(window, min_periods=window).mean()
            x.iloc[idx, x.columns.get_loc(f"sma_{window}_gap")] = (
                c / avg - 1.0
            ).to_numpy()

        returns = c.pct_change()
        x.iloc[idx, x.columns.get_loc("volatility_12")] = (
            returns.rolling(12, min_periods=12).std()
        ).to_numpy()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi.loc[(loss == 0) & (gain > 0)] = 100.0
        rsi.loc[(loss == 0) & (gain == 0)] = 50.0

        x.iloc[idx, x.columns.get_loc("rsi_14")] = rsi.to_numpy()

    return x.reset_index()


def add_labels(features, minutes):
    """Join each candle to the price at its exact future timestamp."""
    out = features.copy()

    # Keep the original timestamps of the actual future prices.
    future = features[["datetime", "close"]].rename(
        columns={
            "datetime": "target_time",
            "close": "future_close",
        }
    )

    # For a candle at t, the required target is the candle at t + horizon.
    out["target_time"] = (
        out["datetime"] + pd.Timedelta(minutes=minutes)
    )

    # Match t + horizon to the actual candle timestamp, never t - horizon.
    out = out.merge(
        future,
        on="target_time",
        how="left",
        validate="many_to_one",
    )

    out["price_change"] = out["future_close"] - out["close"]
    out["target_up"] = np.where(
        out["future_close"].notna(),
        (out["price_change"] > 0).astype(int),
        np.nan,
    )

    return out


def pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def classification_metrics(y, p, decisions):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    decisions = np.asarray(decisions, dtype=int)
    return {
        "accuracy": accuracy_score(y, decisions),
        "balanced_accuracy": balanced_accuracy_score(y, decisions),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "predicted_up_rate": decisions.mean(),
        "actual_up_rate": y.mean(),
    }


def add_cls_result(results, horizon, fold, method, y, p, decisions, **meta):
    results.append({
        "horizon": horizon, "fold": fold, "task": "classification",
        "method": method, "rows": len(y), **meta,
        **classification_metrics(y, p, decisions),
    })


def add_reg_result(results, horizon, fold, method, actual, pred, **meta):
    results.append({
        "horizon": horizon, "fold": fold, "task": "regression",
        "method": method, "rows": len(actual), **meta,
        "mae": mean_absolute_error(actual, pred),
        "rmse": np.sqrt(mean_squared_error(actual, pred)),
        "mean_error": np.mean(pred - actual),
    })


def main():
    raw = load_data()
    features = make_features(raw)

    print("RIGOROUS MULTI-HORIZON VALIDATION — RESEARCH ONLY")
    print("Candles:", len(raw))
    print("First UTC:", raw.datetime.min())
    print("Last UTC:", raw.datetime.max())
    gaps = raw.datetime.diff().dropna()
    print("Intervals over 5 minutes:", int((gaps > pd.Timedelta(minutes=5)).sum()))

    fold_results, predictions, label_audits = [], [], []

    for horizon, minutes in HORIZONS.items():
        labeled = add_labels(features, minutes)
        complete = labeled.dropna(
            subset=FEATURES + ["target_up", "price_change", "target_time"]
        ).sort_values("datetime").reset_index(drop=True)

        exact = labeled["future_close"].notna()
        mismatch = (
            labeled.loc[exact, "target_time"].to_numpy()
            != (labeled.loc[exact, "datetime"] + pd.Timedelta(minutes=minutes)).to_numpy()
        ).sum()

        label_audits.append({
            "horizon": horizon,
            "minutes": minutes,
            "input_rows": len(labeled),
            "exact_future_matches": int(exact.sum()),
            "missing_future_matches": int((~exact).sum()),
            "complete_feature_label_rows": len(complete),
            "target_time_mismatches": int(mismatch),
        })

        if len(complete) < MIN_TRAIN_ROWS + 100:
            print(f"\n{horizon}: insufficient rows; skipped.")
            continue

        first_test = int(len(complete) * INITIAL_TRAIN_FRACTION)
        fold_size = (len(complete) - first_test) // FOLDS
        print(f"\n{horizon}: usable rows={len(complete)}")

        for fold_idx in range(FOLDS):
            start = first_test + fold_idx * fold_size
            end = len(complete) if fold_idx == FOLDS - 1 else start + fold_size
            test = complete.iloc[start:end].copy()
            test_start = test.datetime.min()

            train = complete[
                (complete.datetime < test_start)
                & (complete.target_time < test_start)
            ].copy()

            if len(train) < MIN_TRAIN_ROWS:
                print(f" Fold {fold_idx + 1}: skipped; train={len(train)}")
                continue

            Xtr, Xte = train[FEATURES], test[FEATURES]
            ytr = train.target_up.astype(int).to_numpy()
            yte = test.target_up.astype(int).to_numpy()
            ctr = train.price_change.to_numpy()
            cte = test.price_change.to_numpy()

            # All baselines below are determined using training data only.
            train_up_rate = float(ytr.mean())
            train_up_rate_smooth = float(
                (ytr.sum() + 0.5) / (len(ytr) + 1.0)
            )
            majority_class = int(train_up_rate >= 0.5)
            train_mean_change = float(ctr.mean())

            clf = pipeline(LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(Xtr, ytr)
            p_model = clf.predict_proba(Xte)[:, 1]

            # Momentum decisions use only the current 15-minute return.
            momentum_decision = (test.ret_15m.to_numpy() > 0).astype(int)
            momentum_p = np.where(momentum_decision == 1, 0.75, 0.25)

            model_decision = (p_model >= 0.5).astype(int)
            majority_p = np.full(len(test), train_up_rate_smooth)
            majority_decision = np.full(len(test), majority_class, dtype=int)
            prevalence_p = np.full(len(test), train_up_rate_smooth)
            prevalence_decision = (prevalence_p >= 0.5).astype(int)
            coinflip_p = np.full(len(test), 0.5)
            coinflip_decision = np.zeros(len(test), dtype=int)

            common = {
                "train_rows": len(train),
                "train_end": train.datetime.max(),
                "test_start": test.datetime.min(),
                "test_end": test.datetime.max(),
                "train_up_rate": train_up_rate,
                "train_up_rate_smoothed": train_up_rate_smooth,
                "train_mean_change": train_mean_change,
                "train_majority_class": majority_class,
            }

            classifiers = [
                ("model_logistic", p_model, model_decision),
                ("baseline_majority", majority_p, majority_decision),
                ("baseline_training_prevalence", prevalence_p, prevalence_decision),
                ("baseline_momentum_15m", momentum_p, momentum_decision),
                ("baseline_probability_50pct", coinflip_p, coinflip_decision),
            ]
            for method, prob, decision in classifiers:
                add_cls_result(
                    fold_results, horizon, fold_idx + 1, method,
                    yte, prob, decision, **common
                )

            reg = pipeline(Ridge(alpha=10.0))
            reg.fit(Xtr, ctr)
            pred_model = reg.predict(Xte)
            pred_zero = np.zeros(len(test))
            pred_mean = np.full(len(test), train_mean_change)

            for method, pred in [
                ("model_ridge", pred_model),
                ("baseline_zero_change", pred_zero),
                ("baseline_training_mean_change", pred_mean),
            ]:
                add_reg_result(
                    fold_results, horizon, fold_idx + 1,
                    method, cte, pred, **common
                )

            for i, (_, r) in enumerate(test.iterrows()):
                predictions.append({
                    "horizon": horizon, "fold": fold_idx + 1,
                    "datetime": r.datetime, "target_time": r.target_time,
                    "close": r.close, "future_close": r.future_close,
                    "actual_up": int(yte[i]),
                    "actual_change": cte[i],
                    "actual_change_pct": cte[i] / r.close,
                    "p_up_model": p_model[i],
                    "decision_model": model_decision[i],
                    "p_up_majority": majority_p[i],
                    "decision_majority": majority_decision[i],
                    "p_up_prevalence": prevalence_p[i],
                    "decision_prevalence": prevalence_decision[i],
                    "p_up_momentum_illustrative": momentum_p[i],
                    "decision_momentum": momentum_decision[i],
                    "p_up_coinflip": 0.5,
                    "decision_coinflip_convention": 0,
                    "pred_change_model": pred_model[i],
                    "pred_change_zero": 0.0,
                    "pred_change_train_mean": train_mean_change,
                    **common,
                })

            print(
                f" Fold {fold_idx + 1}: train={len(train)}, test={len(test)}, "
                f"UP rate in train={train_up_rate:.3f}, "
                f"model balanced accuracy="
                f"{balanced_accuracy_score(yte, model_decision):.3f}, "
                f"model MAE={mean_absolute_error(cte, pred_model):.4f}, "
                f"zero MAE={mean_absolute_error(cte, pred_zero):.4f}"
            )

    folds_df = pd.DataFrame(fold_results)
    preds_df = pd.DataFrame(predictions)
    labels_df = pd.DataFrame(label_audits)

    folds_df.to_csv(BASE / "validation_audit_fold_metrics_v2.csv", index=False)
    preds_df.to_csv(BASE / "validation_audit_oos_predictions_v2.csv", index=False)
    labels_df.to_csv(BASE / "validation_audit_label_checks_v2.csv", index=False)

    summary = []
    if not preds_df.empty:
        for horizon, part in preds_df.groupby("horizon", sort=False):
            y = part.actual_up.astype(int).to_numpy()

            classification_methods = [
                ("model_logistic", "p_up_model", "decision_model"),
                ("baseline_majority", "p_up_majority", "decision_majority"),
                ("baseline_training_prevalence", "p_up_prevalence", "decision_prevalence"),
                ("baseline_momentum_15m_illustrative", "p_up_momentum_illustrative", "decision_momentum"),
                ("baseline_probability_50pct", "p_up_coinflip", "decision_coinflip_convention"),
            ]
            for method, pcol, dcol in classification_methods:
                summary.append({
                    "horizon": horizon, "task": "classification",
                    "method": method, **classification_metrics(
                        y, part[pcol].to_numpy(),
                        part[dcol].astype(int).to_numpy()
                    ),
                    "rows": len(part),
                })

            actual = part.actual_change.to_numpy()
            for method, col in [
                ("model_ridge", "pred_change_model"),
                ("baseline_zero_change", "pred_change_zero"),
                ("baseline_training_mean_change", "pred_change_train_mean"),
            ]:
                pred = part[col].to_numpy()
                summary.append({
                    "horizon": horizon, "task": "regression",
                    "method": method, "rows": len(part),
                    "mae": mean_absolute_error(actual, pred),
                    "rmse": np.sqrt(mean_squared_error(actual, pred)),
                    "mean_error": np.mean(pred - actual),
                })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(BASE / "validation_audit_summary_v2.csv", index=False)

    print("\nPOOLED OUT-OF-SAMPLE SUMMARY — V2")
    print(summary_df.round(4).to_string(index=False))
    print("\nLABEL AUDIT — V2")
    print(labels_df.to_string(index=False))
    print("\nNew files saved:")
    for name in [
        "validation_audit_fold_metrics_v2.csv",
        "validation_audit_oos_predictions_v2.csv",
        "validation_audit_label_checks_v2.csv",
        "validation_audit_summary_v2.csv",
    ]:
        print(BASE / name)

    print(
        "\nNotes: momentum probabilities 0.75/0.25 are illustrative, not calibrated. "
        "The 50% baseline's all-DOWN class decision is only a reporting convention; "
        "its probability scores are the meaningful comparison. Research only."
    )


if __name__ == "__main__":
    main()
