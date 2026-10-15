from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_features_clean.csv"

THRESHOLDS = [0.52, 0.55, 0.60, 0.65]
COSTS = [0.0005, 0.0010, 0.0015]
MIN_TRAIN = 500
FOLDS = 5

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


def load_data():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip().str.lower()

    for col in ["datetime", "target_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    required = [
        "datetime", "target_time", "close",
        "future_close_4h", "target_up",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in ["close", "future_close_4h", "target_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for cols in FEATURE_SETS.values():
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "target_time", "close",
                          "future_close_4h", "target_up"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

    df = df[
        (df["target_time"] > df["datetime"])
        & df["target_up"].isin([0, 1])
    ].copy()

    expected = (df["future_close_4h"] > df["close"]).astype(int)
    if (expected != df["target_up"].astype(int)).any():
        raise ValueError(
            "Stored target_up disagrees with future_close_4h. "
            "Check target definitions before testing."
        )

    return df.reset_index(drop=True)


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000, random_state=42
        )),
    ])


def generate_predictions(df):
    start = int(len(df) * 0.50)
    edges = np.linspace(start, len(df), FOLDS + 1, dtype=int)
    results = []

    for fold in range(FOLDS):
        test = df.iloc[edges[fold]:edges[fold + 1]].copy()
        if test.empty:
            continue

        test_start = test["datetime"].min()

        for name, features in FEATURE_SETS.items():
            valid = df[features].notna().all(axis=1)
            train = df[
                (df["datetime"] < test_start)
                & (df["target_time"] <= test_start)
                & valid
                & df["target_up"].isin([0, 1])
            ]

            test_valid = test[
                test[features].notna().all(axis=1)
            ].copy()

            if len(train) < MIN_TRAIN:
                print(f"Skipped fold {fold + 1}, {name}: "
                      f"{len(train)} training rows")
                continue

            if train["target_up"].nunique() != 2:
                print(f"Skipped fold {fold + 1}, {name}: one class")
                continue

            model = make_model()
            model.fit(train[features], train["target_up"].astype(int))
            classes = list(model.named_steps["classifier"].classes_)
            pos = classes.index(1)
            probabilities = model.predict_proba(test_valid[features])[:, pos]

            out = test_valid[
                ["datetime", "target_time", "close",
                 "future_close_4h", "target_up"]
            ].copy()
            out["model"] = name
            out["fold"] = fold + 1
            out["training_rows"] = len(train)
            out["probability_up"] = probabilities
            out["forward_return"] = (
                out["future_close_4h"] / out["close"] - 1
            )
            results.append(out)

    if not results:
        raise ValueError("No predictions generated. Check dataset and features.")

    pred = pd.concat(results, ignore_index=True)
    if pred.duplicated(["model", "datetime"]).any():
        raise ValueError("Duplicate out-of-sample predictions detected.")

    return pred.sort_values(["model", "datetime"]).reset_index(drop=True)


def max_drawdown(equity):
    peak = equity.cummax()
    return float(((equity / peak) - 1).min() * 100)


def evaluate(pred):
    count_rows = []
    trade_rows = []

    for model_name, group in pred.groupby("model"):
        for threshold in THRESHOLDS:
            d = group.sort_values("datetime").copy()
            p = d["probability_up"]
            d["signal"] = np.select(
                [p >= threshold, p <= 1 - threshold],
                [1, -1], default=0
            )

            active = d[d["signal"] != 0].copy()
            accuracy = (
                ((active["signal"] == 1)
                 == (active["target_up"].astype(int) == 1)).mean() * 100
                if not active.empty else np.nan
            )

            count_rows.append({
                "model": model_name,
                "threshold": threshold,
                "oos_predictions": len(d),
                "long_signals": int((d["signal"] == 1).sum()),
                "short_signals": int((d["signal"] == -1).sum()),
                "no_signals": int((d["signal"] == 0).sum()),
                "directional_accuracy_pct": accuracy,
            })

            # One position at a time per model and threshold.
            for cost in COSTS:
                equity = 1.0
                equity_path = [equity]
                trade_returns = []
                next_entry = None

                for _, row in d.iterrows():
                    signal = int(row["signal"])
                    if signal == 0:
                        continue

                    entry = row["datetime"]
                    exit_time = row["target_time"]

                    if next_entry is not None and entry < next_entry:
                        continue

                    move = float(row["forward_return"])
                    if not np.isfinite(move):
                        continue

                    gross = move if signal == 1 else -move
                    net = (1 + gross) * (1 - cost) - 1
                    equity *= (1 + net)
                    equity_path.append(equity)
                    trade_returns.append(net)
                    next_entry = exit_time

                t = np.array(trade_returns)
                gains = t[t > 0].sum()
                losses = -t[t < 0].sum()
                pf = gains / losses if losses > 0 else np.nan

                trade_rows.append({
                    "model": model_name,
                    "threshold": threshold,
                    "round_trip_cost_pct": cost * 100,
                    "trades": len(t),
                    "net_return_pct": (equity - 1) * 100,
                    "win_rate_pct": (t > 0).mean() * 100 if len(t) else np.nan,
                    "profit_factor": pf,
                    "max_drawdown_pct": max_drawdown(
                        pd.Series(equity_path)
                    ),
                })

    return pd.DataFrame(count_rows), pd.DataFrame(trade_rows)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find {DATA}")

    df = load_data()
    print(f"Usable labeled rows: {len(df)}")
    print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print("Generating leakage-aware chronological predictions...")

    pred = generate_predictions(df)
    counts, trades = evaluate(pred)

    pred.to_csv(BASE / "threshold_oos_predictions.csv", index=False)
    counts.to_csv(BASE / "threshold_signal_counts.csv", index=False)
    trades.to_csv(BASE / "threshold_performance.csv", index=False)

    print("\nSIGNAL COUNTS AND DIRECTIONAL ACCURACY")
    print(counts.round(3).to_string(index=False))

    print("\nNON-OVERLAPPING PAPER PERFORMANCE")
    print(trades.round(3).to_string(index=False))

    print("\nSaved three threshold CSV files in:", BASE)
    print("Historical research only; not a live trading recommendation.")


if __name__ == "__main__":
    main()
