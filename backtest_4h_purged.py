from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_features_clean.csv"
OUTPUT = BASE / "backtest_4h_purged_results.csv"

FEATURES = [
    "return_1h", "return_4h", "return_12h",
    "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14",
]
TRAIN_FRACTION = 0.80
THRESHOLD = 0.55
ROUND_TRIP_COST = 0.001
INITIAL_CAPITAL = 100_000

def main():
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True, errors="coerce")

    for col in FEATURES + ["close", "future_close_4h", "target_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [
        "datetime", "target_time", "close", "future_close_4h", "target_up"
    ]).sort_values("datetime").reset_index(drop=True)

    elapsed = (df["target_time"] - df["datetime"]).dt.total_seconds() / 3600
    df = df.loc[elapsed.eq(4)].copy()
    df["target_up"] = df["target_up"].astype(int)

    expected = (df["future_close_4h"] > df["close"]).astype(int)
    if not expected.equals(df["target_up"]):
        raise ValueError("target_up does not match future_close_4h.")

    # Establish the chronological boundary before purging.
    split = int(len(df) * TRAIN_FRACTION)
    test_start = df.iloc[split]["datetime"]

    # Purge training examples whose future labels reach the test period.
    train = df.iloc[:split].loc[
        df.iloc[:split]["target_time"] < test_start
    ].copy()
    test_all = df.iloc[split:].copy()

    if len(train) < 100 or len(test_all) < 100:
        raise ValueError("Insufficient train/test rows.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        )),
    ])
    model.fit(train[FEATURES], train["target_up"])

    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("Training data has no positive target class.")
    prob_up = model.predict_proba(test_all[FEATURES])[:, classes.index(1)]

    test_all["prob_up"] = prob_up
    test_all["predicted_class"] = (prob_up >= 0.5).astype(int)
    test_all["forward_return"] = (
        test_all["future_close_4h"] / test_all["close"] - 1
    )

    accuracy = accuracy_score(test_all["target_up"], test_all["predicted_class"])
    brier = brier_score_loss(test_all["target_up"], prob_up)
    ll = log_loss(test_all["target_up"], prob_up, labels=[0, 1])

    # Majority-class benchmark and constant-probability benchmark.
    majority_class = int(train["target_up"].mean() >= 0.5)
    baseline_accuracy = accuracy_score(
        test_all["target_up"], np.full(len(test_all), majority_class)
    )
    baseline_prob = float(train["target_up"].mean())
    baseline_brier = brier_score_loss(
        test_all["target_up"], np.full(len(test_all), baseline_prob)
    )
    baseline_logloss = log_loss(
        test_all["target_up"],
        np.full(len(test_all), baseline_prob),
        labels=[0, 1],
    )

    test_all["position"] = np.select(
        [prob_up >= THRESHOLD, prob_up <= 1 - THRESHOLD],
        [1, -1], default=0
    )

    # Select non-overlapping four-hour test entries.
    chosen = []
    next_allowed = None
    for idx, row in test_all.iterrows():
        if next_allowed is None or row["datetime"] >= next_allowed:
            chosen.append(idx)
            next_allowed = row["target_time"]

    trades = test_all.loc[chosen].copy()
    trades["gross_return"] = trades["position"] * trades["forward_return"]
    trades["cost"] = np.where(
        trades["position"] != 0, ROUND_TRIP_COST, 0.0
    )
    trades["net_return"] = trades["gross_return"] - trades["cost"]
    trades["equity"] = INITIAL_CAPITAL * (
        1 + trades["net_return"]
    ).cumprod()

    equity_path = np.r_[INITIAL_CAPITAL, trades["equity"].to_numpy()]
    peaks = np.maximum.accumulate(equity_path)[1:]
    trades["drawdown"] = trades["equity"] / peaks - 1

    active = trades.loc[trades["position"] != 0]
    final_equity = float(trades["equity"].iloc[-1])
    total_return = final_equity / INITIAL_CAPITAL - 1
    max_drawdown = float(trades["drawdown"].min())

    print("\n=== PURGED FOUR-HOUR BACKTEST — RESEARCH ONLY ===")
    print("Total rows:", len(df))
    print("Test boundary:", test_start)
    print("Training rows before purge:", split)
    print("Training rows after purge:", len(train))
    print("Purged rows:", split - len(train))
    print("Test rows:", len(test_all))
    print("Non-overlapping periods:", len(trades))
    print("Trades:", len(active))
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Majority baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Model Brier score: {brier:.4f} (lower is better)")
    print(f"Constant baseline Brier score: {baseline_brier:.4f}")
    print(f"Model log loss: {ll:.4f} (lower is better)")
    print(f"Constant baseline log loss: {baseline_logloss:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST:.3%}")
    print(f"Final simulated equity: {final_equity:,.2f}")
    print(f"Net compounded return: {total_return:.2%}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    test_all.to_csv(OUTPUT.with_name("backtest_4h_purged_predictions.csv"), index=False)
    trades.to_csv(OUTPUT, index=False)
    print("\nSaved trade results:", OUTPUT.name)
    print("Saved all test predictions:", "backtest_4h_purged_predictions.csv")

if __name__ == "__main__":
    main()
