
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

DATA_FILE = "gold_features_clean.csv"
OUTPUT_FILE = "backtest_4h_corrected_results.csv"

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

# Research assumptions only; replace with realistic broker-specific costs.
ROUND_TRIP_COST = 0.001
PROBABILITY_THRESHOLD = 0.55
INITIAL_CAPITAL = 100_000
HORIZON_HOURS = 4
TRAIN_FRACTION = 0.80


def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()

    required = FEATURES + [
        "datetime",
        "target_time",
        "close",
        "future_close_4h",
        "target_up",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce", utc=True
    )
    df["target_time"] = pd.to_datetime(
        df["target_time"], errors="coerce", utc=True
    )

    for col in FEATURES + ["close", "future_close_4h", "target_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Enforce exact four-hour targets.
    elapsed = (
        df["target_time"] - df["datetime"]
    ).dt.total_seconds() / 3600

    df = df.loc[elapsed.eq(HORIZON_HOURS)].copy()
    df["target_up"] = df["target_up"].astype(int)

    # Confirm the label agrees with the actual future price.
    expected_target = (
        df["future_close_4h"] > df["close"]
    ).astype(int)

    if not expected_target.equals(df["target_up"]):
        raise ValueError("Target labels do not match future prices.")

    if len(df) < 100:
        raise ValueError("Not enough usable data.")

    # Split chronologically. Keep the most recent 20% for testing.
    split = int(len(df) * TRAIN_FRACTION)
    train = df.iloc[:split].copy()
    test_all = df.iloc[split:].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train["target_up"])

    probabilities = model.predict_proba(test_all[FEATURES])
    classes = list(model.named_steps["classifier"].classes_)
    up_index = classes.index(1)

    test_all["prob_up"] = probabilities[:, up_index]

    test_all["position"] = np.select(
        [
            test_all["prob_up"] >= PROBABILITY_THRESHOLD,
            test_all["prob_up"] <= 1 - PROBABILITY_THRESHOLD,
        ],
        [1, -1],
        default=0,
    )

    # Use the verified exact four-hour future close, not shift(-4).
    test_all["forward_return"] = (
        test_all["future_close_4h"] / test_all["close"] - 1
    )

    # Build non-overlapping test periods using actual timestamps.
    # After a selected entry, the next entry must be at or after
    # the prior entry's target_time.
    selected = []
    next_allowed_time = None

    for idx, row in test_all.iterrows():
        if (
            next_allowed_time is None
            or row["datetime"] >= next_allowed_time
        ):
            selected.append(idx)
            next_allowed_time = row["target_time"]

    test = test_all.loc[selected].copy()

    # Returns use the future close exactly four hours after entry.
    test["gross_return"] = (
        test["position"] * test["forward_return"]
    )

    # Charge assumed round-trip cost only when a position is taken.
    test["trade_cost"] = np.where(
        test["position"] != 0,
        ROUND_TRIP_COST,
        0.0,
    )

    test["net_return"] = test["gross_return"] - test["trade_cost"]
    test["equity"] = (
        INITIAL_CAPITAL * (1 + test["net_return"]).cumprod()
    )

    equity_with_start = np.r_[
        INITIAL_CAPITAL, test["equity"].to_numpy()
    ]
    running_peak = np.maximum.accumulate(equity_with_start)[1:]
    test["drawdown"] = test["equity"] / running_peak - 1

    traded = test.loc[test["position"] != 0]
    total_return = test["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    max_drawdown = test["drawdown"].min()

    # Predictive metrics use every chronological test row,
    # not just the non-overlapping trade-entry rows.
    predicted_class = (
        test_all["prob_up"] >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        test_all["target_up"], predicted_class
    )
    brier = brier_score_loss(
        test_all["target_up"], test_all["prob_up"]
    )

    print("\nCORRECTED FOUR-HOUR BACKTEST — RESEARCH ONLY")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(train)}")
    print(f"Chronological test rows: {len(test_all)}")
    print(f"Non-overlapping test periods: {len(test)}")
    print(f"Trades taken: {len(traded)}")
    print(f"Test start: {test['datetime'].iloc[0]}")
    print(f"Test end: {test['datetime'].iloc[-1]}")
    print(f"Classification accuracy: {accuracy:.2%}")
    print(f"Brier score: {brier:.4f}")
    print(f"Probability threshold: {PROBABILITY_THRESHOLD:.2f}")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST:.3%}")
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Final simulated equity: {test['equity'].iloc[-1]:,.2f}")
    print(f"Net compounded return: {total_return:.2%}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    test.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved detailed results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
