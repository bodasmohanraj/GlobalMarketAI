
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_FILE = "gold_features.csv"
OUTPUT_FILE = "backtest_results.csv"

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

# Simplified research assumptions, not actual broker costs.
ROUND_TRIP_COST = 0.001
PROBABILITY_THRESHOLD = 0.55
INITIAL_CAPITAL = 100_000
HOLDING_CANDLES = 4


def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()

    required = FEATURES + ["datetime", "close", "target_up"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce"
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.dropna(
        subset=FEATURES + ["close", "target_up"]
    ).copy()
    df["target_up"] = df["target_up"].astype(int)

    if len(df) < 50:
        raise ValueError("Not enough usable rows for testing.")

    # Train on earlier data and test on later data.
    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    model.fit(train[FEATURES], train["target_up"])

    probabilities = model.predict_proba(test[FEATURES])
    classes = list(model.named_steps["classifier"].classes_)
    up_index = classes.index(1)

    test["prob_up"] = probabilities[:, up_index]

    # Long when upward probability is high.
    # Short when downward probability is high.
    # Otherwise stay out of the market.
    test["position"] = np.select(
        [
            test["prob_up"] >= PROBABILITY_THRESHOLD,
            test["prob_up"] <= 1 - PROBABILITY_THRESHOLD,
        ],
        [1, -1],
        default=0,
    )

    # Enter at close and exit four candles later.
    test["future_close"] = test["close"].shift(-HOLDING_CANDLES)

    test["forward_return"] = (
        test["future_close"] / test["close"] - 1
    )

    # Use non-overlapping four-candle holding periods.
    test = test.iloc[::HOLDING_CANDLES].copy()
    test = test.dropna(subset=["forward_return"]).copy()

    if test.empty:
        raise ValueError("No usable test periods remain.")

    test["gross_return"] = (
        test["position"] * test["forward_return"]
    )

    test["trade_cost"] = np.where(
        test["position"] != 0,
        ROUND_TRIP_COST,
        0.0,
    )

    test["net_return"] = (
        test["gross_return"] - test["trade_cost"]
    )

    test["equity"] = (
        INITIAL_CAPITAL
        * (1 + test["net_return"]).cumprod()
    )

    equity_with_start = np.r_[
        INITIAL_CAPITAL,
        test["equity"].to_numpy(),
    ]

    running_peak = np.maximum.accumulate(
        equity_with_start
    )[1:]

    test["drawdown"] = (
        test["equity"] / running_peak - 1
    )

    traded = test[test["position"] != 0]

    win_rate = (
        (traded["net_return"] > 0).mean()
        if len(traded)
        else 0.0
    )

    total_return = (
        test["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    )

    max_drawdown = test["drawdown"].min()

    print("\nBACKTEST RESULTS — RESEARCH ONLY")
    print(
        f"Test period: {test['datetime'].iloc[0]} "
        f"to {test['datetime'].iloc[-1]}"
    )
    print(f"Evaluated periods: {len(test)}")
    print(f"Trades taken: {len(traded)}")
    print(f"Probability threshold: {PROBABILITY_THRESHOLD:.2f}")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST:.3%}")
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Final simulated equity: {test['equity'].iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")
    print(f"Trade win rate: {win_rate:.2%}")

    test.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved detailed results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
