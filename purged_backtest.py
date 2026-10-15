
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DATA_FILE = "gold_features.csv"
OUTPUT_FILE = "purged_backtest_results.csv"

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

INITIAL_CAPITAL = 100_000.0
ROUND_TRIP_COST = 0.001
PROBABILITY_THRESHOLD = 0.55
HOLDING_CANDLES = 4
TRAIN_FRACTION = 0.80


def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()

    required = FEATURES + ["datetime", "close", "target_up"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Keep only rows with known labels and complete features.
    df = df.dropna(subset=FEATURES + ["target_up"]).copy()
    df["target_up"] = df["target_up"].astype(int)

    split = int(len(df) * TRAIN_FRACTION)

    # Each target looks four candles into the future.
    # Purge the last four training rows so their labels cannot
    # extend into the test period.
    train_end = split - HOLDING_CANDLES
    if train_end < 50 or len(df) - split < 10:
        raise ValueError("Not enough rows for the purged split.")

    train = df.iloc[:train_end].copy()
    test = df.iloc[split:].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train["target_up"])

    probabilities = model.predict_proba(test[FEATURES])
    classes = list(model.named_steps["classifier"].classes_)
    test["prob_up"] = probabilities[:, classes.index(1)]

    # A signal is available only after its candle closes.
    # Enter at the next candle close and exit four candles later.
    test["entry_close"] = test["close"].shift(-1)
    test["exit_close"] = test["close"].shift(-(HOLDING_CANDLES + 1))
    test = test.dropna(subset=["entry_close", "exit_close"]).copy()

    # Non-overlapping four-candle trades.
    test = test.iloc[::HOLDING_CANDLES].copy()
    if test.empty:
        raise ValueError("No test periods remain.")

    test["forward_return"] = (
        test["exit_close"] / test["entry_close"] - 1
    )

    test["position"] = np.select(
        [
            test["prob_up"] >= PROBABILITY_THRESHOLD,
            test["prob_up"] <= 1 - PROBABILITY_THRESHOLD,
        ],
        [1, -1],
        default=0,
    )

    test["gross_return"] = test["position"] * test["forward_return"]
    test["cost"] = np.where(
        test["position"] != 0, ROUND_TRIP_COST, 0.0
    )
    test["net_return"] = test["gross_return"] - test["cost"]

    test["equity"] = INITIAL_CAPITAL * (
        1 + test["net_return"]
    ).cumprod()

    equity_with_start = np.r_[
        INITIAL_CAPITAL, test["equity"].to_numpy()
    ]
    peaks = np.maximum.accumulate(equity_with_start)[1:]
    test["drawdown"] = test["equity"].to_numpy() / peaks - 1

    traded = test[test["position"] != 0]
    win_rate = (
        (traded["net_return"] > 0).mean()
        if len(traded) else np.nan
    )

    print("\nPURGED BACKTEST — PRELIMINARY RESEARCH")
    print(f"Usable rows: {len(df)}")
    print(f"Training rows after purge: {len(train)}")
    print(f"Purged training rows: {split - train_end}")
    print(f"Test holding periods: {len(test)}")
    print(f"Training ends: {train['datetime'].iloc[-1]}")
    print(f"Test begins: {test['datetime'].iloc[0]}")
    print(f"Test ends: {test['datetime'].iloc[-1]}")
    print(f"Trades taken: {len(traded)}")
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST:.3%}")
    print(f"Total return: {test['equity'].iloc[-1] / INITIAL_CAPITAL - 1:.2%}")
    print(f"Maximum drawdown: {test['drawdown'].min():.2%}")
    print(
        f"Trade win rate: {win_rate:.2%}"
        if pd.notna(win_rate)
        else "Trade win rate: no trades"
    )

    test.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
