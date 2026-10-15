
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_FILE = "gold_features.csv"
OUTPUT_FILE = "strategy_comparison.csv"

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

    # The model target looks four candles ahead. Keep rows with
    # complete features and known labels for the chronological split.
    df = df.dropna(subset=FEATURES + ["target_up"]).copy()
    df["target_up"] = df["target_up"].astype(int)

    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
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

    test["position"] = np.select(
        [
            test["prob_up"] >= PROBABILITY_THRESHOLD,
            test["prob_up"] <= 1 - PROBABILITY_THRESHOLD,
        ],
        [1, -1],
        default=0,
    )

    # Use non-overlapping four-candle holding periods.
    test["future_close"] = test["close"].shift(-HOLDING_CANDLES)
    test["forward_return"] = test["future_close"] / test["close"] - 1
    test = test.iloc[::HOLDING_CANDLES].copy()
    test = test.dropna(subset=["forward_return"]).copy()

    if test.empty:
        raise ValueError("No test periods remain for comparison.")

    # AI strategy: take the model's long, short, or flat position.
    test["ai_gross_return"] = test["position"] * test["forward_return"]
    test["ai_cost"] = np.where(
        test["position"] != 0, ROUND_TRIP_COST, 0.0
    )
    test["ai_net_return"] = test["ai_gross_return"] - test["ai_cost"]

    # Buy-and-hold: stay long throughout the same evaluation periods.
    test["buy_hold_gross_return"] = test["forward_return"]
    test["buy_hold_cost"] = 0.0
    test["buy_hold_net_return"] = test["buy_hold_gross_return"]

    # No-trade baseline: leave capital unchanged.
    test["no_trade_net_return"] = 0.0

    strategies = {
        "AI strategy": "ai_net_return",
        "Buy and hold": "buy_hold_net_return",
        "No trade": "no_trade_net_return",
    }

    comparison = []

    for name, return_col in strategies.items():
        returns = test[return_col].fillna(0)
        equity = INITIAL_CAPITAL * (1 + returns).cumprod()

        equity_with_start = np.r_[INITIAL_CAPITAL, equity.to_numpy()]
        peaks = np.maximum.accumulate(equity_with_start)[1:]
        drawdown = equity.to_numpy() / peaks - 1

        if name == "AI strategy":
            traded = test[test["position"] != 0]
            win_rate = (
                (traded["ai_net_return"] > 0).mean()
                if len(traded) else np.nan
            )
            trade_count = len(traded)
        elif name == "Buy and hold":
            win_rate = (returns > 0).mean()
            trade_count = 1
        else:
            win_rate = np.nan
            trade_count = 0

        comparison.append({
            "Strategy": name,
            "Evaluation periods": len(test),
            "Trades / entries": trade_count,
            "Final equity": equity.iloc[-1],
            "Total return (%)": (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100,
            "Maximum drawdown (%)": drawdown.min() * 100,
            "Positive period rate (%)": (
                win_rate * 100 if pd.notna(win_rate) else np.nan
            ),
        })

    results = pd.DataFrame(comparison)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nSTRATEGY COMPARISON — RESEARCH ONLY")
    print(f"Test period: {test['datetime'].iloc[0]} to {test['datetime'].iloc[-1]}")
    print(f"Evaluation periods: {len(test)}")
    print(f"Initial capital: {INITIAL_CAPITAL:,.2f}")
    print(f"Assumed AI round-trip cost: {ROUND_TRIP_COST:.3%}\n")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved comparison to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
