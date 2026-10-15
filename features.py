
import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()

    # Normalize column names
    df.columns = [
        str(col).lower().strip().replace(" ", "_")
        for col in df.columns
    ]

    # Identify timestamp and closing-price columns
    if "datetime" not in df.columns:
        for candidate in ["date", "timestamp"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "datetime"})
                break

    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"Required datetime/close columns missing. Found: {list(df.columns)}"
        )

    # Prepare and sort data
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    close = df["close"]

    # 1. Returns and momentum
    df["return_1h"] = close.pct_change()
    df["return_4h"] = close.pct_change(4)
    df["return_12h"] = close.pct_change(12)

    # 2. Moving averages
    df["sma_5"] = close.rolling(5).mean()
    df["sma_20"] = close.rolling(20).mean()

    # Normalize moving averages relative to price
    df["sma_5_gap"] = close / df["sma_5"] - 1
    df["sma_20_gap"] = close / df["sma_20"] - 1

    # Trend strength: short average relative to long average
    df["trend_5_20"] = df["sma_5"] / df["sma_20"] - 1

    # 3. Volatility
    df["volatility_10"] = df["return_1h"].rolling(10).std()
    df["volatility_20"] = df["return_1h"].rolling(20).std()

    # 4. RSI (14 periods)
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Handle uninterrupted rising periods
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100

    # 5. Target: direction four candles ahead
    df["future_close"] = close.shift(-4)
    df["target_up"] = np.where(
        df["future_close"].isna(),
        np.nan,
        (df["future_close"] > close).astype(float),
    )

    # Save full feature dataset
    df.to_csv("gold_features.csv", index=False)

    # Report
    display_columns = [
        "datetime",
        "close",
        "return_1h",
        "return_4h",
        "sma_5_gap",
        "sma_20_gap",
        "trend_5_20",
        "volatility_10",
        "rsi_14",
        "target_up",
    ]

    print("\nLatest calculated features:")
    print(df[display_columns].tail(10).to_string(index=False))
    print("\nDataset shape:", df.shape)
    print("\nMissing values in model features:")

    feature_columns = [
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

    print(df[feature_columns].isna().sum())
    print("\nSaved features to gold_features.csv")

    return df


if __name__ == "__main__":
    data = pd.read_csv("gold_data.csv")
    create_features(data)
