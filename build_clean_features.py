
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# GLOBAL MARKET AI — CLEAN GOLD FEATURE BUILDER
# ============================================================

INPUT_FILE = Path("gold_data.csv")
OUTPUT_FILE = Path("gold_features_clean.csv")

HORIZON_HOURS = 4

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


def load_data():
    """Load, validate, sort, and deduplicate historical candles."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce", utc=True
    )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "high", "low", "close"])
          .sort_values("datetime")
          .drop_duplicates(subset=["datetime"], keep="last")
          .reset_index(drop=True)
    )

    invalid = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df[["open", "low", "close"]].max(axis=1))
        | (df["low"] > df[["open", "high", "close"]].min(axis=1))
    )

    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} invalid price candles."
        )

    return df


def add_features(df):
    """Calculate historical features without using future prices."""

    df = df.copy()

    close = df["close"]
    timestamps = pd.DatetimeIndex(df["datetime"])

    # Identify intervals between observations.
    gap_hours = (
        df["datetime"].diff().dt.total_seconds() / 3600
    )
    consecutive_hour = gap_hours.eq(1)

    # --------------------------------------------------------
    # 1. RETURNS
    # --------------------------------------------------------

    df["return_1h"] = close.pct_change().where(consecutive_hour)

    # Find exact elapsed-time historical observations.
    # This avoids treating four rows as four elapsed hours.
    time_lookup = pd.Series(
        np.arange(len(df)),
        index=timestamps
    )

    for hours in [4, 12]:
        past_times = pd.DatetimeIndex(
            df["datetime"] - pd.Timedelta(hours=hours)
        )

        positions = time_lookup.reindex(past_times).to_numpy()
        valid = ~pd.isna(positions)

        values = np.full(len(df), np.nan)

        current_positions = np.flatnonzero(valid)
        previous_positions = positions[valid].astype(int)

        close_values = close.to_numpy()

        values[current_positions] = (
            close_values[current_positions]
            / close_values[previous_positions]
            - 1
        )

        df[f"return_{hours}h"] = values

    # --------------------------------------------------------
    # 2. MOVING AVERAGES
    # --------------------------------------------------------
    # These use the latest 5 and 20 observations, respectively.
    # Across market closures, they are observation-based windows,
    # not fixed elapsed-time windows.

    sma_5 = close.rolling(window=5, min_periods=5).mean()
    sma_20 = close.rolling(window=20, min_periods=20).mean()

    df["sma_5"] = sma_5
    df["sma_20"] = sma_20

    df["sma_5_gap"] = close / sma_5 - 1
    df["sma_20_gap"] = close / sma_20 - 1
    df["trend_5_20"] = sma_5 / sma_20 - 1

    # --------------------------------------------------------
    # 3. VOLATILITY
    # --------------------------------------------------------
    # Minimum counts are explicit. Missing one-hour returns
    # are not replaced with zero.

    returns = df["return_1h"]

    df["volatility_10"] = returns.rolling(
        window=10, min_periods=8
    ).std()

    df["volatility_20"] = returns.rolling(
        window=20, min_periods=12
    ).std()

    # --------------------------------------------------------
    # 4. RSI
    # --------------------------------------------------------

    delta = close.diff().where(consecutive_hour)

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=14, min_periods=10
    ).mean()

    avg_loss = loss.rolling(
        window=14, min_periods=10
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0), 50
    )

    df["rsi_14"] = rsi

    # --------------------------------------------------------
    # 5. EXACT FOUR-ELAPSED-HOUR TARGET
    # --------------------------------------------------------

    future_times = pd.DatetimeIndex(
        df["datetime"] + pd.Timedelta(hours=HORIZON_HOURS)
    )

    future_positions = time_lookup.reindex(future_times).to_numpy()

    valid_future = ~pd.isna(future_positions)
    future_close = np.full(len(df), np.nan)

    valid_rows = np.flatnonzero(valid_future)
    future_rows = future_positions[valid_future].astype(int)

    close_values = close.to_numpy()
    future_close[valid_rows] = close_values[future_rows]

    df["future_close_4h"] = future_close
    df["target_time"] = (
        df["datetime"] + pd.Timedelta(hours=HORIZON_HOURS)
    )

    # 1 = future close strictly higher; 0 = lower or equal.
    df["target_up"] = np.where(
        pd.notna(future_close),
        (future_close > close_values).astype(float),
        np.nan
    )

    return df


def main():
    df = load_data()
    df = add_features(df)

    valid_features = df[FEATURES].notna().all(axis=1)
    valid_targets = df["target_up"].notna()

    print("\n--- FEATURE DIAGNOSTICS ---")
    print("Historical rows:", len(df))
    print("Exact four-hour targets:", int(valid_targets.sum()))

    print("\nMissing values by feature:")
    print(df[FEATURES].isna().sum())

    print("\nRows with all features:", int(valid_features.sum()))
    print(
        "Rows with valid features and targets:",
        int((valid_features & valid_targets).sum())
    )

    # Keep only rows with complete inputs and exact four-hour labels.
    ready = df.loc[valid_features & valid_targets].copy()
    ready["target_up"] = ready["target_up"].astype(int)

    ready.to_csv(OUTPUT_FILE, index=False)

    print("\n--- CLEAN FEATURE DATASET ---")
    print("Model-ready rows:", len(ready))
    print("Missing feature values:", int(
        ready[FEATURES].isna().sum().sum()
    ))

    print("\nTarget distribution:")
    print(ready["target_up"].value_counts().sort_index())

    print("\nTarget proportions:")
    print(
        ready["target_up"]
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
    )

    print(f"\nSaved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

