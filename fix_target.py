
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("gold_data.csv")
OUTPUT_FILE = Path("gold_features_timebased.csv")

# Forecast horizon is based on elapsed time, not row count.
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


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.lower()

    if not {"datetime", "close"}.issubset(df.columns):
        raise ValueError("gold_data.csv must contain datetime and close columns.")

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce", utc=True
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["datetime", "close"])
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.reset_index(drop=True)

    # Calculate returns using available rows.
    # These row-based features are retained for now; the next step
    # will audit gaps so multi-hour jumps are not treated as one hour.
    df["return_1h"] = df["close"].pct_change(1)
    df["return_4h"] = df["close"].pct_change(4)
    df["return_12h"] = df["close"].pct_change(12)

    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_5_gap"] = df["close"] / df["sma_5"] - 1
    df["sma_20_gap"] = df["close"] / df["sma_20"] - 1
    df["trend_5_20"] = df["sma_5"] / df["sma_20"] - 1

    df["volatility_10"] = df["return_1h"].rolling(10).std()
    df["volatility_20"] = df["return_1h"].rolling(20).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gain / loss

    df["rsi_14"] = 100 - (100 / (1 + relative_strength))
    df.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi_14"] = 50

    # Find the first observation at least four elapsed hours later.
    # Require the timestamp to be within one additional hour of the
    # intended horizon, rather than blindly using the next four rows.
    future_times = df["datetime"] + pd.Timedelta(hours=HORIZON_HOURS)
    future_indices = df["datetime"].searchsorted(future_times, side="left")

    future_close = []
    actual_horizon_hours = []

    for i, j in enumerate(future_indices):
        if j >= len(df):
            future_close.append(float("nan"))
            actual_horizon_hours.append(float("nan"))
            continue

        elapsed = (
            df.loc[j, "datetime"] - df.loc[i, "datetime"]
        ).total_seconds() / 3600

        if HORIZON_HOURS <= elapsed <= HORIZON_HOURS + 1:
            future_close.append(df.loc[j, "close"])
            actual_horizon_hours.append(elapsed)
        else:
            future_close.append(float("nan"))
            actual_horizon_hours.append(float("nan"))

    df["future_close_4h"] = future_close
    df["actual_horizon_hours"] = actual_horizon_hours
    df["target_up"] = (
        df["future_close_4h"] > df["close"]
    ).where(df["future_close_4h"].notna())

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTIME-BASED TARGET DATASET CREATED")
    print(f"Input rows: {len(pd.read_csv(INPUT_FILE))}")
    print(f"Output rows: {len(df)}")
    print(f"Target horizon: {HORIZON_HOURS} elapsed hours")
    print(f"Valid targets: {df['target_up'].notna().sum()}")
    print(f"Rows without valid target: {df['target_up'].isna().sum()}")
    print("\nActual horizon distribution:")
    print(df["actual_horizon_hours"].value_counts(dropna=False).head())
    print(f"\nSaved to: {OUTPUT_FILE.resolve()}")
    print(
        "\nNote: features still use row-based rolling windows. "
        "We will audit time gaps before using this dataset for training."
    )


if __name__ == "__main__":
    main()
