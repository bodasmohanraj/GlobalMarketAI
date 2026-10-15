from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gold_data.csv"
OUTPUT_FILE = BASE_DIR / "gold_market_regimes.csv"

def classify_market(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "datetime" not in df.columns:
        for candidate in ["date", "timestamp"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "datetime"})
                break

    required = {"datetime", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

    # Trend and volatility features use only current and prior observations.
    df["sma_5"] = df["close"].rolling(5, min_periods=5).mean()
    df["sma_20"] = df["close"].rolling(20, min_periods=20).mean()
    df["sma_50"] = df["close"].rolling(50, min_periods=50).mean()

    df["trend_strength_pct"] = (
        (df["sma_5"] - df["sma_20"]) / df["sma_20"] * 100
    )
    df["return_1"] = df["close"].pct_change()
    df["volatility_20"] = (
        df["return_1"].rolling(20, min_periods=20).std() * np.sqrt(20) * 100
    )

    # Compare volatility with its historical rolling median.
    df["volatility_median_100"] = (
        df["volatility_20"].rolling(100, min_periods=50).median()
    )
    df["high_volatility"] = (
        df["volatility_20"] > df["volatility_median_100"] * 1.5
    )

    def regime(row):
        if pd.isna(row["sma_5"]) or pd.isna(row["sma_20"]) or pd.isna(row["sma_50"]):
            return "INSUFFICIENT_HISTORY"
        if pd.isna(row["volatility_median_100"]):
            return "INSUFFICIENT_HISTORY"

        if row["high_volatility"]:
            return "HIGH_VOLATILITY"

        if row["sma_5"] > row["sma_20"] > row["sma_50"]:
            return "UPTREND"

        if row["sma_5"] < row["sma_20"] < row["sma_50"]:
            return "DOWNTREND"

        return "SIDEWAYS_OR_MIXED"

    df["market_regime"] = df.apply(regime, axis=1)

    return df

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find {DATA_FILE.name}")

    raw = pd.read_csv(DATA_FILE)
    result = classify_market(raw)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n=== GOLD MARKET REGIME ENGINE ===")
    print("Input rows:", len(raw))
    print("Valid rows:", len(result))
    print("First timestamp:", result["datetime"].min())
    print("Last timestamp:", result["datetime"].max())

    print("\n=== REGIME COUNTS ===")
    print(result["market_regime"].value_counts().to_string())

    print("\n=== LATEST OBSERVATIONS ===")
    cols = [
        "datetime", "close", "sma_5", "sma_20", "sma_50",
        "trend_strength_pct", "volatility_20", "market_regime"
    ]
    print(result[cols].tail(10).to_string(index=False))

    print("\nSaved:", OUTPUT_FILE.name)
    print("Research classification only; not a buy/sell signal.")

if __name__ == "__main__":
    main()
