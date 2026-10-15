from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gold_data.csv"
OUTPUT_FILE = BASE_DIR / "gold_support_resistance.csv"

# Research parameters — we will test these later.
LEFT_BARS = 3
RIGHT_BARS = 3
LOOKBACK = 200
ZONE_TOLERANCE_PCT = 0.25
MIN_TOUCHES = 2


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("gold_data.csv not found.")

    df = pd.read_csv(INPUT_FILE)
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    if "datetime" not in df.columns:
        for candidate in ["date", "timestamp"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "datetime"})
                break

    required = {"datetime", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )
    return df


def confirmed_pivots(df):
    """A pivot becomes available only after RIGHT_BARS later observations."""
    n = len(df)
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for pivot_idx in range(LEFT_BARS, n - RIGHT_BARS):
        left_highs = highs[pivot_idx - LEFT_BARS:pivot_idx]
        right_highs = highs[pivot_idx + 1:pivot_idx + RIGHT_BARS + 1]
        left_lows = lows[pivot_idx - LEFT_BARS:pivot_idx]
        right_lows = lows[pivot_idx + 1:pivot_idx + RIGHT_BARS + 1]

        if highs[pivot_idx] > max(left_highs) and highs[pivot_idx] >= max(right_highs):
            # Store at confirmation time, not at the original pivot time.
            pivot_high[pivot_idx + RIGHT_BARS] = highs[pivot_idx]

        if lows[pivot_idx] < min(left_lows) and lows[pivot_idx] <= min(right_lows):
            pivot_low[pivot_idx + RIGHT_BARS] = lows[pivot_idx]

    return pivot_high, pivot_low


def cluster_levels(levels, tolerance):
    """Group nearby historical pivots into price zones."""
    if not levels:
        return []

    levels = sorted(levels)
    clusters = []

    for level in levels:
        if not clusters:
            clusters.append([level])
            continue

        center = float(np.mean(clusters[-1]))
        if abs(level - center) / center * 100 <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [
        {"price": float(np.mean(cluster)), "touches": len(cluster)}
        for cluster in clusters
    ]


def main():
    df = load_data()
    pivot_high, pivot_low = confirmed_pivots(df)

    df["confirmed_pivot_high"] = pivot_high
    df["confirmed_pivot_low"] = pivot_low

    support = []
    resistance = []
    support_touches = []
    resistance_touches = []
    breakout_up = []
    breakout_down = []

    previous_resistance = np.nan
    previous_support = np.nan
    previous_close = np.nan

    for i, row in df.iterrows():
        # Only use pivot points confirmed before or at this observation.
        start = max(0, i - LOOKBACK + 1)
        recent = df.iloc[start:i + 1]

        high_levels = recent["confirmed_pivot_high"].dropna().tolist()
        low_levels = recent["confirmed_pivot_low"].dropna().tolist()

        high_zones = cluster_levels(high_levels, ZONE_TOLERANCE_PCT)
        low_zones = cluster_levels(low_levels, ZONE_TOLERANCE_PCT)

        close = float(row["close"])

        # A resistance must be above price; support must be below price.
        valid_resistance = [
            z for z in high_zones
            if z["price"] > close and z["touches"] >= MIN_TOUCHES
        ]
        valid_support = [
            z for z in low_zones
            if z["price"] < close and z["touches"] >= MIN_TOUCHES
        ]

        nearest_r = (
            min(valid_resistance, key=lambda z: z["price"]) 
            if valid_resistance else None
        )
        nearest_s = (
            max(valid_support, key=lambda z: z["price"])
            if valid_support else None
        )

        r_price = nearest_r["price"] if nearest_r else np.nan
        s_price = nearest_s["price"] if nearest_s else np.nan

        # Compare against the prior observation's known levels.
        up_break = (
            pd.notna(previous_resistance)
            and pd.notna(previous_close)
            and previous_close <= previous_resistance
            and close > previous_resistance
        )
        down_break = (
            pd.notna(previous_support)
            and pd.notna(previous_close)
            and previous_close >= previous_support
            and close < previous_support
        )

        resistance.append(r_price)
        support.append(s_price)
        resistance_touches.append(
            nearest_r["touches"] if nearest_r else 0
        )
        support_touches.append(
            nearest_s["touches"] if nearest_s else 0
        )
        breakout_up.append(bool(up_break))
        breakout_down.append(bool(down_break))

        previous_resistance = r_price
        previous_support = s_price
        previous_close = close

    df["nearest_support"] = support
    df["support_touches"] = support_touches
    df["nearest_resistance"] = resistance
    df["resistance_touches"] = resistance_touches
    df["possible_resistance_breakout"] = breakout_up
    df["possible_support_breakdown"] = breakout_down

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== SUPPORT & RESISTANCE ENGINE ===")
    print("Input rows:", len(pd.read_csv(INPUT_FILE)))
    print("Valid rows:", len(df))
    print("Confirmed pivot highs:", df["confirmed_pivot_high"].notna().sum())
    print("Confirmed pivot lows:", df["confirmed_pivot_low"].notna().sum())
    print("Rows with a support zone:", df["nearest_support"].notna().sum())
    print("Rows with a resistance zone:", df["nearest_resistance"].notna().sum())
    print("Possible resistance breakouts:", df["possible_resistance_breakout"].sum())
    print("Possible support breakdowns:", df["possible_support_breakdown"].sum())

    print("\n=== LATEST OBSERVATIONS ===")
    cols = [
        "datetime", "close", "nearest_support", "support_touches",
        "nearest_resistance", "resistance_touches",
        "possible_resistance_breakout", "possible_support_breakdown"
    ]
    print(df[cols].tail(10).to_string(index=False))

    print("\nSaved:", OUTPUT_FILE.name)
    print("Research zones only; not buy/sell signals.")


if __name__ == "__main__":
    main()
