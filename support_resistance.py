from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gold_data.csv"
OUTPUT_FILE = BASE_DIR / "gold_support_resistance.csv"

LEFT_BARS = 3
RIGHT_BARS = 3
LOOKBACK = 200
ZONE_TOLERANCE_PCT = 0.25
MIN_TOUCHES = 2


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

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
    """Record each pivot only when its right-side bars have arrived."""
    n = len(df)
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for p in range(LEFT_BARS, n - RIGHT_BARS):
        left_highs = highs[p - LEFT_BARS:p]
        right_highs = highs[p + 1:p + RIGHT_BARS + 1]
        left_lows = lows[p - LEFT_BARS:p]
        right_lows = lows[p + 1:p + RIGHT_BARS + 1]

        if (
            highs[p] > max(left_highs)
            and highs[p] >= max(right_highs)
        ):
            pivot_high[p + RIGHT_BARS] = highs[p]

        if (
            lows[p] < min(left_lows)
            and lows[p] <= min(right_lows)
        ):
            pivot_low[p + RIGHT_BARS] = lows[p]

    return pivot_high, pivot_low


def cluster_levels(levels, tolerance):
    if not levels:
        return []

    levels = sorted(float(x) for x in levels if np.isfinite(x))
    clusters = []

    for level in levels:
        if not clusters:
            clusters.append([level])
            continue

        center = float(np.mean(clusters[-1]))
        if (
            center != 0
            and abs(level - center) / abs(center) * 100 <= tolerance
        ):
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [
        {"price": float(np.mean(group)), "touches": len(group)}
        for group in clusters
    ]


def main():
    df = load_data()
    pivot_high, pivot_low = confirmed_pivots(df)

    df["confirmed_pivot_high"] = pivot_high
    df["confirmed_pivot_low"] = pivot_low

    supports = []
    resistances = []
    support_counts = []
    resistance_counts = []
    tested_supports = []
    tested_resistances = []
    breakout_up = []
    breakdown_down = []

    previous_support = np.nan
    previous_resistance = np.nan
    previous_close = np.nan

    for i, row in df.iterrows():
        close = float(row["close"])

        # Test this candle against levels calculated at the prior close.
        tested_supports.append(previous_support)
        tested_resistances.append(previous_resistance)

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

        breakout_up.append(bool(up_break))
        breakdown_down.append(bool(down_break))

        # Calculate levels for the NEXT candle using only pivots confirmed
        # by this candle. The current candle's high/low is not a pivot
        # unless its required right-side observations have already arrived.
        start = max(0, i - LOOKBACK + 1)
        recent = df.iloc[start:i + 1]

        high_levels = recent["confirmed_pivot_high"].dropna().tolist()
        low_levels = recent["confirmed_pivot_low"].dropna().tolist()

        high_zones = cluster_levels(high_levels, ZONE_TOLERANCE_PCT)
        low_zones = cluster_levels(low_levels, ZONE_TOLERANCE_PCT)

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

        resistances.append(
            nearest_r["price"] if nearest_r else np.nan
        )
        supports.append(
            nearest_s["price"] if nearest_s else np.nan
        )
        resistance_counts.append(
            nearest_r["touches"] if nearest_r else 0
        )
        support_counts.append(
            nearest_s["touches"] if nearest_s else 0
        )

        # These levels become the prior-known levels for the next row.
        previous_resistance = (
            nearest_r["price"] if nearest_r else np.nan
        )
        previous_support = (
            nearest_s["price"] if nearest_s else np.nan
        )
        previous_close = close

    df["nearest_support"] = supports
    df["support_touches"] = support_counts
    df["nearest_resistance"] = resistances
    df["resistance_touches"] = resistance_counts

    df["tested_support_level"] = tested_supports
    df["tested_resistance_level"] = tested_resistances
    df["possible_resistance_breakout"] = breakout_up
    df["possible_support_breakdown"] = breakdown_down

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== SUPPORT & RESISTANCE ENGINE ===")
    print("Valid rows:", len(df))
    print("Confirmed pivot highs:", df["confirmed_pivot_high"].notna().sum())
    print("Confirmed pivot lows:", df["confirmed_pivot_low"].notna().sum())
    print("Rows with support:", df["nearest_support"].notna().sum())
    print("Rows with resistance:", df["nearest_resistance"].notna().sum())
    print("Possible breakouts:", int(df["possible_resistance_breakout"].sum()))
    print("Possible breakdowns:", int(df["possible_support_breakdown"].sum()))

    print("\n=== LATEST OBSERVATIONS ===")
    cols = [
        "datetime", "close",
        "tested_support_level", "tested_resistance_level",
        "nearest_support", "nearest_resistance",
        "possible_resistance_breakout",
        "possible_support_breakdown",
    ]
    print(df[cols].tail(10).to_string(index=False))
    print("\nSaved:", OUTPUT_FILE.name)
    print("Research events only; not buy/sell signals.")


if __name__ == "__main__":
    main()
