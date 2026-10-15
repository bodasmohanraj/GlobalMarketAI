
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "gold_data.csv"
CLEAN_FILE = BASE_DIR / "gold_features_clean.csv"

TOLERANCE = 0.02  # Small allowance for CSV floating-point differences


def find_column(df, candidates, description):
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_prices(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    time_col = find_column(
        df,
        ["datetime", "timestamp", "date", "time"],
        "timestamp",
    )

    close_col = find_column(
        df,
        ["close"],
        "close price",
    )

    df = df[[time_col, close_col]].copy()
    df.columns = ["datetime", "close"]

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = df.dropna(subset=["datetime", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if df["datetime"].duplicated().any():
        duplicates = int(df["datetime"].duplicated().sum())
        raise ValueError(
            f"{path.name} contains {duplicates} duplicate timestamps."
        )

    return df


def main():
    print("\nGOLD FOUR-HOUR TARGET AUDIT")
    print("=" * 60)

    raw = load_prices(RAW_FILE)

    clean = pd.read_csv(CLEAN_FILE)

    required = [
        "datetime",
        "close",
        "future_close_4h",
        "target_up",
    ]

    missing = [col for col in required if col not in clean.columns]

    if missing:
        raise ValueError(
            f"Missing columns in clean dataset: {missing}"
        )

    clean["datetime"] = pd.to_datetime(
        clean["datetime"],
        utc=True,
        errors="coerce",
    )

    for col in ["close", "future_close_4h", "target_up"]:
        clean[col] = pd.to_numeric(
            clean[col],
            errors="coerce",
        )

    clean = clean.dropna(subset=required).copy()
    clean = clean.sort_values("datetime").reset_index(drop=True)

    # Check that timestamps are unique.
    print(f"\nRaw price rows:       {len(raw)}")
    print(f"Clean feature rows:   {len(clean)}")
    print(f"Raw data starts:      {raw['datetime'].min()}")
    print(f"Raw data ends:        {raw['datetime'].max()}")
    print(f"Clean data starts:    {clean['datetime'].min()}")
    print(f"Clean data ends:      {clean['datetime'].max()}")

    # Build an exact timestamp-to-price lookup.
    raw_lookup = raw.set_index("datetime")["close"]

    # Check the current close against raw data.
    clean["raw_current_close"] = clean["datetime"].map(raw_lookup)

    current_match = clean["raw_current_close"].notna()

    current_diff = (
        clean.loc[current_match, "close"]
        - clean.loc[current_match, "raw_current_close"]
    ).abs()

    current_price_match = current_diff <= TOLERANCE

    print("\n1. CURRENT CLOSE VALIDATION")
    print("-" * 60)
    print(f"Rows with matching raw timestamp: {int(current_match.sum())}")
    print(
        "Rows whose close matches raw data: "
        f"{int(current_price_match.sum())}"
    )
    print(
        "Rows with missing raw timestamps: "
        f"{int((~current_match).sum())}"
    )
    print(
        "Rows with mismatched close prices: "
        f"{int((~current_price_match).sum())}"
    )

    # Each target must refer to exactly datetime + 4 hours.
    clean["expected_target_time"] = (
        clean["datetime"] + pd.Timedelta(hours=4)
    )

    clean["raw_future_close"] = clean[
        "expected_target_time"
    ].map(raw_lookup)

    future_exists = clean["raw_future_close"].notna()

    future_diff = (
        clean.loc[future_exists, "future_close_4h"]
        - clean.loc[future_exists, "raw_future_close"]
    ).abs()

    future_price_match = future_diff <= TOLERANCE

    print("\n2. FUTURE CLOSE VALIDATION")
    print("-" * 60)
    print(
        "Targets with an actual raw candle exactly four hours later: "
        f"{int(future_exists.sum())}"
    )
    print(
        "Targets whose future close matches raw data: "
        f"{int(future_price_match.sum())}"
    )
    print(
        "Targets with no exact future candle: "
        f"{int((~future_exists).sum())}"
    )
    print(
        "Targets with mismatched future prices: "
        f"{int((~future_price_match).sum())}"
    )

    # Independently recompute the binary target.
    expected_label = (
        clean["future_close_4h"] > clean["close"]
    ).astype(int)

    label_match = (
        clean["target_up"].astype(int) == expected_label
    )

    print("\n3. TARGET LABEL VALIDATION")
    print("-" * 60)
    print(f"Correct target labels: {int(label_match.sum())}")
    print(f"Incorrect target labels: {int((~label_match).sum())}")

    # Summarize all conditions for a fully verified row.
    fully_verified = (
        current_match
        & current_price_match.reindex(clean.index, fill_value=False)
        & future_exists
        & future_price_match.reindex(clean.index, fill_value=False)
        & label_match
    )

    print("\n4. OVERALL RESULT")
    print("-" * 60)
    print(f"Fully verified rows: {int(fully_verified.sum())}")
    print(f"Rows audited:        {len(clean)}")

    # Save rows that need investigation.
    audit = clean[
        [
            "datetime",
            "close",
            "future_close_4h",
            "target_up",
            "expected_target_time",
            "raw_current_close",
            "raw_future_close",
        ]
    ].copy()

    audit["current_timestamp_found"] = current_match
    audit["future_timestamp_found"] = future_exists
    audit["target_label_correct"] = label_match

    audit["current_close_matches"] = False
    audit.loc[current_match, "current_close_matches"] = (
        current_price_match.to_numpy()
    )

    audit["future_close_matches"] = False
    audit.loc[future_exists, "future_close_matches"] = (
        future_price_match.to_numpy()
    )

    audit["fully_verified"] = fully_verified

    output_path = BASE_DIR / "target_alignment_audit.csv"
    audit.to_csv(output_path, index=False)

    print(f"Audit details saved to: {output_path}")

    if fully_verified.all():
        print("\nPASS: Every clean row passed these target alignment checks.")
    else:
        print(
            "\nWARNING: Some rows failed validation. "
            "Inspect target_alignment_audit.csv before changing the model."
        )

    print(
        "\nNote: This audit checks price and label alignment. "
        "It does not independently prove that every feature is free "
        "from future information."
    )


if __name__ == "__main__":
    main()
