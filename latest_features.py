from pathlib import Path

from build_clean_features import load_data, add_features, FEATURES

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "gold_latest_features.csv"


def main():
    df = add_features(load_data())

    valid = df[FEATURES].notna().all(axis=1)
    ready = df.loc[valid].copy()

    if ready.empty:
        raise ValueError("No rows have complete model features.")

    latest = ready.iloc[[-1]].copy()
    latest.to_csv(OUTPUT_FILE, index=False)

    raw_latest = df["datetime"].max()
    feature_time = latest["datetime"].iloc[0]

    print("\n--- LATEST FEATURE SNAPSHOT ---")
    print(f"Latest raw candle (UTC): {raw_latest}")
    print(f"Latest usable feature time (UTC): {feature_time}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nFeatures:")
    print(latest[FEATURES].to_string(index=False))

    if feature_time != raw_latest:
        print(
            "\nWARNING: Latest raw candle does not have "
            "all required features. Snapshot is older."
        )
    else:
        print("\nLatest raw candle has all required features.")


if __name__ == "__main__":
    main()
