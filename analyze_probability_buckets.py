
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "reliable_backtest_predictions.csv"

OUTPUT_FILE = BASE_DIR / "probability_bucket_results.csv"

BUCKETS = [
    0.0,
    0.45,
    0.48,
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60,
    0.65,
    1.000001,
]

LABELS = [
    "<45%",
    "45–48%",
    "48–50%",
    "50–52%",
    "52–54%",
    "54–56%",
    "56–58%",
    "58–60%",
    "60–65%",
    "65%+",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}. "
            "Run reliable_backtest.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "datetime",
        "close",
        "future_close_4h",
        "prob_up",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in ["close", "future_close_4h", "prob_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid rows found.")

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if not df["prob_up"].between(0, 1).all():
        raise ValueError("prob_up must be between 0 and 1.")

    if not (df["close"] > 0).all():
        raise ValueError("Close prices must be positive.")

    if not (df["future_close_4h"] > 0).all():
        raise ValueError("Future prices must be positive.")

    # Calculate outcomes directly from the recorded prices.
    df["forward_return"] = (
        df["future_close_4h"] / df["close"]
    ) - 1

    df["actual_up"] = (
        df["future_close_4h"] > df["close"]
    ).astype(int)

    # Prediction direction at the standard 0.50 cutoff.
    df["predicted_up"] = (
        df["prob_up"] >= 0.50
    ).astype(int)

    df["direction_correct"] = (
        df["predicted_up"] == df["actual_up"]
    ).astype(int)

    df["probability_bucket"] = pd.cut(
        df["prob_up"],
        bins=BUCKETS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )

    grouped = df.groupby(
        "probability_bucket",
        observed=False,
    )

    results = grouped.agg(
        observations=("prob_up", "size"),
        average_probability=("prob_up", "mean"),
        actual_up_rate=("actual_up", "mean"),
        average_forward_return=("forward_return", "mean"),
        median_forward_return=("forward_return", "median"),
        directional_accuracy=("direction_correct", "mean"),
    ).reset_index()

    # Convert rates to percentages for easier reading.
    results["actual_up_rate_pct"] = (
        results["actual_up_rate"] * 100
    )

    results["average_forward_return_pct"] = (
        results["average_forward_return"] * 100
    )

    results["median_forward_return_pct"] = (
        results["median_forward_return"] * 100
    )

    results["directional_accuracy_pct"] = (
        results["directional_accuracy"] * 100
    )

    # Save only populated buckets.
    populated = results[
        results["observations"] > 0
    ].copy()

    populated.to_csv(OUTPUT_FILE, index=False)

    print("\nAI PROBABILITY-BUCKET ANALYSIS")
    print("=" * 100)
    print(f"Observations: {len(df)}")
    print(f"First timestamp: {df['datetime'].iloc[0]}")
    print(f"Last timestamp:  {df['datetime'].iloc[-1]}")
    print(
        "Outcome: actual close four hours later compared with current close"
    )
    print("No new model was trained.")

    print("\nBUCKET RESULTS")
    print("-" * 100)

    display = populated[
        [
            "probability_bucket",
            "observations",
            "average_probability",
            "actual_up_rate_pct",
            "average_forward_return_pct",
            "median_forward_return_pct",
            "directional_accuracy_pct",
        ]
    ]

    print(
        display.to_string(
            index=False,
            formatters={
                "average_probability": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.4f}"
                ),
                "actual_up_rate_pct": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.2f}%"
                ),
                "average_forward_return_pct": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.4f}%"
                ),
                "median_forward_return_pct": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.4f}%"
                ),
                "directional_accuracy_pct": lambda x: (
                    "N/A" if pd.isna(x) else f"{x:.2f}%"
                ),
            },
        )
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")

    print("\nHOW TO INTERPRET")
    print("-" * 100)
    print(
        "Actual up rate: percentage of observations where the future close "
        "was higher than the current close."
    )
    print(
        "Average forward return: mean price return before trading costs."
    )
    print(
        "Directional accuracy: accuracy when classifying upward moves "
        "at probability 0.50."
    )
    print(
        "Check sample sizes. Buckets with very few observations can show "
        "misleadingly strong or weak results."
    )
    print(
        "This analysis is descriptive and does not prove profitability "
        "or probability calibration."
    )


if __name__ == "__main__":
    main()
