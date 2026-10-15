
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "reliable_backtest_predictions.csv"
OUTPUT_FILE = BASE_DIR / "probability_model_comparison.csv"

DEVELOPMENT_FRACTION = 0.70


def clipped_logit(probabilities):
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE.name}. Run reliable_backtest.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "datetime",
        "prob_up",
        "target_up",
        "training_up_rate",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in ["prob_up", "target_up", "training_up_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if not df["prob_up"].between(0, 1).all():
        raise ValueError("prob_up must be between 0 and 1.")

    if not df["training_up_rate"].between(0, 1).all():
        raise ValueError("training_up_rate must be between 0 and 1.")

    if not df["target_up"].isin([0, 1]).all():
        raise ValueError("target_up must contain only 0 and 1.")

    if len(df) < 100:
        raise ValueError("Not enough predictions for this experiment.")

    return df


def evaluate(name, y_true, probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    y_true = np.asarray(y_true, dtype=int)

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "observations": len(y_true),
        "accuracy_pct": accuracy_score(
            y_true, predictions
        ) * 100,
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true, probabilities, labels=[0, 1]
        ),
        "mean_probability_pct": probabilities.mean() * 100,
        "actual_up_rate_pct": y_true.mean() * 100,
    }


def main():
    df = load_data()

    split = int(len(df) * DEVELOPMENT_FRACTION)

    development = df.iloc[:split].copy()
    holdout = df.iloc[split:].copy()

    y_dev = development["target_up"].to_numpy(dtype=int)
    y_test = holdout["target_up"].to_numpy(dtype=int)

    raw_dev = development["prob_up"].to_numpy(dtype=float)
    raw_test = holdout["prob_up"].to_numpy(dtype=float)

    baseline_test = holdout[
        "training_up_rate"
    ].to_numpy(dtype=float)

    if np.unique(y_dev).size < 2:
        raise ValueError(
            "Development data needs both target classes for calibration."
        )

    # Fit a calibration mapping using development predictions only.
    # The underlying gold prediction model is not retrained here.
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(clipped_logit(raw_dev), y_dev)

    calibrated_test = calibrator.predict_proba(
        clipped_logit(raw_test)
    )[:, 1]

    results = [
        evaluate("Raw AI probabilities", y_test, raw_test),
        evaluate("Calibrated AI probabilities", y_test, calibrated_test),
        evaluate("Historical-rate baseline", y_test, baseline_test),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nPROBABILITY MODEL COMPARISON")
    print("=" * 85)
    print(f"Total predictions: {len(df)}")
    print(f"Development predictions: {len(development)}")
    print(f"Holdout predictions: {len(holdout)}")
    print(f"Development ends: {development['datetime'].iloc[-1]}")
    print(f"Holdout starts:   {holdout['datetime'].iloc[0]}")
    print(f"Holdout ends:     {holdout['datetime'].iloc[-1]}")

    print("\nHOLDOUT METRICS")
    print("-" * 85)
    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy_pct": lambda x: f"{x:.2f}%",
                "brier_score": lambda x: f"{x:.6f}",
                "log_loss": lambda x: f"{x:.6f}",
                "mean_probability_pct": lambda x: f"{x:.2f}%",
                "actual_up_rate_pct": lambda x: f"{x:.2f}%",
            },
        )
    )

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("\nINTERPRETATION")
    print("- Lower Brier score and log loss are better.")
    print("- Accuracy measures classification at a 0.50 cutoff.")
    print("- Calibration can improve probability estimates without")
    print("  improving the underlying model's ability to rank outcomes.")
    print("- This is exploratory because the period has been inspected before.")
    print("- A later untouched period is still needed for independent validation.")


if __name__ == "__main__":
    main()
