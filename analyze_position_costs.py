import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_FILE = BASE_DIR / "position_changes_original_results.csv"
FILTERED_FILE = BASE_DIR / "position_changes_filtered_results.csv"
OUTPUT_FILE = BASE_DIR / "position_cost_sensitivity.csv"

COST_SCENARIOS_BPS = [0, 1, 2, 5, 10, 15]


# ==========================================
# LOAD AND VALIDATE DATA
# ==========================================

def load_results(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path.name}\n"
            "Run backtest_position_changes.py first."
        )

    df = pd.read_csv(file_path)

    required = [
        "datetime",
        "target_time",
        "position",
        "gross_return",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: {missing}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True, errors="coerce"
    )
    df["position"] = pd.to_numeric(
        df["position"], errors="coerce"
    )
    df["gross_return"] = pd.to_numeric(
        df["gross_return"], errors="coerce"
    )

    df = (
        df.dropna(
            subset=[
                "datetime",
                "target_time",
                "position",
                "gross_return",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(f"No valid data in {file_path.name}.")

    if not df["position"].isin([0, 1]).all():
        raise ValueError(
            f"{file_path.name}: position must contain only 0 or 1."
        )

    if (df["target_time"] <= df["datetime"]).any():
        raise ValueError("Found an invalid holding interval.")

    if (df["gross_return"] <= -1).any():
        raise ValueError("Gross returns must be greater than -100%.")

    return df


# ==========================================
# SIMULATE TRANSACTION COSTS
# ==========================================

def calculate_turnover(df):
    positions = df["position"].astype(int).to_numpy()
    turnover = np.zeros(len(df), dtype=float)

    previous_position = 0

    for i, current_position in enumerate(positions):
        if i == 0:
            # Begin with no position.
            turnover[i] = abs(current_position)

        else:
            previous_target = df.loc[i - 1, "target_time"]
            current_time = df.loc[i, "datetime"]

            if current_time > previous_target:
                # Data gap: close the previous position at its end,
                # stay flat during the gap, then enter the new position.
                turnover[i] = (
                    previous_position + current_position
                )
            else:
                # Continuous intervals: charge only when position changes.
                turnover[i] = abs(
                    current_position - previous_position
                )

        previous_position = current_position

    # Close any open position at the end of the test.
    if positions[-1] == 1:
        turnover[-1] += 1

    return turnover


def evaluate_strategy(df, strategy_name):
    turnover = calculate_turnover(df)

    gross_returns = df["gross_return"].to_numpy(dtype=float)

    rows = []

    for bps in COST_SCENARIOS_BPS:
        cost_per_side = bps / 10000.0

        transaction_costs = turnover * cost_per_side
        net_returns = gross_returns - transaction_costs

        # Stop compounding if the simulated equity becomes invalid.
        if np.any(net_returns <= -1):
            net_total_return = -100.0
            max_drawdown = -100.0
        else:
            equity = np.cumprod(1.0 + net_returns)
            peaks = np.maximum.accumulate(
                np.concatenate(([1.0], equity))
            )[1:]
            drawdowns = equity / peaks - 1.0

            net_total_return = (equity[-1] - 1.0) * 100
            max_drawdown = drawdowns.min() * 100

        gross_equity = np.cumprod(1.0 + gross_returns)
        gross_total_return = (
            gross_equity[-1] - 1.0
        ) * 100

        rows.append({
            "strategy": strategy_name,
            "cost_bps_per_side": bps,
            "test_intervals": len(df),
            "long_intervals": int(df["position"].sum()),
            "exposure_pct": df["position"].mean() * 100,
            "turnover_units": int(turnover.sum()),
            "gross_compounded_return_pct": gross_total_return,
            "net_compounded_return_pct": net_total_return,
            "max_drawdown_pct": max_drawdown,
            "total_transaction_cost_pct": (
                transaction_costs.sum() * 100
            ),
        })

    return rows


# ==========================================
# MAIN
# ==========================================

def main():
    original = load_results(ORIGINAL_FILE)
    filtered = load_results(FILTERED_FILE)

    # Ensure both strategies cover identical test intervals.
    if not original["datetime"].equals(filtered["datetime"]):
        raise ValueError(
            "The strategy files do not have matching timestamps. "
            "Run backtest_position_changes.py again."
        )

    results = []

    results.extend(
        evaluate_strategy(original, "Original Momentum")
    )

    results.extend(
        evaluate_strategy(filtered, "High-Volatility Filter")
    )

    report = pd.DataFrame(results)

    print("\n" + "=" * 76)
    print("TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 76)
    print("Cost assumptions are per side.")
    print("The simulation uses the existing test-period signals.")
    print("No new model is trained in this script.\n")

    print(
        report.round(3).to_string(index=False)
    )

    report.to_csv(OUTPUT_FILE, index=False)

    print(f"\nResults saved to: {OUTPUT_FILE.name}")
    print("\nImportant: historical simulation only; not a live-trading result.")


if __name__ == "__main__":
    main()
