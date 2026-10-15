
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "risk_managed_v2_trades.csv"

DETAILS_OUT = BASE / "volatility_fold_details.csv"
SUMMARY_OUT = BASE / "volatility_fold_summary.csv"

ATR_PERIOD = 14
VOLATILITY_BUCKETS = 3


def load_trades():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE.name}. Run risk_managed_backtest_v2.py first."
        )

    df = pd.read_csv(SOURCE)
    df.columns = df.columns.str.strip().str.lower()

    required = [
        "fold", "entry_time", "entry_price", "atr_at_signal",
        "gross_pnl", "total_cost", "net_pnl", "exit_reason",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in [
        "entry_price", "atr_at_signal", "gross_pnl",
        "total_cost", "net_pnl",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True, errors="coerce"
    )

    df = df.dropna(
        subset=[
            "fold", "entry_time", "entry_price",
            "atr_at_signal", "net_pnl",
        ]
    ).copy()

    df["atr_pct"] = df["atr_at_signal"] / df["entry_price"] * 100

    # Use one set of thresholds for the entire sample, matching the
    # original diagnostic's full-sample tercile approach.
    df["volatility_bucket"] = pd.qcut(
        df["atr_pct"],
        q=VOLATILITY_BUCKETS,
        labels=["low", "medium", "high"],
        duplicates="drop",
    ).astype(str)

    return df


def profit_factor(pnl):
    pnl = np.asarray(pnl, dtype=float)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan

    return gains / losses


def summarize(group):
    pnl = group["net_pnl"].to_numpy(dtype=float)

    return {
        "trades": len(group),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else 0,
        "net_pnl": float(pnl.sum()),
        "mean_net_pnl": float(pnl.mean()) if len(pnl) else 0,
        "profit_factor": float(profit_factor(pnl)),
    }


def main():
    df = load_trades()

    # Detailed summary by fold and volatility.
    rows = []

    for (fold, bucket), group in df.groupby(
        ["fold", "volatility_bucket"], observed=True
    ):
        row = {
            "fold": int(fold),
            "volatility": bucket,
        }
        row.update(summarize(group))
        rows.append(row)

    details = pd.DataFrame(rows).sort_values(["fold", "volatility"])
    details.to_csv(DETAILS_OUT, index=False)

    # Overall volatility summary.
    overall_rows = []

    for bucket, group in df.groupby("volatility_bucket", observed=True):
        row = {"volatility": bucket}
        row.update(summarize(group))

        fold_values = (
            details[details["volatility"] == bucket]
            .sort_values("fold")
        )

        row["positive_folds"] = int((fold_values["net_pnl"] > 0).sum())
        row["folds_with_trades"] = len(fold_values)

        overall_rows.append(row)

    overall = pd.DataFrame(overall_rows).sort_values("volatility")
    overall.to_csv(SUMMARY_OUT, index=False)

    print("\n=== VOLATILITY PERFORMANCE BY FOLD ===")
    print(
        details.to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n=== OVERALL VOLATILITY SUMMARY ===")
    print(
        overall.to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nCreated:", DETAILS_OUT.name)
    print("Created:", SUMMARY_OUT.name)

    print(
        "\nImportant: volatility thresholds are calculated from the full "
        "trade sample, so this is exploratory analysis, not an independent "
        "out-of-sample test. A live-use filter would need thresholds fixed "
        "using past data only and evaluated on later unseen data."
    )


if __name__ == "__main__":
    main()
