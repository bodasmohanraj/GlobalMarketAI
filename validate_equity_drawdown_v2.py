
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

TRADES_FILE = BASE / "cost_sensitivity_trades.csv"
PRICES_FILE = BASE / "gold_data.csv"
REPORT_FILE = BASE / "equity_drawdown_validation_v5.csv"
ISSUES_FILE = BASE / "equity_drawdown_issues_v5.csv"

INITIAL_EQUITY = 100_000.0
TOLERANCE = 0.02

GROUP_COLUMNS = [
    "variant",
    "round_trip_cost_pct",
    "slippage_bps_per_side",
]


def main():
    trades = pd.read_csv(TRADES_FILE)
    prices = pd.read_csv(PRICES_FILE)

    trades.columns = trades.columns.str.strip().str.lower()
    prices.columns = prices.columns.str.strip().str.lower()

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    )
    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"], utc=True, errors="coerce"
    )
    prices["datetime"] = pd.to_datetime(
        prices["datetime"], utc=True, errors="coerce"
    )

    numeric_cols = [
        "round_trip_cost_pct",
        "slippage_bps_per_side",
        "entry_price_after_slippage",
        "exit_price_after_slippage",
        "quantity_abstract_units",
        "gross_pnl",
        "total_cost",
        "net_pnl",
        "equity_before",
        "equity_after",
    ]

    for col in numeric_cols:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    for col in ["high", "low", "close"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    trades = trades.dropna(
        subset=[
            "entry_time", "exit_time", "net_pnl",
            "equity_before", "equity_after",
        ]
    ).copy()

    prices = (
        prices.dropna(
            subset=["datetime", "high", "low", "close"]
        )
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    # Use the same exact representation for all timestamp comparisons.
    # pandas Timestamp.value returns integer nanoseconds.
    price_ns = np.array(
        [timestamp.value for timestamp in prices["datetime"]],
        dtype=np.int64,
    )

    close_prices = prices["close"].to_numpy(dtype=float)
    high_prices = prices["high"].to_numpy(dtype=float)
    low_prices = prices["low"].to_numpy(dtype=float)

    # Ensure timestamps and prices remain aligned and sorted.
    order = np.argsort(price_ns)
    price_ns = price_ns[order]
    close_prices = close_prices[order]
    high_prices = high_prices[order]
    low_prices = low_prices[order]

    if not np.all(price_ns[1:] >= price_ns[:-1]):
        raise ValueError("Price timestamps are not sorted.")

    print("Price candles:", len(price_ns))
    print(
        "Price range:",
        prices["datetime"].min(),
        "to",
        prices["datetime"].max(),
    )

    reports = []
    issues = []

    for key, group in trades.groupby(GROUP_COLUMNS, dropna=False):
        variant, cost_pct, slippage_bps = key

        group = group.sort_values(
            ["entry_time", "exit_time"]
        ).reset_index(drop=True)

        equity = INITIAL_EQUITY
        peak = INITIAL_EQUITY
        close_max_dd = 0.0
        adverse_max_dd = 0.0

        issue_count = 0
        max_abs_error = 0.0
        overlap_count = 0
        missing_marks = 0
        previous_exit_ns = None

        for i, trade in group.iterrows():
            entry_ns = int(trade["entry_time"].value)
            exit_ns = int(trade["exit_time"].value)

            if exit_ns < entry_ns:
                issue_count += 1
                issues.append({
                    "variant": variant,
                    "cost_pct": cost_pct,
                    "slippage_bps": slippage_bps,
                    "trade_index": i,
                    "issue": "exit_before_entry",
                    "difference": np.nan,
                })
                continue

            if (
                previous_exit_ns is not None
                and entry_ns < previous_exit_ns
            ):
                overlap_count += 1

            previous_exit_ns = (
                exit_ns if previous_exit_ns is None
                else max(previous_exit_ns, exit_ns)
            )

            before = float(trade["equity_before"])
            after = float(trade["equity_after"])
            gross = float(trade["gross_pnl"])
            cost = float(trade["total_cost"])
            net = float(trade["net_pnl"])

            checks = [
                ("equity_before_mismatch", before - equity),
                ("pnl_mismatch", gross - cost - net),
                ("equity_after_mismatch", before + net - after),
            ]

            for name, difference in checks:
                max_abs_error = max(max_abs_error, abs(difference))
                if abs(difference) > TOLERANCE:
                    issue_count += 1
                    issues.append({
                        "variant": variant,
                        "cost_pct": cost_pct,
                        "slippage_bps": slippage_bps,
                        "trade_index": i,
                        "issue": name,
                        "difference": difference,
                    })

            entry_price = float(
                trade["entry_price_after_slippage"]
            )
            exit_price = float(
                trade["exit_price_after_slippage"]
            )
            qty = float(trade["quantity_abstract_units"])

            direction = str(trade["direction"]).strip().upper()
            if direction == "LONG":
                side = 1.0
            elif direction == "SHORT":
                side = -1.0
            else:
                issue_count += 1
                continue

            # Inclusive entry/exit candle range.
            left = int(np.searchsorted(
                price_ns, entry_ns, side="left"
            ))
            right = int(np.searchsorted(
                price_ns, exit_ns, side="right"
            ))

            if left >= right:
                missing_marks += 1
                issues.append({
                    "variant": variant,
                    "cost_pct": cost_pct,
                    "slippage_bps": slippage_bps,
                    "trade_index": i,
                    "issue": "no_matching_candles",
                    "difference": np.nan,
                })
            else:
                entry_notional = abs(entry_price * qty)
                exit_notional = abs(exit_price * qty)
                notional_sum = entry_notional + exit_notional

                # Approximate entry fee allocation from recorded costs.
                entry_cost = (
                    cost * entry_notional / notional_sum
                    if notional_sum > 0
                    else cost / 2.0
                )

                segment_close = close_prices[left:right]
                segment_high = high_prices[left:right]
                segment_low = low_prices[left:right]

                # Account equity at candle close while trade is open.
                close_equity = (
                    before - entry_cost
                    + side * qty * (segment_close - entry_price)
                )

                # Adverse candle estimate: low for longs, high for shorts.
                adverse_prices = (
                    segment_low if side == 1 else segment_high
                )
                adverse_equity = (
                    before - entry_cost
                    + side * qty * (adverse_prices - entry_price)
                )

                # Include intratrade marks and realized equity in
                # a chronological equity-peak calculation.
                for mark_close, mark_adverse in zip(
                    close_equity, adverse_equity
                ):
                    peak = max(peak, float(mark_close))
                    dd_close = (
                        float(mark_close) / peak - 1
                        if peak > 0 else 0.0
                    )
                    close_max_dd = min(close_max_dd, dd_close)

                    # Compare adverse mark against the peak already
                    # established by previous marks and this close.
                    dd_adverse = (
                        float(mark_adverse) / peak - 1
                        if peak > 0 else 0.0
                    )
                    adverse_max_dd = min(
                        adverse_max_dd, dd_adverse
                    )

            equity = after
            peak = max(peak, equity)

            realized_dd = (
                equity / peak - 1 if peak > 0 else 0.0
            )
            close_max_dd = min(close_max_dd, realized_dd)
            adverse_max_dd = min(adverse_max_dd, realized_dd)

        reports.append({
            "variant": variant,
            "round_trip_cost_pct": cost_pct,
            "slippage_bps_per_side": slippage_bps,
            "trades": len(group),
            "ending_equity_from_csv": equity,
            "realized_return_pct": (
                equity / INITIAL_EQUITY - 1
            ) * 100,
            "close_based_max_drawdown_pct": close_max_dd * 100,
            "adverse_candle_drawdown_estimate_pct":
                adverse_max_dd * 100,
            "reconciliation_issue_count": issue_count,
            "max_abs_reconciliation_error": max_abs_error,
            "overlapping_trade_count": overlap_count,
            "trades_without_price_marks": missing_marks,
        })

    report = pd.DataFrame(reports)
    report.to_csv(REPORT_FILE, index=False)
    pd.DataFrame(issues).to_csv(ISSUES_FILE, index=False)

    print("\n=== VALIDATION V5 ===")
    print(report.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    ))

    print("\nSaved:", REPORT_FILE.name)
    print("Saved:", ISSUES_FILE.name)

    if (report["trades_without_price_marks"] == 0).all():
        print("PASS: every trade matched to price candles.")
    else:
        print("WARNING: some trades have no matching price candles.")

    if (report["reconciliation_issue_count"] == 0).all():
        print("PASS: equity/P&L reconciliation checks passed.")
    else:
        print("WARNING: inspect the issues CSV.")

    if (report["overlapping_trade_count"] == 0).all():
        print("PASS: no overlapping trade intervals detected.")
    else:
        print("WARNING: overlapping trade intervals detected.")

    print(
        "\nCaution: adverse candle drawdown is an estimate, not an "
        "exact intrabar simulation. It does not model the full price "
        "path, execution ordering, or every possible fee timing."
    )


if __name__ == "__main__":
    main()
