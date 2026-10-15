
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

TRADES_FILE = BASE / "cost_sensitivity_trades.csv"
PRICES_FILE = BASE / "gold_data.csv"
OUTPUT_FILE = BASE / "intratrade_drawdown_report.csv"

INITIAL_EQUITY = 100_000.0


def main():
    if not TRADES_FILE.exists() or not PRICES_FILE.exists():
        raise FileNotFoundError(
            "Missing cost_sensitivity_trades.csv or gold_data.csv"
        )

    trades = pd.read_csv(TRADES_FILE)
    prices = pd.read_csv(PRICES_FILE)

    trades.columns = trades.columns.str.strip().str.lower()
    prices.columns = prices.columns.str.strip().str.lower()

    prices["datetime"] = pd.to_datetime(
        prices["datetime"], utc=True, errors="coerce"
    )
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    )
    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    for col in [
        "entry_price_after_slippage",
        "exit_price_after_slippage",
        "quantity_abstract_units",
        "equity_before",
        "net_pnl",
    ]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    prices = (
        prices.dropna(subset=["datetime", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    rows = []

    group_cols = [
        "variant",
        "round_trip_cost_pct",
        "slippage_bps_per_side",
    ]

    for key, group in trades.groupby(group_cols, dropna=False):
        variant, cost_pct, slippage_bps = key
        group = group.sort_values("entry_time")

        equity = INITIAL_EQUITY
        equity_peak = INITIAL_EQUITY
        max_dd = 0.0

        for _, trade in group.iterrows():
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]

            if pd.isna(entry_time) or pd.isna(exit_time):
                continue

            entry_price = trade["entry_price_after_slippage"]
            qty = trade["quantity_abstract_units"]
            side = 1 if str(trade["direction"]).upper() == "LONG" else -1

            entry_cost = (
                trade["total_cost"] / 2
                if "total_cost" in trade.index
                and pd.notna(trade["total_cost"])
                else 0.0
            )

            # Use all available candle closes during the trade.
            segment = prices[
                (prices["datetime"] >= entry_time)
                & (prices["datetime"] <= exit_time)
            ]

            if segment.empty or pd.isna(entry_price) or pd.isna(qty):
                continue

            # Start from the account equity before this trade.
            trade_start_equity = equity

            for _, candle in segment.iterrows():
                mark_price = float(candle["close"])
                unrealized = side * qty * (mark_price - entry_price)

                # Entry cost is charged immediately; the exit cost
                # is recognized when the trade closes.
                marked_equity = (
                    trade_start_equity + unrealized - entry_cost
                )

                equity_peak = max(equity_peak, marked_equity)

                if equity_peak > 0:
                    drawdown = marked_equity / equity_peak - 1
                    max_dd = min(max_dd, drawdown)

            # Use the recorded realized result at exit.
            equity = max(0.0, trade_start_equity + float(trade["net_pnl"]))
            equity_peak = max(equity_peak, equity)

            if equity_peak > 0:
                max_dd = min(max_dd, equity / equity_peak - 1)

        rows.append({
            "variant": variant,
            "round_trip_cost_pct": cost_pct,
            "slippage_bps_per_side": slippage_bps,
            "trades": len(group),
            "ending_realized_equity": equity,
            "realized_return_pct": (equity / INITIAL_EQUITY - 1) * 100,
            "approx_close_based_max_drawdown_pct": max_dd * 100,
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n=== APPROXIMATE MARK-TO-MARKET DRAWDOWN ===")
    print(report.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    print("\nCreated:", OUTPUT_FILE.name)
    print(
        "\nCaution: this is an approximation using candle closes. "
        "It does not capture every intrabar equity extreme, and it "
        "assumes trades are processed in entry-time order."
    )


if __name__ == "__main__":
    main()
