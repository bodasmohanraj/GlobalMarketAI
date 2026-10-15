
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"

TRADES_OUT = BASE / "risk_managed_v2_trades.csv"
FOLDS_OUT = BASE / "risk_managed_v2_folds.csv"
SUMMARY_OUT = BASE / "risk_managed_v2_summary.txt"

INITIAL_EQUITY = 100_000.0
RISK_FRACTION = 0.005
ATR_PERIOD = 14
STOP_ATR_MULTIPLE = 1.0
REWARD_RISK_RATIO = 2.0
ROUND_TRIP_COST = 0.0005
MAX_HOLD_CANDLES = 24
N_FOLDS = 5
TEST_START_FRACTION = 0.50
MAX_NOTIONAL_MULTIPLE = 1.0


def load_data():
    df = pd.read_csv(SOURCE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    return df


def build_indicators(df):
    df = df.copy()

    resistance = (
        df["high"].rolling(20, min_periods=20).max().shift(1)
    )
    support = (
        df["low"].rolling(20, min_periods=20).min().shift(1)
    )
    previous_close = df["close"].shift(1)

    up = (
        (df["close"] > resistance)
        & (previous_close <= resistance.shift(1))
    )
    down = (
        (df["close"] < support)
        & (previous_close >= support.shift(1))
    )

    df["signal"] = np.select([up, down], [1, -1], default=0)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(
        ATR_PERIOD, min_periods=ATR_PERIOD
    ).mean()

    return df


def find_exit(df, entry_index, fold_end, side, stop, target):
    last_index = min(
        entry_index + MAX_HOLD_CANDLES - 1,
        fold_end - 1,
        len(df) - 1,
    )

    for j in range(entry_index, last_index + 1):
        c = df.iloc[j]
        o, h, l = float(c["open"]), float(c["high"]), float(c["low"])

        if side == 1:
            if o <= stop:
                return j, o, "STOP_GAP"
            if o >= target:
                return j, target, "TARGET_GAP"

            stop_hit = l <= stop
            target_hit = h >= target

        else:
            if o >= stop:
                return j, o, "STOP_GAP"
            if o <= target:
                return j, target, "TARGET_GAP"

            stop_hit = h >= stop
            target_hit = l <= target

        # Conservative assumption if both levels are touched in one candle.
        if stop_hit:
            return j, stop, "STOP"
        if target_hit:
            return j, target, "TARGET"

    return last_index, float(df.iloc[last_index]["close"]), "TIME_EXIT"


def profit_factor(pnls):
    pnls = np.asarray(pnls, dtype=float)
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing file: {SOURCE}")

    df = build_indicators(load_data())
    n = len(df)

    first_test = int(n * TEST_START_FRACTION)
    boundaries = np.linspace(first_test, n, N_FOLDS + 1, dtype=int)

    equity = INITIAL_EQUITY
    peak_equity = equity
    overall_max_dd = 0.0

    trades = []
    fold_rows = []

    for fold in range(1, N_FOLDS + 1):
        start_i = int(boundaries[fold - 1])
        end_i = int(boundaries[fold])

        start_time = df.loc[start_i, "datetime"]
        end_time = (
            df.loc[end_i, "datetime"]
            if end_i < n
            else df.loc[end_i - 1, "datetime"]
                 + pd.Timedelta(nanoseconds=1)
        )

        start_equity = equity
        fold_peak = equity
        fold_max_dd = 0.0
        fold_trade_start = len(trades)

        # Mark-to-market equity while a position is open.
        i = start_i

        while i < end_i and equity > 0:
            side = int(df.loc[i, "signal"])

            if side == 0:
                i += 1
                continue

            entry_i = i + 1
            if entry_i >= end_i:
                break

            atr = float(df.loc[i, "atr"])
            entry = float(df.loc[entry_i, "open"])

            if not np.isfinite(atr) or atr <= 0 or entry <= 0:
                i += 1
                continue

            stop_distance = STOP_ATR_MULTIPLE * atr
            stop = entry - side * stop_distance
            target = entry + side * REWARD_RISK_RATIO * stop_distance

            if stop <= 0 or target <= 0:
                i += 1
                continue

            risk_budget = equity * RISK_FRACTION
            qty_by_risk = risk_budget / stop_distance
            qty_by_notional = equity * MAX_NOTIONAL_MULTIPLE / entry
            qty = min(qty_by_risk, qty_by_notional)

            if qty <= 0:
                i += 1
                continue

            exit_i, exit_price, exit_reason = find_exit(
                df, entry_i, end_i, side, stop, target
            )

            entry_time = df.loc[entry_i, "datetime"]
            exit_time = df.loc[exit_i, "datetime"]
            entry_notional = qty * entry

            # Charge half the round-trip cost at entry and half at exit.
            # This is an accounting convention; the total is unchanged.
            entry_cost = entry_notional * ROUND_TRIP_COST / 2
            exit_notional = qty * exit_price
            exit_cost = exit_notional * ROUND_TRIP_COST / 2

            equity_before = equity
            equity -= entry_cost

            position_equity_peak = equity
            position_max_dd = 0.0

            # Mark to market at every available candle close.
            for mark_i in range(entry_i, exit_i + 1):
                mark_price = float(df.loc[mark_i, "close"])
                unrealized = side * qty * (mark_price - entry)
                mark_equity = equity_before + unrealized - entry_cost

                position_equity_peak = max(
                    position_equity_peak, mark_equity
                )
                dd = (
                    mark_equity / position_equity_peak - 1
                    if position_equity_peak > 0 else -1
                )
                position_max_dd = min(position_max_dd, dd)

                fold_peak = max(fold_peak, mark_equity)
                fold_dd = (
                    mark_equity / fold_peak - 1
                    if fold_peak > 0 else -1
                )
                fold_max_dd = min(fold_max_dd, fold_dd)

                peak_equity = max(peak_equity, mark_equity)
                overall_dd = (
                    mark_equity / peak_equity - 1
                    if peak_equity > 0 else -1
                )
                overall_max_dd = min(overall_max_dd, overall_dd)

            gross_pnl = side * qty * (exit_price - entry)
            total_cost = entry_cost + exit_cost
            net_pnl = gross_pnl - total_cost

            equity = max(0.0, equity_before + net_pnl)
            fold_peak = max(fold_peak, equity)
            fold_dd = equity / fold_peak - 1 if fold_peak else -1
            fold_max_dd = min(fold_max_dd, fold_dd)

            peak_equity = max(peak_equity, equity)
            overall_dd = equity / peak_equity - 1 if peak_equity else -1
            overall_max_dd = min(overall_max_dd, overall_dd)

            trades.append({
                "fold": fold,
                "signal_time": df.loc[i, "datetime"],
                "direction": "LONG" if side == 1 else "SHORT",
                "entry_time": entry_time,
                "entry_price": entry,
                "atr_at_signal": atr,
                "stop_price": stop,
                "target_price": target,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "quantity_abstract_units": qty,
                "entry_notional": entry_notional,
                "gross_pnl": gross_pnl,
                "entry_cost": entry_cost,
                "exit_cost": exit_cost,
                "total_cost": total_cost,
                "net_pnl": net_pnl,
                "equity_before": equity_before,
                "equity_after": equity,
                "intratrade_marked_drawdown_pct": position_max_dd * 100,
            })

            # Do not open another trade until this position has closed.
            i = exit_i + 1

        ft = trades[fold_trade_start:]
        pnls = [t["net_pnl"] for t in ft]

        fold_rows.append({
            "fold": fold,
            "test_start": start_time,
            "test_end_exclusive": end_time,
            "trades": len(ft),
            "starting_equity": start_equity,
            "ending_equity": equity,
            "net_pnl": equity - start_equity,
            "return_pct": (
                (equity / start_equity - 1) * 100
                if start_equity > 0 else np.nan
            ),
            "win_rate_pct": (
                np.mean(np.asarray(pnls) > 0) * 100 if pnls else 0.0
            ),
            "profit_factor": profit_factor(pnls),
            "fold_max_drawdown_pct": fold_max_dd * 100,
        })

    trades_df = pd.DataFrame(trades)
    folds_df = pd.DataFrame(fold_rows)

    trades_df.to_csv(TRADES_OUT, index=False)
    folds_df.to_csv(FOLDS_OUT, index=False)

    net_pnls = (
        trades_df["net_pnl"].to_numpy(dtype=float)
        if not trades_df.empty else np.array([])
    )

    total_return = (equity / INITIAL_EQUITY - 1) * 100
    worst_trade_drawdown = (
        trades_df["intratrade_marked_drawdown_pct"].min()
        if not trades_df.empty else 0.0
    )

    lines = [
        "RISK-MANAGED BACKTEST V2",
        "=" * 45,
        f"Valid candles: {len(df)}",
        f"Initial equity: {INITIAL_EQUITY:.2f}",
        f"Final equity: {equity:.2f}",
        f"Net return: {total_return:.3f}%",
        f"Completed trades: {len(trades_df)}",
        f"Profit factor after costs: {profit_factor(net_pnls):.3f}",
        f"Overall marked-to-market max drawdown: {overall_max_dd * 100:.3f}%",
        f"Worst per-trade marked drawdown: {worst_trade_drawdown:.3f}%",
        f"Risk budget: {RISK_FRACTION * 100:.2f}% equity",
        f"Stop: {STOP_ATR_MULTIPLE:.2f} x ATR({ATR_PERIOD})",
        f"Reward/risk: 1:{REWARD_RISK_RATIO:.2f}",
        f"Round-trip cost: {ROUND_TRIP_COST * 100:.3f}% of entry notional",
        "",
        "EXIT REASONS",
        trades_df["exit_reason"].value_counts().to_string()
        if not trades_df.empty else "No trades",
        "",
        "FOLD RESULTS",
        folds_df.to_string(index=False) if not folds_df.empty else "No folds",
        "",
        "LIMITATIONS",
        "- Intrabar drawdown is approximated using candle closes, not every price tick.",
        "- Position sizing uses abstract units, not actual gold contract specifications.",
        "- Costs, slippage, spread, margin, and broker execution are simplified.",
        "- OHLC candles cannot reveal exact stop/target order when both are touched.",
        "- No trade carries across fold boundaries.",
        "- This is a fixed-rule historical simulation, not a profitability guarantee.",
    ]

    summary = "\n".join(lines)
    SUMMARY_OUT.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"\nSaved: {TRADES_OUT.name}")
    print(f"Saved: {FOLDS_OUT.name}")
    print(f"Saved: {SUMMARY_OUT.name}")


if __name__ == "__main__":
    main()
