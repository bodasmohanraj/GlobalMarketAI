
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"

TRADES_OUT = BASE / "risk_managed_trades.csv"
FOLDS_OUT = BASE / "risk_managed_fold_results.csv"
SUMMARY_OUT = BASE / "risk_managed_summary.txt"

INITIAL_EQUITY = 100_000.0
RISK_FRACTION = 0.005       # 0.5% of current equity
ATR_PERIOD = 14
STOP_ATR_MULTIPLE = 1.0
REWARD_RISK_RATIO = 2.0
ROUND_TRIP_COST = 0.0005    # 0.05% of entry notional per completed trade

MAX_HOLD_CANDLES = 24
N_FOLDS = 5
TEST_START_FRACTION = 0.50

# Research-only exposure cap: position notional cannot exceed equity.
# Quantity is an abstract unit, NOT a real GC futures contract quantity.
MAX_NOTIONAL_MULTIPLE = 1.0


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing file: {SOURCE}")

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

    if len(df) < 500:
        raise ValueError("Not enough valid candles.")

    return df


def build_indicators(df):
    df = df.copy()

    # Breakout levels use only candles preceding the signal candle.
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

    # ATR uses the signal candle and earlier candles only.
    previous_close = df["close"].shift(1)

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


def calculate_exit(df, entry_index, end_index, side, entry, stop, target):
    """
    Search forward for stop, target, or maximum holding-period exit.

    end_index is exclusive. No trade can use candles from the next fold.
    Stop wins if both stop and target are touched within the same candle.
    Stop gaps are filled at the worse opening price.
    Target gaps are conservatively filled at the target price.
    """
    last_index = min(
        entry_index + MAX_HOLD_CANDLES,
        end_index - 1,
        len(df) - 1,
    )

    for j in range(entry_index, last_index + 1):
        candle = df.iloc[j]
        open_price = float(candle["open"])
        high = float(candle["high"])
        low = float(candle["low"])

        if side == 1:
            # Long position
            if open_price <= stop:
                return j, open_price, "STOP_GAP"
            if open_price >= target:
                return j, target, "TARGET"
            stop_hit = low <= stop
            target_hit = high >= target

            if stop_hit:
                return j, stop, "STOP"
            if target_hit:
                return j, target, "TARGET"

        else:
            # Short position
            if open_price >= stop:
                return j, open_price, "STOP_GAP"
            if open_price <= target:
                return j, target, "TARGET"
            stop_hit = high >= stop
            target_hit = low <= target

            if stop_hit:
                return j, stop, "STOP"
            if target_hit:
                return j, target, "TARGET"

    # Neither barrier was touched: close at the last eligible candle close.
    return last_index, float(df.iloc[last_index]["close"]), "TIME_EXIT"


def profit_factor(net_pnls):
    net_pnls = np.asarray(net_pnls, dtype=float)
    gains = net_pnls[net_pnls > 0].sum()
    losses = -net_pnls[net_pnls < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan

    return float(gains / losses)


def max_drawdown_pct(equity_values):
    values = np.asarray(equity_values, dtype=float)

    if len(values) == 0:
        return 0.0

    peaks = np.maximum.accumulate(values)
    drawdowns = np.where(peaks > 0, values / peaks - 1, 0)

    return float(drawdowns.min() * 100)


def main():
    df = build_indicators(load_data())

    n = len(df)
    first_test = int(n * TEST_START_FRACTION)
    boundaries = np.linspace(
        first_test, n, N_FOLDS + 1, dtype=int
    )

    equity = INITIAL_EQUITY
    equity_curve = [equity]
    all_trades = []
    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        start_index = int(boundaries[fold - 1])
        end_index = int(boundaries[fold])

        fold_start_time = df.loc[start_index, "datetime"]
        fold_end_time = (
            df.loc[end_index, "datetime"]
            if end_index < n
            else df.loc[end_index - 1, "datetime"]
                 + pd.Timedelta(nanoseconds=1)
        )

        starting_equity = equity
        fold_trade_start = len(all_trades)
        i = start_index

        while i < end_index:
            side = int(df.loc[i, "signal"])

            if side == 0:
                i += 1
                continue

            # Signal is known at this candle's close.
            # Execute at the next candle's open.
            entry_index = i + 1

            if entry_index >= end_index:
                break

            atr = float(df.loc[i, "atr"])
            entry = float(df.loc[entry_index, "open"])

            if not np.isfinite(atr) or atr <= 0 or entry <= 0:
                i += 1
                continue

            stop_distance = STOP_ATR_MULTIPLE * atr

            if side == 1:
                stop = entry - stop_distance
                target = entry + REWARD_RISK_RATIO * stop_distance
            else:
                stop = entry + stop_distance
                target = entry - REWARD_RISK_RATIO * stop_distance

            if stop <= 0 or target <= 0:
                i += 1
                continue

            risk_budget = equity * RISK_FRACTION
            quantity_by_risk = risk_budget / stop_distance

            # Exposure cap prevents unrealistically large notional positions.
            max_quantity = (
                equity * MAX_NOTIONAL_MULTIPLE / entry
            )
            quantity = min(quantity_by_risk, max_quantity)

            if quantity <= 0:
                i += 1
                continue

            exit_index, exit_price, exit_reason = calculate_exit(
                df=df,
                entry_index=entry_index,
                end_index=end_index,
                side=side,
                entry=entry,
                stop=stop,
                target=target,
            )

            exit_time = df.loc[exit_index, "datetime"]
            entry_time = df.loc[entry_index, "datetime"]

            gross_pnl = side * quantity * (exit_price - entry)
            entry_notional = quantity * entry
            cost = entry_notional * ROUND_TRIP_COST
            net_pnl = gross_pnl - cost

            equity_before = equity
            equity += net_pnl

            # A loss cannot be treated as a positive equity balance.
            if equity < 0:
                equity = 0.0

            equity_curve.append(equity)

            planned_risk = quantity * stop_distance
            realized_r = (
                net_pnl / planned_risk if planned_risk > 0 else np.nan
            )

            all_trades.append({
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
                "holding_candles": exit_index - entry_index + 1,
                "quantity_abstract_units": quantity,
                "entry_notional": entry_notional,
                "planned_risk_amount": planned_risk,
                "gross_pnl": gross_pnl,
                "transaction_cost": cost,
                "net_pnl": net_pnl,
                "realized_R_multiple": realized_r,
                "equity_before": equity_before,
                "equity_after": equity,
            })

            # Skip signals while this position is open.
            i = exit_index + 1

            if equity <= 0:
                break

        fold_trades = all_trades[fold_trade_start:]
        pnls = [t["net_pnl"] for t in fold_trades]
        ending_equity = equity

        fold_results.append({
            "fold": fold,
            "test_start": fold_start_time,
            "test_end_exclusive": fold_end_time,
            "trades": len(fold_trades),
            "starting_equity": starting_equity,
            "ending_equity": ending_equity,
            "net_pnl": ending_equity - starting_equity,
            "return_pct": (
                (ending_equity / starting_equity - 1) * 100
                if starting_equity > 0 else np.nan
            ),
            "win_rate_pct": (
                np.mean(np.asarray(pnls) > 0) * 100
                if pnls else 0.0
            ),
            "profit_factor": profit_factor(pnls),
            "max_drawdown_pct": max_drawdown_pct(equity_curve),
        })

        if equity <= 0:
            print("Equity reached zero; remaining folds cannot be traded.")
            break

    trades_df = pd.DataFrame(all_trades)
    folds_df = pd.DataFrame(fold_results)

    trades_df.to_csv(TRADES_OUT, index=False)
    folds_df.to_csv(FOLDS_OUT, index=False)

    net_pnls = (
        trades_df["net_pnl"].to_numpy(dtype=float)
        if not trades_df.empty else np.array([])
    )
    total_return = (
        (equity / INITIAL_EQUITY - 1) * 100
    )

    summary = [
        "RISK-MANAGED BREAKOUT BACKTEST",
        "=" * 45,
        f"Valid candles: {len(df)}",
        f"Initial simulated equity: {INITIAL_EQUITY:.2f}",
        f"Final simulated equity: {equity:.2f}",
        f"Total return after assumed costs: {total_return:.3f}%",
        f"Completed trades: {len(trades_df)}",
        f"Risk budget: {RISK_FRACTION * 100:.2f}% of current equity",
        f"Stop distance: {STOP_ATR_MULTIPLE:.2f} x ATR({ATR_PERIOD})",
        f"Reward/risk ratio: 1:{REWARD_RISK_RATIO:.2f}",
        f"Round-trip cost: {ROUND_TRIP_COST * 100:.3f}% of entry notional",
        f"Maximum notional exposure: {MAX_NOTIONAL_MULTIPLE:.1f} x equity",
        f"Maximum holding: {MAX_HOLD_CANDLES} available candles",
        f"Profit factor after costs: {profit_factor(net_pnls):.3f}",
        f"Maximum drawdown: {max_drawdown_pct(equity_curve):.3f}%",
        "",
        "Exit reasons:",
        (
            trades_df["exit_reason"].value_counts().to_string()
            if not trades_df.empty else "No trades"
        ),
        "",
        "Fold results:",
        (
            folds_df.to_string(index=False)
            if not folds_df.empty else "No fold results"
        ),
        "",
        "LIMITATIONS",
        "- Quantity is abstract; this is not actual GC futures contract sizing.",
        "- No real broker margin, exchange fees, spread, or slippage model.",
        "- Cost is charged once per completed trade as specified.",
        "- Stop/target order priority is approximated from OHLC candles.",
        "- ATR stop distance and maximum holding period are assumptions.",
        "- The final candle's completeness has not been verified.",
        "- Historical performance does not establish a profitable edge.",
    ]

    summary_text = "\n".join(summary)
    SUMMARY_OUT.write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)
    print(f"\nSaved: {TRADES_OUT.name}")
    print(f"Saved: {FOLDS_OUT.name}")
    print(f"Saved: {SUMMARY_OUT.name}")


if __name__ == "__main__":
    main()

