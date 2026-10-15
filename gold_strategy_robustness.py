
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


# ============================================================
# GOLD STRATEGY ROBUSTNESS TEST
# Research only — not a live trading system.
# ============================================================

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"

TRADES_OUT = BASE / "robustness_trades.csv"
FOLDS_OUT = BASE / "robustness_folds.csv"
SUMMARY_OUT = BASE / "robustness_summary.csv"

INITIAL_EQUITY = 100_000.0
RISK_FRACTION = 0.005
ATR_PERIOD = 14
MAX_HOLD_CANDLES = 24
MAX_NOTIONAL_MULTIPLE = 1.0
N_FOLDS = 5
TEST_START_FRACTION = 0.50

LOOKBACKS = [15, 20, 30]
STOP_MULTIPLES = [1.0, 1.5]
TARGET_MULTIPLES = [1.5, 2.0, 3.0]
ROUND_TRIP_COSTS = [0.0005, 0.0010, 0.0015]

# Run both the baseline and medium-volatility filter.
VARIANTS = ["baseline", "filtered"]
FILTER_BUCKET = "medium"


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

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    prev_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(
        ATR_PERIOD, min_periods=ATR_PERIOD
    ).mean()

    df["atr_pct"] = df["atr"] / df["close"] * 100

    return df


def add_signals(df, lookback):
    result = df.copy()

    resistance = result["high"].rolling(lookback).max().shift(1)
    support = result["low"].rolling(lookback).min().shift(1)

    previous_close = result["close"].shift(1)
    previous_resistance = resistance.shift(1)
    previous_support = support.shift(1)

    result["signal"] = 0

    long_signal = (
        (result["close"] > resistance)
        & (previous_close <= previous_resistance)
    )

    short_signal = (
        (result["close"] < support)
        & (previous_close >= previous_support)
    )

    result.loc[long_signal, "signal"] = 1
    result.loc[short_signal, "signal"] = -1

    return result


def classify_volatility(value, low_cutoff, high_cutoff):
    if value < low_cutoff:
        return "low"
    if value < high_cutoff:
        return "medium"
    return "high"


def find_exit(df, entry_i, fold_end, side, stop, target):
    last_i = min(
        entry_i + MAX_HOLD_CANDLES - 1,
        fold_end - 1,
        len(df) - 1,
    )

    for j in range(entry_i, last_i + 1):
        candle = df.iloc[j]
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])

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

        # Conservative assumption: stop first if both are touched.
        if stop_hit:
            return j, stop, "STOP"
        if target_hit:
            return j, target, "TARGET"

    return last_i, float(df.iloc[last_i]["close"]), "TIME_EXIT"


def profit_factor(pnls):
    pnls = np.asarray(pnls, dtype=float)
    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan

    return gains / losses


def run_test(
    df,
    lookback,
    stop_multiple,
    target_multiple,
    cost,
    variant,
):
    # Recalculate signals for this lookback configuration.
    data = add_signals(df, lookback)

    n = len(data)
    first_test = int(n * TEST_START_FRACTION)
    boundaries = np.linspace(
        first_test, n, N_FOLDS + 1, dtype=int
    )

    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    max_drawdown = 0.0

    all_trades = []
    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        start_i = int(boundaries[fold - 1])
        end_i = int(boundaries[fold])

        history = data.iloc[:start_i]["atr_pct"].dropna()

        if len(history) < 30:
            continue

        # Thresholds use only data before this fold.
        low_cutoff = float(history.quantile(1 / 3))
        high_cutoff = float(history.quantile(2 / 3))

        starting_equity = equity
        fold_max_drawdown = 0.0
        fold_trades = []
        i = start_i

        while i < end_i and equity > 0:
            side = int(data.loc[i, "signal"])

            if side == 0:
                i += 1
                continue

            atr = float(data.loc[i, "atr"])
            volatility = float(data.loc[i, "atr_pct"])

            if (
                not np.isfinite(atr)
                or atr <= 0
                or not np.isfinite(volatility)
            ):
                i += 1
                continue

            bucket = classify_volatility(
                volatility, low_cutoff, high_cutoff
            )

            if variant == "filtered" and bucket != FILTER_BUCKET:
                i += 1
                continue

            entry_i = i + 1

            if entry_i >= end_i:
                break

            entry = float(data.loc[entry_i, "open"])

            if not np.isfinite(entry) or entry <= 0:
                i += 1
                continue

            stop_distance = stop_multiple * atr
            stop = entry - side * stop_distance
            target = entry + side * target_multiple * stop_distance

            if stop <= 0 or target <= 0:
                i += 1
                continue

            risk_budget = equity * RISK_FRACTION
            qty_by_risk = risk_budget / stop_distance
            qty_by_notional = equity * MAX_NOTIONAL_MULTIPLE / entry
            qty = min(qty_by_risk, qty_by_notional)

            if not np.isfinite(qty) or qty <= 0:
                i += 1
                continue

            exit_i, exit_price, exit_reason = find_exit(
                data, entry_i, end_i, side, stop, target
            )

            gross_pnl = side * qty * (exit_price - entry)
            entry_cost = qty * entry * cost / 2
            exit_cost = qty * exit_price * cost / 2
            total_cost = entry_cost + exit_cost
            net_pnl = gross_pnl - total_cost

            equity_before = equity
            equity = max(0.0, equity + net_pnl)
            peak_equity = max(peak_equity, equity)

            dd = equity / peak_equity - 1 if peak_equity else -1
            fold_max_drawdown = min(fold_max_drawdown, dd)
            max_drawdown = min(max_drawdown, dd)

            trade = {
                "variant": variant,
                "lookback": lookback,
                "stop_atr": stop_multiple,
                "target_atr": target_multiple,
                "round_trip_cost_pct": cost * 100,
                "fold": fold,
                "signal_time": data.loc[i, "datetime"],
                "direction": "LONG" if side == 1 else "SHORT",
                "volatility_bucket": bucket,
                "entry_time": data.loc[entry_i, "datetime"],
                "entry_price": entry,
                "exit_time": data.loc[exit_i, "datetime"],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "quantity": qty,
                "gross_pnl": gross_pnl,
                "cost": total_cost,
                "net_pnl": net_pnl,
                "equity_before": equity_before,
                "equity_after": equity,
            }

            all_trades.append(trade)
            fold_trades.append(trade)

            # One position at a time.
            i = exit_i + 1

        fold_pnls = np.array(
            [t["net_pnl"] for t in fold_trades], dtype=float
        )

        fold_results.append(
            {
                "variant": variant,
                "lookback": lookback,
                "stop_atr": stop_multiple,
                "target_atr": target_multiple,
                "round_trip_cost_pct": cost * 100,
                "fold": fold,
                "start_time": data.loc[start_i, "datetime"],
                "end_time": data.loc[end_i - 1, "datetime"],
                "starting_equity": starting_equity,
                "ending_equity": equity,
                "fold_return_pct": (
                    (equity / starting_equity - 1) * 100
                    if starting_equity > 0 else np.nan
                ),
                "cumulative_return_pct": (
                    (equity / INITIAL_EQUITY - 1) * 100
                ),
                "trades": len(fold_trades),
                "win_rate_pct": (
                    (fold_pnls > 0).mean() * 100
                    if len(fold_pnls) else 0.0
                ),
                "profit_factor": profit_factor(fold_pnls),
                "fold_realized_drawdown_pct": fold_max_drawdown * 100,
                "cumulative_realized_drawdown_pct": max_drawdown * 100,
            }
        )

    return all_trades, fold_results


def main():
    df = load_data()

    total = (
        len(LOOKBACKS)
        * len(STOP_MULTIPLES)
        * len(TARGET_MULTIPLES)
        * len(ROUND_TRIP_COSTS)
        * len(VARIANTS)
    )

    print("=" * 70)
    print("GOLD STRATEGY ROBUSTNESS TEST")
    print("=" * 70)
    print(f"Valid candles: {len(df)}")
    print(f"Configurations: {total}")
    print(f"Initial equity per configuration: {INITIAL_EQUITY:,.2f}")
    print("This may take a little time...\n")

    all_trades = []
    all_folds = []
    summaries = []

    config_num = 0

    configurations = product(
        LOOKBACKS,
        STOP_MULTIPLES,
        TARGET_MULTIPLES,
        ROUND_TRIP_COSTS,
        VARIANTS,
    )

    for lookback, stop_mult, target_mult, cost, variant in configurations:
        config_num += 1

        print(
            f"[{config_num}/{total}] "
            f"lookback={lookback}, stop={stop_mult} ATR, "
            f"target={target_mult} ATR, cost={cost * 100:.2f}%, "
            f"variant={variant}"
        )

        trades, folds = run_test(
            df,
            lookback,
            stop_mult,
            target_mult,
            cost,
            variant,
        )

        all_trades.extend(trades)
        all_folds.extend(folds)

        pnls = np.array(
            [t["net_pnl"] for t in trades], dtype=float
        )

        final_equity = (
            folds[-1]["ending_equity"]
            if folds else INITIAL_EQUITY
        )

        summaries.append(
            {
                "variant": variant,
                "lookback": lookback,
                "stop_atr": stop_mult,
                "target_atr": target_mult,
                "round_trip_cost_pct": cost * 100,
                "initial_equity": INITIAL_EQUITY,
                "final_equity": final_equity,
                "total_return_pct": (
                    (final_equity / INITIAL_EQUITY - 1) * 100
                ),
                "trades": len(trades),
                "win_rate_pct": (
                    (pnls > 0).mean() * 100
                    if len(pnls) else 0.0
                ),
                "profit_factor": profit_factor(pnls),
                "cumulative_realized_drawdown_pct": (
                    folds[-1]["cumulative_realized_drawdown_pct"]
                    if folds else 0.0
                ),
                "positive_folds": sum(
                    f["fold_return_pct"] > 0 for f in folds
                ),
                "tested_folds": len(folds),
            }
        )

    trades_df = pd.DataFrame(all_trades)
    folds_df = pd.DataFrame(all_folds)
    summary_df = pd.DataFrame(summaries)

    trades_df.to_csv(TRADES_OUT, index=False)
    folds_df.to_csv(FOLDS_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    # A diagnostic ranking, not proof that the best configuration
    # will perform best on future data.
    ranked = summary_df.sort_values(
        ["total_return_pct", "profit_factor"],
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("TOP 10 HISTORICAL CONFIGURATIONS — DIAGNOSTIC ONLY")
    print("=" * 70)

    columns = [
        "variant",
        "lookback",
        "stop_atr",
        "target_atr",
        "round_trip_cost_pct",
        "total_return_pct",
        "profit_factor",
        "cumulative_realized_drawdown_pct",
        "trades",
        "positive_folds",
        "tested_folds",
    ]

    print(
        ranked[columns].head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nCreated files:")
    print(TRADES_OUT.name)
    print(FOLDS_OUT.name)
    print(SUMMARY_OUT.name)

    print(
        "\nResearch warning: configurations share the same historical "
        "dataset. Selecting the best result introduces selection bias. "
        "These are not live-trading recommendations."
    )


if __name__ == "__main__":
    main()
