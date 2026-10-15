
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"

DETAILS_OUT = BASE / "continuous_volatility_trades.csv"
FOLDS_OUT = BASE / "continuous_volatility_folds.csv"
SUMMARY_OUT = BASE / "continuous_volatility_summary.txt"
COMPARISON_OUT = BASE / "continuous_volatility_comparison.csv"

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

FILTER_BUCKET = "medium"


# ============================================================
# LOAD DATA AND BUILD SIGNALS
# ============================================================

def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE.name} in {BASE}"
        )

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

    if len(df) < 100:
        raise ValueError(
            "Not enough valid price candles to run the backtest."
        )

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

    # Original 20-candle breakout strategy.
    resistance = df["high"].rolling(20).max().shift(1)
    support = df["low"].rolling(20).min().shift(1)

    previous_close = df["close"].shift(1)
    previous_resistance = resistance.shift(1)
    previous_support = support.shift(1)

    df["signal"] = 0

    long_signal = (
        (df["close"] > resistance)
        & (previous_close <= previous_resistance)
    )

    short_signal = (
        (df["close"] < support)
        & (previous_close >= previous_support)
    )

    df.loc[long_signal, "signal"] = 1
    df.loc[short_signal, "signal"] = -1

    df["atr_pct"] = df["atr"] / df["close"] * 100

    return df


# ============================================================
# EXIT LOGIC
# ============================================================

def find_exit(df, entry_index, fold_end, side, stop, target):
    last_index = min(
        entry_index + MAX_HOLD_CANDLES - 1,
        fold_end - 1,
        len(df) - 1,
    )

    for j in range(entry_index, last_index + 1):
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

        # Conservative assumption when both levels are touched.
        if stop_hit:
            return j, stop, "STOP"

        if target_hit:
            return j, target, "TARGET"

    return (
        last_index,
        float(df.iloc[last_index]["close"]),
        "TIME_EXIT",
    )


# ============================================================
# METRICS
# ============================================================

def profit_factor(pnls):
    pnls = np.asarray(pnls, dtype=float)

    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan

    return gains / losses


def classify_volatility(value, low_cutoff, high_cutoff):
    if value < low_cutoff:
        return "low"

    if value < high_cutoff:
        return "medium"

    return "high"


# ============================================================
# BACKTEST ONE STRATEGY THROUGH ONE FOLD
# Equity and peak equity carry forward between folds.
# ============================================================

def run_variant(
    df,
    fold,
    start_i,
    end_i,
    low_cutoff,
    high_cutoff,
    use_filter,
    starting_equity,
    starting_peak_equity,
    starting_max_drawdown,
):
    equity = float(starting_equity)
    peak_equity = float(starting_peak_equity)

    cumulative_max_drawdown = float(starting_max_drawdown)
    fold_max_drawdown = 0.0

    trades = []
    i = start_i

    while i < end_i and equity > 0:
        side = int(df.loc[i, "signal"])

        if side == 0:
            i += 1
            continue

        atr = float(df.loc[i, "atr"])
        volatility = float(df.loc[i, "atr_pct"])

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        if not np.isfinite(volatility):
            i += 1
            continue

        bucket = classify_volatility(
            volatility, low_cutoff, high_cutoff
        )

        # Filter is applied at signal time, before entry.
        if use_filter and bucket != FILTER_BUCKET:
            i += 1
            continue

        entry_i = i + 1

        if entry_i >= end_i:
            break

        entry = float(df.loc[entry_i, "open"])

        if entry <= 0:
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
        qty_by_notional = (
            equity * MAX_NOTIONAL_MULTIPLE / entry
        )

        qty = min(qty_by_risk, qty_by_notional)

        if not np.isfinite(qty) or qty <= 0:
            i += 1
            continue

        exit_i, exit_price, exit_reason = find_exit(
            df, entry_i, end_i, side, stop, target
        )

        entry_notional = qty * entry
        exit_notional = qty * exit_price

        entry_cost = entry_notional * ROUND_TRIP_COST / 2
        exit_cost = exit_notional * ROUND_TRIP_COST / 2

        gross_pnl = side * qty * (exit_price - entry)
        total_cost = entry_cost + exit_cost
        net_pnl = gross_pnl - total_cost

        equity_before = equity
        equity = max(0.0, equity_before + net_pnl)

        peak_equity = max(peak_equity, equity)

        drawdown = (
            equity / peak_equity - 1.0
            if peak_equity > 0
            else -1.0
        )

        fold_max_drawdown = min(fold_max_drawdown, drawdown)
        cumulative_max_drawdown = min(
            cumulative_max_drawdown, drawdown
        )

        trades.append(
            {
                "variant": (
                    "filtered" if use_filter else "baseline"
                ),
                "fold": fold,
                "signal_time": df.loc[i, "datetime"],
                "direction": "LONG" if side == 1 else "SHORT",
                "volatility": bucket,
                "atr_pct_at_signal": volatility,
                "entry_time": df.loc[entry_i, "datetime"],
                "entry_price": entry,
                "atr_at_signal": atr,
                "stop_price": stop,
                "target_price": target,
                "exit_time": df.loc[exit_i, "datetime"],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "quantity_abstract_units": qty,
                "gross_pnl": gross_pnl,
                "total_cost": total_cost,
                "net_pnl": net_pnl,
                "equity_before": equity_before,
                "equity_after": equity,
                "peak_equity_after": peak_equity,
                "drawdown_pct_after": drawdown * 100,
            }
        )

        # Only one position at a time; no position crosses a fold.
        i = exit_i + 1

    pnls = np.array(
        [trade["net_pnl"] for trade in trades],
        dtype=float,
    )

    fold_return_pct = (
        (equity / starting_equity - 1.0) * 100
        if starting_equity > 0
        else np.nan
    )

    cumulative_return_pct = (
        (equity / INITIAL_EQUITY - 1.0) * 100
    )

    return {
        "trades": trades,
        "trade_count": len(trades),
        "wins": int((pnls > 0).sum()),
        "win_rate_pct": (
            float((pnls > 0).mean() * 100)
            if len(pnls)
            else 0.0
        ),
        "net_pnl": float(pnls.sum()),
        "starting_equity": starting_equity,
        "ending_equity": equity,
        "fold_return_pct": fold_return_pct,
        "cumulative_return_pct": cumulative_return_pct,
        "profit_factor": profit_factor(pnls),
        "fold_max_drawdown_pct": fold_max_drawdown * 100,
        "cumulative_max_drawdown_pct": (
            cumulative_max_drawdown * 100
        ),
        "peak_equity": peak_equity,
        "cumulative_max_drawdown": cumulative_max_drawdown,
    }


# ============================================================
# MAIN WALK-FORWARD TEST
# ============================================================

def main():
    df = load_data()
    n = len(df)

    first_test = int(n * TEST_START_FRACTION)

    boundaries = np.linspace(
        first_test, n, N_FOLDS + 1, dtype=int
    )

    all_trades = []
    fold_rows = []

    # Independent continuous accounts for the two strategies.
    states = {
        "baseline": {
            "equity": INITIAL_EQUITY,
            "peak_equity": INITIAL_EQUITY,
            "max_drawdown": 0.0,
        },
        "filtered": {
            "equity": INITIAL_EQUITY,
            "peak_equity": INITIAL_EQUITY,
            "max_drawdown": 0.0,
        },
    }

    print("=" * 70)
    print("CONTINUOUS-EQUITY VOLATILITY WALK-FORWARD BACKTEST")
    print("=" * 70)
    print(f"Valid candles: {n}")
    print(f"Initial equity per strategy: {INITIAL_EQUITY:,.2f}")
    print(f"Test folds: {N_FOLDS}")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.4%}")
    print(f"Risk per trade: {RISK_FRACTION:.2%}")
    print(
        "\nNote: drawdown is based on realized trade equity, "
        "not full intrabar mark-to-market equity."
    )

    for fold in range(1, N_FOLDS + 1):
        start_i = int(boundaries[fold - 1])
        end_i = int(boundaries[fold])

        if start_i >= end_i:
            continue

        # Thresholds use only history before this test fold.
        history = df.iloc[:start_i]["atr_pct"].dropna()

        if len(history) < 30:
            print(f"Skipping fold {fold}: insufficient history.")
            continue

        low_cutoff = float(history.quantile(1 / 3))
        high_cutoff = float(history.quantile(2 / 3))

        print(f"\n{'-' * 70}")
        print(f"FOLD {fold}")
        print(f"Start: {df.loc[start_i, 'datetime']}")
        print(f"End:   {df.loc[end_i - 1, 'datetime']}")
        print(
            f"Training thresholds: low < {low_cutoff:.5f}%, "
            f"medium < {high_cutoff:.5f}%"
        )

        fold_configs = [
            ("baseline", False),
            ("filtered", True),
        ]

        for name, use_filter in fold_configs:
            state = states[name]

            starting_equity = state["equity"]
            starting_peak_equity = state["peak_equity"]
            starting_max_drawdown = state["max_drawdown"]

            result = run_variant(
                df=df,
                fold=fold,
                start_i=start_i,
                end_i=end_i,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                use_filter=use_filter,
                starting_equity=starting_equity,
                starting_peak_equity=starting_peak_equity,
                starting_max_drawdown=starting_max_drawdown,
            )

            # Carry this strategy's equity into the next fold.
            state["equity"] = result["ending_equity"]
            state["peak_equity"] = result["peak_equity"]
            state["max_drawdown"] = (
                result["cumulative_max_drawdown"]
            )

            all_trades.extend(result["trades"])

            fold_rows.append(
                {
                    "fold": fold,
                    "test_start": df.loc[start_i, "datetime"],
                    "test_end": df.loc[end_i - 1, "datetime"],
                    "variant": name,
                    "training_low_cutoff_pct": low_cutoff,
                    "training_high_cutoff_pct": high_cutoff,
                    "starting_equity": starting_equity,
                    "ending_equity": result["ending_equity"],
                    "trades": result["trade_count"],
                    "wins": result["wins"],
                    "win_rate_pct": result["win_rate_pct"],
                    "net_pnl": result["net_pnl"],
                    "fold_return_pct": result["fold_return_pct"],
                    "cumulative_return_pct": (
                        result["cumulative_return_pct"]
                    ),
                    "profit_factor": result["profit_factor"],
                    "fold_max_drawdown_pct": (
                        result["fold_max_drawdown_pct"]
                    ),
                    "cumulative_max_drawdown_pct": (
                        result["cumulative_max_drawdown_pct"]
                    ),
                }
            )

            print(
                f"{name:8s} | "
                f"start={starting_equity:,.2f} | "
                f"end={result['ending_equity']:,.2f} | "
                f"fold return={result['fold_return_pct']:.3f}% | "
                f"cumulative return="
                f"{result['cumulative_return_pct']:.3f}% | "
                f"trades={result['trade_count']}"
            )

    # Save detailed trades and fold results.
    trades_df = pd.DataFrame(all_trades)
    folds_df = pd.DataFrame(fold_rows)

    trades_df.to_csv(DETAILS_OUT, index=False)
    folds_df.to_csv(FOLDS_OUT, index=False)

    # Final comparison, using the carried-forward account equity.
    summary_rows = []

    for name in ["baseline", "filtered"]:
        group = trades_df[
            trades_df["variant"] == name
        ] if not trades_df.empty else pd.DataFrame()

        pnls = (
            group["net_pnl"].to_numpy(dtype=float)
            if not group.empty
            else np.array([], dtype=float)
        )

        state = states[name]
        final_equity = state["equity"]

        summary_rows.append(
            {
                "variant": name,
                "initial_equity": INITIAL_EQUITY,
                "final_equity": final_equity,
                "total_return_pct": (
                    final_equity / INITIAL_EQUITY - 1
                ) * 100,
                "trades": len(group),
                "wins": int((pnls > 0).sum()),
                "win_rate_pct": (
                    float((pnls > 0).mean() * 100)
                    if len(pnls)
                    else 0.0
                ),
                "total_net_pnl": float(pnls.sum()),
                "profit_factor": profit_factor(pnls),
                "realized_equity_max_drawdown_pct": (
                    state["max_drawdown"] * 100
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(COMPARISON_OUT, index=False)

    print(f"\n{'=' * 70}")
    print("FINAL CONTINUOUS-EQUITY COMPARISON")
    print("=" * 70)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nCreated output files:")
    print(f"- {DETAILS_OUT.name}")
    print(f"- {FOLDS_OUT.name}")
    print(f"- {COMPARISON_OUT.name}")

    SUMMARY_OUT.write_text(
        "Continuous-equity walk-forward volatility comparison.\n"
        "Both variants start with the same initial equity and "
        "carry their own equity forward across folds.\n"
        "Volatility thresholds use data strictly before each fold.\n"
        "The original breakout, ATR, stop, target, position-sizing, "
        "and transaction-cost rules are preserved.\n"
        "Positions do not cross fold boundaries.\n"
        "Drawdown measures realized trade equity, not complete "
        "intrabar mark-to-market drawdown.\n"
        "This is historical research, not a guarantee of future results.\n",
        encoding="utf-8",
    )

    print(f"- {SUMMARY_OUT.name}")


if __name__ == "__main__":
    main()
