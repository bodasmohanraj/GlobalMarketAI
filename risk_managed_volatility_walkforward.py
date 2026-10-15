
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"

DETAILS_OUT = BASE / "volatility_filter_trades.csv"
FOLDS_OUT = BASE / "volatility_filter_folds.csv"
SUMMARY_OUT = BASE / "volatility_filter_summary.txt"

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

# Test only the medium-volatility filter.
# Its boundaries are learned from training data, never the test fold.
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

    # This reproduces the baseline breakout signal.
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

    # Threshold calculation uses the ATR percentage available at
    # the signal candle. Thresholds will be calculated separately
    # for each fold from data strictly before that fold.
    df["atr_pct"] = df["atr"] / df["close"] * 100

    return df


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

        # Conservative assumption if both are hit in one candle.
        if stop_hit:
            return j, stop, "STOP"
        if target_hit:
            return j, target, "TARGET"

    return last_index, float(df.iloc[last_index]["close"]), "TIME_EXIT"


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


def run_variant(df, fold, start_i, end_i, low_cutoff,
                high_cutoff, use_filter):
    equity = INITIAL_EQUITY
    peak_equity = equity
    max_drawdown = 0.0
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

        # Apply filter at the signal candle, before entry.
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
        qty_by_notional = equity * MAX_NOTIONAL_MULTIPLE / entry
        qty = min(qty_by_risk, qty_by_notional)

        if qty <= 0:
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
        dd = equity / peak_equity - 1 if peak_equity else -1
        max_drawdown = min(max_drawdown, dd)

        trades.append({
            "variant": "filtered" if use_filter else "baseline",
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
        })

        # One open position per variant at a time.
        i = exit_i + 1

    pnls = np.array([t["net_pnl"] for t in trades], dtype=float)

    return {
        "trades": trades,
        "trade_count": len(trades),
        "wins": int((pnls > 0).sum()),
        "win_rate_pct": float((pnls > 0).mean() * 100)
        if len(pnls) else 0.0,
        "net_pnl": float(pnls.sum()),
        "ending_equity": equity,
        "return_pct": (equity / INITIAL_EQUITY - 1) * 100,
        "profit_factor": profit_factor(pnls),
        "max_drawdown_pct": max_drawdown * 100,
    }


def main():
    df = load_data()
    n = len(df)

    first_test = int(n * TEST_START_FRACTION)
    boundaries = np.linspace(
        first_test, n, N_FOLDS + 1, dtype=int
    )

    all_trades = []
    fold_rows = []

    for fold in range(1, N_FOLDS + 1):
        start_i = int(boundaries[fold - 1])
        end_i = int(boundaries[fold])

        # Only candles strictly before the test fold define thresholds.
        history = df.iloc[:start_i]["atr_pct"].dropna()

        if len(history) < 30:
            print(f"Skipping fold {fold}: insufficient history.")
            continue

        low_cutoff = float(history.quantile(1 / 3))
        high_cutoff = float(history.quantile(2 / 3))

        print(
            f"\nFold {fold}: "
            f"{df.loc[start_i, 'datetime']} to "
            f"{df.loc[end_i - 1, 'datetime']}"
        )
        print(
            f"Training-only thresholds: "
            f"low < {low_cutoff:.5f}%, "
            f"medium < {high_cutoff:.5f}%"
        )

        baseline = run_variant(
            df, fold, start_i, end_i,
            low_cutoff, high_cutoff, False
        )
        filtered = run_variant(
            df, fold, start_i, end_i,
            low_cutoff, high_cutoff, True
        )

        all_trades.extend(baseline["trades"])
        all_trades.extend(filtered["trades"])

        for name, result in [
            ("baseline", baseline),
            ("filtered", filtered),
        ]:
            fold_rows.append({
                "fold": fold,
                "test_start": df.loc[start_i, "datetime"],
                "test_end": df.loc[end_i - 1, "datetime"],
                "variant": name,
                "training_low_cutoff_pct": low_cutoff,
                "training_high_cutoff_pct": high_cutoff,
                "trades": result["trade_count"],
                "wins": result["wins"],
                "win_rate_pct": result["win_rate_pct"],
                "net_pnl": result["net_pnl"],
                "return_pct": result["return_pct"],
                "profit_factor": result["profit_factor"],
                "max_drawdown_pct": result["max_drawdown_pct"],
            })

    trades_df = pd.DataFrame(all_trades)
    folds_df = pd.DataFrame(fold_rows)

    trades_df.to_csv(DETAILS_OUT, index=False)
    folds_df.to_csv(FOLDS_OUT, index=False)

    print("\n=== WALK-FORWARD RESULTS BY FOLD ===")
    print(folds_df.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    summary_rows = []

    for variant, group in trades_df.groupby("variant"):
        pnls = group["net_pnl"].to_numpy(dtype=float)
        summary_rows.append({
            "variant": variant,
            "trades": len(group),
            "wins": int((pnls > 0).sum()),
            "win_rate_pct": float((pnls > 0).mean() * 100)
            if len(pnls) else 0.0,
            "net_pnl": float(pnls.sum()),
            "profit_factor": profit_factor(pnls),
        })

    summary = pd.DataFrame(summary_rows)

    print("\n=== OVERALL COMPARISON ===")
    print(summary.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    summary.to_csv(BASE / "volatility_filter_comparison.csv", index=False)

    SUMMARY_OUT.write_text(
        "Research-only walk-forward volatility comparison.\n"
        "Thresholds are learned from history before each test fold.\n"
        "Each variant starts each fold with the same initial equity.\n"
        "The model uses the same signal and exit rules for both variants.\n"
        "Review drawdown and fold-level results, not just total P&L.\n",
        encoding="utf-8",
    )

    print("\nCreated:")
    print(DETAILS_OUT.name)
    print(FOLDS_OUT.name)
    print("volatility_filter_comparison.csv")
    print(SUMMARY_OUT.name)


if __name__ == "__main__":
    main()
