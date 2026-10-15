
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_data.csv"

FOLDS_FILE = BASE / "cost_sensitivity_folds.csv"
SUMMARY_FILE = BASE / "cost_sensitivity_summary.csv"
TRADES_FILE = BASE / "cost_sensitivity_trades.csv"

INITIAL_EQUITY = 100_000.0
RISK_FRACTION = 0.005
ATR_PERIOD = 14
STOP_ATR_MULTIPLE = 1.0
REWARD_RISK_RATIO = 2.0
MAX_HOLD_CANDLES = 24
N_FOLDS = 5
TEST_START_FRACTION = 0.50
MAX_NOTIONAL_MULTIPLE = 1.0

# Round-trip cost rates: 0.05%, 0.10%, and 0.15%.
ROUND_TRIP_COSTS = [0.0005, 0.0010, 0.0015]

# Adverse slippage on EACH side of a trade.
# 1 basis point = 0.01%.
SLIPPAGE_BPS = [0, 2, 5, 10]

FILTER_BUCKET = "medium"


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE.name}")

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

        # Conservative rule when both levels are touched.
        if stop_hit:
            return j, stop, "STOP"
        if target_hit:
            return j, target, "TARGET"

    return (
        last_i,
        float(df.iloc[last_i]["close"]),
        "TIME_EXIT",
    )


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


def run_strategy(
    df, boundaries, variant, round_trip_cost, slippage_bps
):
    # Equity persists across all five folds for this strategy/scenario.
    equity = INITIAL_EQUITY
    peak_equity = equity
    max_drawdown = 0.0

    trades = []
    fold_rows = []

    slip = slippage_bps / 10_000

    for fold in range(1, N_FOLDS + 1):
        start_i = int(boundaries[fold - 1])
        end_i = int(boundaries[fold])

        start_equity = equity
        fold_peak = equity
        fold_max_dd = 0.0

        history = df.iloc[:start_i]["atr_pct"].dropna()
        if len(history) < 30:
            raise ValueError(
                f"Not enough past volatility history for fold {fold}"
            )

        # Thresholds use only observations before this fold.
        low_cutoff = float(history.quantile(1 / 3))
        high_cutoff = float(history.quantile(2 / 3))

        i = start_i

        while i < end_i and equity > 0:
            side = int(df.loc[i, "signal"])

            if side == 0:
                i += 1
                continue

            atr = float(df.loc[i, "atr"])
            atr_pct = float(df.loc[i, "atr_pct"])

            if (
                not np.isfinite(atr)
                or atr <= 0
                or not np.isfinite(atr_pct)
            ):
                i += 1
                continue

            bucket = classify_volatility(
                atr_pct, low_cutoff, high_cutoff
            )

            if variant == "filtered" and bucket != FILTER_BUCKET:
                i += 1
                continue

            entry_i = i + 1
            if entry_i >= end_i:
                break

            raw_entry = float(df.loc[entry_i, "open"])
            if raw_entry <= 0:
                i += 1
                continue

            # Slippage worsens the entry price.
            entry = raw_entry * (
                1 + slip if side == 1 else 1 - slip
            )

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

            if qty <= 0:
                i += 1
                continue

            exit_i, raw_exit, reason = find_exit(
                df, entry_i, end_i, side, stop, target
            )

            # Slippage worsens the exit price too.
            exit_price = raw_exit * (
                1 - slip if side == 1 else 1 + slip
            )

            entry_notional = qty * entry
            exit_notional = qty * exit_price

            # Round-trip cost is split between entry and exit.
            entry_cost = entry_notional * round_trip_cost / 2
            exit_cost = exit_notional * round_trip_cost / 2
            total_cost = entry_cost + exit_cost

            gross_pnl = side * qty * (exit_price - entry)
            net_pnl = gross_pnl - total_cost

            equity_before = equity
            equity = max(0.0, equity_before + net_pnl)

            # Realized-equity drawdown. This does not include
            # unrealized losses during an open trade.
            peak_equity = max(peak_equity, equity)
            dd = peak_equity / peak_equity - 1 if peak_equity else -1
            dd = equity / peak_equity - 1 if peak_equity else -1
            max_drawdown = min(max_drawdown, dd)

            fold_peak = max(fold_peak, equity)
            fold_dd = equity / fold_peak - 1 if fold_peak else -1
            fold_max_dd = min(fold_max_dd, fold_dd)

            trades.append({
                "variant": variant,
                "fold": fold,
                "round_trip_cost_pct": round_trip_cost * 100,
                "slippage_bps_per_side": slippage_bps,
                "signal_time": df.loc[i, "datetime"],
                "direction": "LONG" if side == 1 else "SHORT",
                "volatility": bucket,
                "atr_pct_at_signal": atr_pct,
                "entry_time": df.loc[entry_i, "datetime"],
                "raw_entry": raw_entry,
                "entry_price_after_slippage": entry,
                "atr_at_signal": atr,
                "stop_price": stop,
                "target_price": target,
                "exit_time": df.loc[exit_i, "datetime"],
                "raw_exit": raw_exit,
                "exit_price_after_slippage": exit_price,
                "exit_reason": reason,
                "quantity_abstract_units": qty,
                "gross_pnl": gross_pnl,
                "total_cost": total_cost,
                "net_pnl": net_pnl,
                "equity_before": equity_before,
                "equity_after": equity,
            })

            # Do not open another position before this one closes.
            i = exit_i + 1

        fold_trades = [
            t for t in trades if t["fold"] == fold
        ]
        # The list above includes only this variant's trades.
        # run_strategy handles one variant and one cost scenario.
        fold_pnls = np.array(
            [t["net_pnl"] for t in fold_trades], dtype=float
        )

        fold_rows.append({
            "variant": variant,
            "fold": fold,
            "test_start": df.loc[start_i, "datetime"],
            "test_end": df.loc[end_i - 1, "datetime"],
            "round_trip_cost_pct": round_trip_cost * 100,
            "slippage_bps_per_side": slippage_bps,
            "starting_equity": start_equity,
            "ending_equity": equity,
            "net_pnl": equity - start_equity,
            "return_pct": (
                (equity / start_equity - 1) * 100
                if start_equity > 0 else np.nan
            ),
            "trades": len(fold_trades),
            "wins": int((fold_pnls > 0).sum()),
            "win_rate_pct": (
                float((fold_pnls > 0).mean() * 100)
                if len(fold_pnls) else 0.0
            ),
            "profit_factor": profit_factor(fold_pnls),
            "realized_equity_drawdown_pct": fold_max_dd * 100,
            "training_low_cutoff_pct": low_cutoff,
            "training_high_cutoff_pct": high_cutoff,
        })

    all_pnls = np.array(
        [t["net_pnl"] for t in trades], dtype=float
    )

    summary = {
        "variant": variant,
        "round_trip_cost_pct": round_trip_cost * 100,
        "slippage_bps_per_side": slippage_bps,
        "initial_equity": INITIAL_EQUITY,
        "final_equity": equity,
        "total_net_pnl": equity - INITIAL_EQUITY,
        "total_return_pct": (
            (equity / INITIAL_EQUITY - 1) * 100
        ),
        "trades": len(trades),
        "wins": int((all_pnls > 0).sum()),
        "win_rate_pct": (
            float((all_pnls > 0).mean() * 100)
            if len(all_pnls) else 0.0
        ),
        "profit_factor": profit_factor(all_pnls),
        "realized_equity_max_drawdown_pct": max_drawdown * 100,
    }

    return trades, fold_rows, summary


def main():
    df = load_data()
    n = len(df)

    first_test = int(n * TEST_START_FRACTION)
    boundaries = np.linspace(
        first_test, n, N_FOLDS + 1, dtype=int
    )

    all_trades = []
    all_folds = []
    all_summaries = []

    for cost in ROUND_TRIP_COSTS:
        for slip_bps in SLIPPAGE_BPS:
            for variant in ["baseline", "filtered"]:
                print(
                    f"Running {variant}: "
                    f"round-trip cost={cost * 100:.2f}%, "
                    f"slippage={slip_bps} bps/side"
                )

                trades, folds, summary = run_strategy(
                    df,
                    boundaries,
                    variant,
                    cost,
                    slip_bps,
                )

                all_trades.extend(trades)
                all_folds.extend(folds)
                all_summaries.append(summary)

    trades_df = pd.DataFrame(all_trades)
    folds_df = pd.DataFrame(all_folds)
    summary_df = pd.DataFrame(all_summaries)

    trades_df.to_csv(TRADES_FILE, index=False)
    folds_df.to_csv(FOLDS_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\n=== OVERALL COST SENSITIVITY ===")
    print(summary_df.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== FOLD RESULTS ===")
    print(folds_df.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\nCreated:")
    print(TRADES_FILE.name)
    print(FOLDS_FILE.name)
    print(SUMMARY_FILE.name)

    print(
        "\nIMPORTANT: These are simulated results. Slippage is modeled "
        "as an adverse price adjustment on each side. Drawdown uses "
        "realized equity only, not intratrade unrealized losses. "
        "Verify signals against the original script before relying "
        "on this comparison."
    )


if __name__ == "__main__":
    main()
