
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "gold_data.csv"
TRADES_FILE = BASE / "sr_holdout_trades.csv"

SIGNAL_REPORT = BASE / "breakout_signal_audit.csv"
SUMMARY_REPORT = BASE / "breakout_signal_audit_summary.txt"

WINDOW = 20
HOLDOUT_FRACTION = 0.20
PRICE_TOLERANCE = 1e-8


def load_candles():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    if len(df) < WINDOW + 2:
        raise ValueError("Not enough candles for the audit.")

    return df


def load_trades():
    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRADES_FILE}")

    trades = pd.read_csv(TRADES_FILE)
    trades.columns = trades.columns.str.strip().str.lower()

    required = {
        "strategy",
        "signal_time",
        "direction",
        "entry_time",
        "entry_price",
    }
    missing = required - set(trades.columns)

    if missing:
        raise ValueError(
            f"Missing trade columns: {sorted(missing)}"
        )

    trades = trades[
        trades["strategy"].astype(str).str.lower() == "breakout_20"
    ].copy()

    for col in ["signal_time", "entry_time"]:
        trades[col] = pd.to_datetime(
            trades[col], utc=True, errors="coerce"
        )

    trades["entry_price"] = pd.to_numeric(
        trades["entry_price"], errors="coerce"
    )

    return trades.reset_index(drop=True)


def main():
    df = load_candles()

    # Recompute the exact intended breakout rule independently.
    resistance = (
        df["high"]
        .rolling(WINDOW, min_periods=WINDOW)
        .max()
        .shift(1)
    )

    support = (
        df["low"]
        .rolling(WINDOW, min_periods=WINDOW)
        .min()
        .shift(1)
    )

    previous_close = df["close"].shift(1)
    previous_resistance = resistance.shift(1)
    previous_support = support.shift(1)

    breakout_up = (
        (df["close"] > resistance)
        & (previous_close <= previous_resistance)
    )

    breakout_down = (
        (df["close"] < support)
        & (previous_close >= previous_support)
    )

    expected_signal = np.select(
        [breakout_up, breakout_down],
        [1, -1],
        default=0,
    )

    df["resistance20_prior"] = resistance
    df["support20_prior"] = support
    df["expected_breakout20"] = expected_signal

    # The holdout split must match the original backtest convention.
    split_index = int(len(df) * (1 - HOLDOUT_FRACTION))
    holdout_start = df.loc[split_index, "datetime"]

    trades = load_trades()
    rows = []

    candle_index = {
        t: i for i, t in enumerate(df["datetime"])
    }

    for _, trade in trades.iterrows():
        issues = []

        signal_time = trade["signal_time"]
        entry_time = trade["entry_time"]
        direction = str(trade["direction"]).strip().upper()

        signal_index = candle_index.get(signal_time)
        entry_index = candle_index.get(entry_time)

        expected_direction = None
        expected_entry_time = pd.NaT
        expected_entry_price = np.nan
        prior_resistance = np.nan
        prior_support = np.nan

        if signal_index is None:
            issues.append("Signal timestamp is not an exact source candle.")
        else:
            signal_value = int(df.loc[signal_index, "expected_breakout20"])
            prior_resistance = df.loc[
                signal_index, "resistance20_prior"
            ]
            prior_support = df.loc[signal_index, "support20_prior"]

            if signal_value == 1:
                expected_direction = "LONG"
            elif signal_value == -1:
                expected_direction = "SHORT"
            else:
                issues.append(
                    "No breakout_20 signal at recorded signal timestamp."
                )

            # Verify the formula against the source candle values.
            # Levels are rolling max/min shifted by one candle.
            if signal_index >= WINDOW:
                prior_highs = df.loc[
                    signal_index - WINDOW:signal_index - 1, "high"
                ]
                prior_lows = df.loc[
                    signal_index - WINDOW:signal_index - 1, "low"
                ]

                manual_resistance = prior_highs.max()
                manual_support = prior_lows.min()

                if not np.isclose(
                    prior_resistance,
                    manual_resistance,
                    atol=PRICE_TOLERANCE,
                    rtol=0,
                ):
                    issues.append("Resistance includes incorrect values.")

                if not np.isclose(
                    prior_support,
                    manual_support,
                    atol=PRICE_TOLERANCE,
                    rtol=0,
                ):
                    issues.append("Support includes incorrect values.")
            else:
                issues.append("Insufficient prior candles for level check.")

        if signal_index is not None:
            if signal_time < holdout_start:
                issues.append("Signal occurs before holdout start.")

            if signal_index + 1 < len(df):
                expected_entry_time = df.loc[
                    signal_index + 1, "datetime"
                ]
                expected_entry_price = float(
                    df.loc[signal_index + 1, "open"]
                )

                if entry_time != expected_entry_time:
                    issues.append(
                        "Entry timestamp is not the next source candle."
                    )

                if pd.notna(trade["entry_price"]) and not np.isclose(
                    float(trade["entry_price"]),
                    expected_entry_price,
                    atol=PRICE_TOLERANCE,
                    rtol=0,
                ):
                    issues.append(
                        "Entry price does not match next candle open."
                    )
            else:
                issues.append("Signal occurs on the last candle.")

        if (
            expected_direction is not None
            and direction != expected_direction
        ):
            issues.append(
                f"Direction mismatch: stored={direction}, "
                f"expected={expected_direction}."
            )

        rows.append({
            "signal_time": signal_time,
            "direction": direction,
            "expected_direction": expected_direction,
            "entry_time": entry_time,
            "expected_entry_time": expected_entry_time,
            "entry_price": trade["entry_price"],
            "expected_entry_price": expected_entry_price,
            "prior_resistance20": prior_resistance,
            "prior_support20": prior_support,
            "issue_count": len(issues),
            "issues": " | ".join(issues) if issues else "No detected issue",
        })

    report = pd.DataFrame(rows)
    report.to_csv(SIGNAL_REPORT, index=False)

    # Check all holdout candles, not only those that became trades.
    holdout = df.loc[split_index:].copy()
    all_signal_count = int(
        (holdout["expected_breakout20"] != 0).sum()
    )

    matched_trade_signals = set(
        trades["signal_time"].dropna().tolist()
    )

    recorded_signal_times = set(
        holdout.loc[
            holdout["expected_breakout20"] != 0, "datetime"
        ].tolist()
    )

    missing_signals = recorded_signal_times - matched_trade_signals

    # Some valid signals may have been skipped because a trade was active,
    # or because no valid exit candle existed. Therefore, a missing trade
    # is not automatically a signal-generation bug.
    summary_lines = [
        "BREAKOUT_20 SIGNAL AUDIT",
        "=" * 60,
        f"Source candles: {len(df)}",
        f"Holdout start: {holdout_start}",
        f"Holdout breakout signal candles: {all_signal_count}",
        f"Recorded breakout_20 trades: {len(trades)}",
        f"Trade records with issues: {int((report['issue_count'] > 0).sum())}",
        f"Total issue flags: {int(report['issue_count'].sum())}",
        f"Signal candles without a recorded trade: {len(missing_signals)}",
        "",
        "NOTE: Signals without trades may be skipped due to an existing",
        "position, unavailable exit candle, or other trade eligibility rule.",
        "This report audits recorded trades; it does not fully audit every",
        "trade-selection decision or establish profitability.",
        "",
        f"Detailed report: {SIGNAL_REPORT.name}",
    ]

    SUMMARY_REPORT.write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("       BREAKOUT_20 SIGNAL AUDIT")
    print("=" * 60)
    print(f"Source candles: {len(df)}")
    print(f"Holdout start: {holdout_start}")
    print(f"Holdout signal candles: {all_signal_count}")
    print(f"Recorded breakout trades: {len(trades)}")
    print(
        "Trades with detected issues:",
        int((report["issue_count"] > 0).sum()),
    )
    print("Total issue flags:", int(report["issue_count"].sum()))
    print(
        "Signal candles without recorded trades:",
        len(missing_signals),
    )

    problems = report[report["issue_count"] > 0]

    if problems.empty:
        print("\nNo issues detected in the recorded trades checked.")
    else:
        print("\n--- FIRST 20 ISSUES ---")
        print(problems.head(20).to_string(index=False))

    print("\nCreated:")
    print(SIGNAL_REPORT.name)
    print(SUMMARY_REPORT.name)
    print("\nResearch audit only; not a profitability test.")


if __name__ == "__main__":
    main()
