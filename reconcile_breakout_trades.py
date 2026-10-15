
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "gold_data.csv"
TRADES_FILE = BASE / "sr_holdout_trades.csv"
AUDIT_FILE = BASE / "breakout_reconciliation_audit.csv"
SUMMARY_FILE = BASE / "breakout_reconciliation_summary.txt"

HOLDOUT_FRACTION = 0.20
HOLD_HOURS = 4
MAX_EXIT_DELAY_HOURS = 5
WINDOW = 20


def main():
    for path in [DATA_FILE, TRADES_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.name}")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    trades = pd.read_csv(TRADES_FILE)
    trades.columns = trades.columns.str.strip().str.lower()
    trades = trades[
        trades["strategy"].astype(str).str.lower() == "breakout_20"
    ].copy()

    for col in ["signal_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")

    split_index = int(len(df) * (1 - HOLDOUT_FRACTION))
    holdout_start = df.loc[split_index, "datetime"]

    resistance = df["high"].rolling(WINDOW, min_periods=WINDOW).max().shift(1)
    support = df["low"].rolling(WINDOW, min_periods=WINDOW).min().shift(1)
    prev_close = df["close"].shift(1)

    up = (df["close"] > resistance) & (
        prev_close <= resistance.shift(1)
    )
    down = (df["close"] < support) & (
        prev_close >= support.shift(1)
    )
    df["signal"] = np.select([up, down], [1, -1], default=0)

    recorded = {}
    duplicate_recorded_signals = []

    for _, trade in trades.iterrows():
        signal_time = trade["signal_time"]
        if pd.isna(signal_time):
            continue
        if signal_time in recorded:
            duplicate_recorded_signals.append(signal_time)
        recorded.setdefault(signal_time, []).append(trade)

    last_exit_time = None
    rows = []
    matched_recorded = set()

    for i in range(split_index, len(df)):
        side = int(df.loc[i, "signal"])
        if side == 0:
            continue

        signal_time = df.loc[i, "datetime"]
        direction = "LONG" if side == 1 else "SHORT"
        entry_index = i + 1

        row = {
            "signal_time": signal_time,
            "direction": direction,
            "recorded_trade_found": False,
            "expected_status": "",
            "expected_reason": "",
            "expected_entry_time": pd.NaT,
            "expected_exit_time": pd.NaT,
            "saved_entry_time": pd.NaT,
            "saved_exit_time": pd.NaT,
            "saved_direction": "",
            "reconciliation": "",
        }

        # Reconstruct original strategy selection order.
        if entry_index >= len(df):
            reason = "No next candle for entry"
            expected_status = "SKIPPED"
            expected_exit = pd.NaT
            expected_entry = pd.NaT

        else:
            expected_entry = df.loc[entry_index, "datetime"]

            if expected_entry < holdout_start:
                reason = "Entry before holdout"
                expected_status = "SKIPPED"
                expected_exit = pd.NaT

            elif (
                last_exit_time is not None
                and expected_entry < last_exit_time
            ):
                reason = "Previous trade still active"
                expected_status = "SKIPPED"
                expected_exit = pd.NaT

            else:
                target = expected_entry + pd.Timedelta(hours=HOLD_HOURS)
                exit_index = int(
                    df["datetime"].searchsorted(target, side="left")
                )

                if exit_index >= len(df):
                    reason = "No candle available for exit"
                    expected_status = "SKIPPED"
                    expected_exit = pd.NaT
                else:
                    candidate_exit = df.loc[exit_index, "datetime"]
                    elapsed = (
                        candidate_exit - expected_entry
                    ).total_seconds() / 3600

                    if elapsed < HOLD_HOURS or elapsed > MAX_EXIT_DELAY_HOURS:
                        reason = "Exit outside 4-to-5-hour window"
                        expected_status = "SKIPPED"
                        expected_exit = pd.NaT
                    else:
                        reason = "Valid entry and exit"
                        expected_status = "TRADE"
                        expected_exit = candidate_exit

        saved_list = recorded.get(signal_time, [])
        saved = saved_list[0] if saved_list else None

        row["recorded_trade_found"] = saved is not None
        row["expected_status"] = expected_status
        row["expected_reason"] = reason
        row["expected_entry_time"] = expected_entry
        row["expected_exit_time"] = expected_exit

        if saved is not None:
            matched_recorded.add(signal_time)
            row["saved_entry_time"] = saved["entry_time"]
            row["saved_exit_time"] = saved["exit_time"]
            row["saved_direction"] = str(saved["direction"])

        if expected_status == "TRADE" and saved is None:
            row["reconciliation"] = "MISMATCH: expected trade missing"
        elif expected_status == "SKIPPED" and saved is not None:
            row["reconciliation"] = "MISMATCH: saved trade was expected skipped"
        elif saved is not None:
            checks = [
                saved["entry_time"] == expected_entry,
                saved["exit_time"] == expected_exit,
                str(saved["direction"]).upper() == direction,
            ]
            row["reconciliation"] = (
                "PASS: timestamps and direction match"
                if all(checks)
                else "MISMATCH: saved trade details differ"
            )
        else:
            row["reconciliation"] = "PASS: no trade expected"

        rows.append(row)

        # Match original loop: only accepted trades advance last_exit_time.
        if expected_status == "TRADE":
            last_exit_time = expected_exit

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_FILE, index=False)

    unmatched_saved = set(recorded) - matched_recorded
    mismatches = audit[
        audit["reconciliation"].str.startswith("MISMATCH")
    ]

    summary = [
        "BREAKOUT_20 TRADE RECONCILIATION",
        "=" * 45,
        f"Valid candles: {len(df)}",
        f"Holdout start: {holdout_start}",
        f"Signal candles in holdout: {len(audit)}",
        f"Saved breakout_20 trades: {len(trades)}",
        f"Signals matched to saved trades: {len(matched_recorded)}",
        f"Saved signal timestamps not found in reconstructed signals: {len(unmatched_saved)}",
        f"Duplicate saved signal timestamps: {len(duplicate_recorded_signals)}",
        f"Reconciliation mismatches: {len(mismatches)}",
        "",
        "Expected status counts:",
        audit["expected_status"].value_counts().to_string(),
        "",
        "Reconciliation results:",
        audit["reconciliation"].value_counts().to_string(),
    ]

    if not mismatches.empty:
        summary.extend(["", "MISMATCH DETAILS:", mismatches.to_string(index=False)])

    if unmatched_saved:
        summary.extend(["", "UNMATCHED SAVED SIGNAL TIMES:"])
        summary.extend(str(t) for t in sorted(unmatched_saved))

    if duplicate_recorded_signals:
        summary.extend(["", "DUPLICATE SAVED SIGNAL TIMES:"])
        summary.extend(str(t) for t in duplicate_recorded_signals)

    SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nCreated: {AUDIT_FILE.name}")
    print(f"Created: {SUMMARY_FILE.name}")


if __name__ == "__main__":
    main()
