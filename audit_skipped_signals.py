
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "gold_data.csv"
TRADES_FILE = BASE / "sr_holdout_trades.csv"

OUTPUT_FILE = BASE / "breakout_skipped_signals_audit.csv"

WINDOW = 20
NEAR_LEVEL = 0.002
HOLD_HOURS = 4
MAX_EXIT_DELAY_HOURS = 5
HOLDOUT_FRACTION = 0.20


def load_candles():
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

    return df


def load_trades():
    trades = pd.read_csv(TRADES_FILE)
    trades.columns = trades.columns.str.strip().str.lower()

    required = {
        "strategy", "signal_time", "entry_time", "exit_time", "direction"
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Missing trade columns: {sorted(missing)}")

    trades = trades[
        trades["strategy"].astype(str).str.lower() == "breakout_20"
    ].copy()

    for col in ["signal_time", "entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(
            trades[col], utc=True, errors="coerce"
        )

    return trades.sort_values("entry_time").reset_index(drop=True)


def generate_breakout_signals(df):
    resistance = (
        df["high"].rolling(WINDOW, min_periods=WINDOW).max().shift(1)
    )
    support = (
        df["low"].rolling(WINDOW, min_periods=WINDOW).min().shift(1)
    )

    prev_close = df["close"].shift(1)
    prev_resistance = resistance.shift(1)
    prev_support = support.shift(1)

    up = (df["close"] > resistance) & (prev_close <= prev_resistance)
    down = (df["close"] < support) & (prev_close >= prev_support)

    signal = np.select([up, down], [1, -1], default=0)

    return signal, resistance, support


def main():
    if not DATA_FILE.exists() or not TRADES_FILE.exists():
        raise FileNotFoundError(
            "Ensure gold_data.csv and sr_holdout_trades.csv "
            "exist in this folder."
        )

    df = load_candles()
    trades = load_trades()

    split_index = int(len(df) * (1 - HOLDOUT_FRACTION))
    holdout_start = df.loc[split_index, "datetime"]

    signal, resistance, support = generate_breakout_signals(df)
    df["signal"] = signal
    df["resistance20"] = resistance
    df["support20"] = support

    trade_signal_times = set(trades["signal_time"].dropna())

    # Match the original strategy's chronological trade-selection process.
    last_exit_time = None
    rows = []

    for i in range(split_index, len(df)):
        side = int(df.loc[i, "signal"])
        if side == 0:
            continue

        signal_time = df.loc[i, "datetime"]

        # If this signal already has a recorded trade, report it as recorded.
        if signal_time in trade_signal_times:
            status = "recorded_trade"
            reason = "Signal appears in saved trade file."
            entry_time = df.loc[i + 1, "datetime"] if i + 1 < len(df) else pd.NaT
            exit_time = pd.NaT

            matched = trades[trades["signal_time"] == signal_time]
            if not matched.empty:
                entry_time = matched.iloc[0]["entry_time"]
                exit_time = matched.iloc[0]["exit_time"]

            rows.append({
                "signal_time": signal_time,
                "direction": "LONG" if side == 1 else "SHORT",
                "status": status,
                "reason": reason,
                "entry_time": entry_time,
                "exit_time": exit_time,
            })
            last_exit_time = exit_time if pd.notna(exit_time) else last_exit_time
            continue

        entry_index = i + 1

        if entry_index >= len(df):
            rows.append({
                "signal_time": signal_time,
                "direction": "LONG" if side == 1 else "SHORT",
                "status": "skipped",
                "reason": "No next candle available for entry.",
                "entry_time": pd.NaT,
                "exit_time": pd.NaT,
            })
            continue

        entry_time = df.loc[entry_index, "datetime"]

        if entry_time < holdout_start:
            reason = "Entry precedes holdout start."
            exit_time = pd.NaT

        elif last_exit_time is not None and entry_time < last_exit_time:
            reason = "Skipped because a previous trade was still active."
            exit_time = pd.NaT

        else:
            target_time = entry_time + pd.Timedelta(hours=HOLD_HOURS)
            exit_index = int(
                df["datetime"].searchsorted(target_time, side="left")
            )

            if exit_index >= len(df):
                reason = "No candle available at or after the target exit."
                exit_time = pd.NaT
            else:
                candidate_exit = df.loc[exit_index, "datetime"]
                elapsed = (
                    candidate_exit - entry_time
                ).total_seconds() / 3600

                if elapsed < HOLD_HOURS or elapsed > MAX_EXIT_DELAY_HOURS:
                    reason = (
                        "No valid exit candle within the permitted "
                        "4-to-5-hour window."
                    )
                    exit_time = pd.NaT
                else:
                    # A valid exit is available. If the signal isn't recorded,
                    # the reason should be investigated rather than guessed.
                    reason = (
                        "Valid entry/exit appear available, but no saved trade "
                        "exists. Check original trade selection and CSV."
                    )
                    exit_time = candidate_exit

        rows.append({
            "signal_time": signal_time,
            "direction": "LONG" if side == 1 else "SHORT",
            "status": "skipped",
            "reason": reason,
            "entry_time": entry_time,
            "exit_time": exit_time,
        })

        # Do not advance last_exit_time for a skipped trade.
        # The original strategy advances it only after a trade is recorded.

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    skipped = report[report["status"] == "skipped"]

    print("\n" + "=" * 65)
    print("        BREAKOUT_20 SKIPPED SIGNAL DIAGNOSTIC")
    print("=" * 65)
    print(f"Holdout starts: {holdout_start}")
    print(f"All holdout signal candles: {(df.loc[split_index:, 'signal'] != 0).sum()}")
    print(f"Recorded trades in source file: {len(trades)}")
    print(f"Signals classified as skipped: {len(skipped)}")

    print("\n--- SKIPPED SIGNALS BY REASON ---")
    if skipped.empty:
        print("No skipped signals found.")
    else:
        print(skipped["reason"].value_counts().to_string())

        print("\n--- SKIPPED SIGNAL DETAILS ---")
        print(skipped.to_string(index=False))

    print(f"\nSaved detailed report: {OUTPUT_FILE.name}")
    print(
        "\nNote: this is a diagnostic reconstruction. "
        "Recorded trades and skipped signals must be reconciled "
        "against the original strategy loop before drawing conclusions."
    )


if __name__ == "__main__":
    main()
