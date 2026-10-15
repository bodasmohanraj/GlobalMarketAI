
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# BREAKOUT_20 TRADE AUDIT
# Read-only audit: does not modify existing input files.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "gold_data.csv"
TRADES_FILE = BASE_DIR / "sr_holdout_trades.csv"

AUDIT_FILE = BASE_DIR / "breakout_trade_audit.csv"
SUMMARY_FILE = BASE_DIR / "breakout_trade_audit_summary.txt"

STRATEGY_NAME = "breakout_20"
HOLDOUT_START = pd.Timestamp("2026-04-22 10:00:00", tz="UTC")

TARGET_HOURS = 4
MAX_EXIT_HOURS = 5
TIME_TOLERANCE_SECONDS = 60
PRICE_TOLERANCE = 1e-6
RETURN_TOLERANCE_PCT = 0.002

# ============================================================
# HELPERS
# ============================================================

def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in frame.columns
    ]
    return frame


def find_column(frame, candidates):
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def parse_time(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def close_enough(a, b, tolerance=PRICE_TOLERANCE):
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=0, atol=tolerance))


def add_issue(issues, label, condition, message):
    if condition:
        issues.append(f"{label}: {message}")


def price_at(candles, timestamp, column):
    rows = candles.loc[candles["datetime"] == timestamp, column]
    if rows.empty:
        return np.nan
    return rows.iloc[0]


# ============================================================
# 1. LOAD SOURCE DATA
# ============================================================

for path in [DATA_FILE, TRADES_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

candles = normalize_columns(pd.read_csv(DATA_FILE))
trades_all = normalize_columns(pd.read_csv(TRADES_FILE))

required_candle_columns = {"datetime", "open", "close", "high", "low"}

if not required_candle_columns.issubset(candles.columns):
    raise ValueError(
        "gold_data.csv is missing required columns.\n"
        f"Required: {sorted(required_candle_columns)}\n"
        f"Found: {list(candles.columns)}"
    )

candles["datetime"] = parse_time(candles["datetime"])

for col in ["open", "close", "high", "low"]:
    candles[col] = pd.to_numeric(candles[col], errors="coerce")

candles = (
    candles.dropna(
        subset=["datetime", "open", "close", "high", "low"]
    )
    .sort_values("datetime")
    .drop_duplicates(subset=["datetime"])
    .reset_index(drop=True)
)

if candles.empty:
    raise ValueError("No valid candles found in gold_data.csv.")

if "strategy" not in trades_all.columns:
    raise ValueError(
        "No 'strategy' column found in trade file.\n"
        f"Found: {list(trades_all.columns)}"
    )

trades = trades_all[
    trades_all["strategy"].astype(str).str.lower() == STRATEGY_NAME
].copy()

if trades.empty:
    raise ValueError(
        f"No {STRATEGY_NAME} trades found in {TRADES_FILE.name}."
    )

# Candidate column names to accommodate common file layouts.
entry_time_col = find_column(
    trades,
    ["entry_time", "entry_datetime", "entry_timestamp"]
)
exit_time_col = find_column(
    trades,
    ["exit_time", "exit_datetime", "exit_timestamp"]
)
signal_time_col = find_column(
    trades,
    ["signal_time", "signal_datetime", "signal_timestamp"]
)
direction_col = find_column(
    trades,
    ["direction", "side", "trade_direction", "signal"]
)
entry_price_col = find_column(
    trades,
    ["entry_price", "entry", "entry_fill", "entry_open"]
)
exit_price_col = find_column(
    trades,
    ["exit_price", "exit", "exit_fill", "exit_close"]
)
return_col = find_column(
    trades,
    ["strategy_return_pct", "return_pct", "trade_return_pct"]
)

# Normalize times and numeric values.
for col in [entry_time_col, exit_time_col, signal_time_col]:
    if col:
        trades[col] = parse_time(trades[col])

for col in [entry_price_col, exit_price_col, return_col]:
    if col:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

trades = trades.reset_index(drop=True)

# ============================================================
# 2. MAP SOURCE CANDLE TIMESTAMPS
# ============================================================

candle_index = {
    timestamp: idx
    for idx, timestamp in enumerate(candles["datetime"])
}

candle_times = candles["datetime"].tolist()

# ============================================================
# 3. AUDIT EACH TRADE
# ============================================================

audit_rows = []

for trade_number, trade in trades.iterrows():
    issues = []
    checks = {}

    entry_time = (
        trade[entry_time_col] if entry_time_col else pd.NaT
    )
    exit_time = (
        trade[exit_time_col] if exit_time_col else pd.NaT
    )
    signal_time = (
        trade[signal_time_col] if signal_time_col else pd.NaT
    )

    direction_raw = (
        str(trade[direction_col]).strip().lower()
        if direction_col else ""
    )

    entry_price = (
        trade[entry_price_col]
        if entry_price_col else np.nan
    )
    exit_price = (
        trade[exit_price_col]
        if exit_price_col else np.nan
    )
    stored_return = (
        trade[return_col]
        if return_col else np.nan
    )

    # ----- Required fields -----
    checks["entry_time_available"] = entry_time_col is not None
    checks["exit_time_available"] = exit_time_col is not None
    checks["signal_time_available"] = signal_time_col is not None
    checks["direction_available"] = direction_col is not None
    checks["entry_price_available"] = entry_price_col is not None
    checks["exit_price_available"] = exit_price_col is not None
    checks["return_available"] = return_col is not None

    # ----- Timestamp validity -----
    if entry_time_col:
        checks["entry_time_valid"] = pd.notna(entry_time)
        add_issue(
            issues, "ENTRY_TIME", pd.isna(entry_time),
            "Missing or invalid entry timestamp."
        )

    if exit_time_col:
        checks["exit_time_valid"] = pd.notna(exit_time)
        add_issue(
            issues, "EXIT_TIME", pd.isna(exit_time),
            "Missing or invalid exit timestamp."
        )

    if pd.notna(entry_time) and pd.notna(exit_time):
        checks["exit_after_entry"] = exit_time > entry_time
        add_issue(
            issues, "TIME_ORDER", exit_time <= entry_time,
            "Exit is not later than entry."
        )

        elapsed_hours = (
            exit_time - entry_time
        ).total_seconds() / 3600

        checks["exit_within_5_hours"] = (
            0 < elapsed_hours <= MAX_EXIT_HOURS
        )

        add_issue(
            issues, "EXIT_WINDOW",
            not (0 < elapsed_hours <= MAX_EXIT_HOURS),
            f"Elapsed time is {elapsed_hours:.3f} hours."
        )
    else:
        elapsed_hours = np.nan
        checks["exit_after_entry"] = None
        checks["exit_within_5_hours"] = None

    # ----- Holdout membership -----
    if pd.notna(entry_time):
        checks["entry_in_holdout"] = entry_time >= HOLDOUT_START
        add_issue(
            issues, "HOLDOUT",
            entry_time < HOLDOUT_START,
            f"Entry precedes holdout start {HOLDOUT_START}."
        )
    else:
        checks["entry_in_holdout"] = None

    # ----- Entry candle exists -----
    entry_idx = candle_index.get(entry_time, None) if pd.notna(entry_time) else None
    exit_idx = candle_index.get(exit_time, None) if pd.notna(exit_time) else None

    checks["entry_timestamp_is_candle"] = entry_idx is not None
    if pd.notna(entry_time):
        add_issue(
            issues, "ENTRY_CANDLE", entry_idx is None,
            "Entry timestamp does not exactly match a source candle."
        )

    checks["exit_timestamp_is_candle"] = exit_idx is not None
    if pd.notna(exit_time):
        add_issue(
            issues, "EXIT_CANDLE", exit_idx is None,
            "Exit timestamp does not exactly match a source candle."
        )

    # ----- Entry price check -----
    expected_entry_open = np.nan

    if entry_idx is not None:
        expected_entry_open = float(candles.iloc[entry_idx]["open"])

    if entry_price_col and entry_idx is not None:
        checks["entry_matches_open"] = close_enough(
            entry_price, expected_entry_open
        )
        add_issue(
            issues, "ENTRY_PRICE",
            not checks["entry_matches_open"],
            f"Stored entry={entry_price}; candle open={expected_entry_open}. "
            "Verify the original entry-price convention."
        )
    else:
        checks["entry_matches_open"] = None

    # ----- Exit price check -----
    expected_exit_close = np.nan

    if exit_idx is not None:
        expected_exit_close = float(candles.iloc[exit_idx]["close"])

    if exit_price_col and exit_idx is not None:
        checks["exit_matches_close"] = close_enough(
            exit_price, expected_exit_close
        )
        add_issue(
            issues, "EXIT_PRICE",
            not checks["exit_matches_close"],
            f"Stored exit={exit_price}; candle close={expected_exit_close}. "
            "Verify the original exit-price convention."
        )
    else:
        checks["exit_matches_close"] = None

    # ----- Reconstruct the target exit timestamp -----
    # The assumed rule is first source candle at/after entry + 4 hours,
    # provided that candle is no more than 5 hours after entry.
    expected_exit_time = pd.NaT

    if pd.notna(entry_time):
        target_time = entry_time + pd.Timedelta(hours=TARGET_HOURS)
        later = candles.loc[candles["datetime"] >= target_time, "datetime"]

        if not later.empty:
            candidate = later.iloc[0]
            candidate_elapsed = (
                candidate - entry_time
            ).total_seconds() / 3600

            if 0 < candidate_elapsed <= MAX_EXIT_HOURS:
                expected_exit_time = candidate

    if pd.notna(expected_exit_time) and pd.notna(exit_time):
        checks["exit_matches_rule"] = (
            expected_exit_time == exit_time
        )
        add_issue(
            issues, "EXIT_RULE",
            expected_exit_time != exit_time,
            f"Expected {expected_exit_time}; stored {exit_time}. "
            "Confirm the exit rule used by the original backtest."
        )
    else:
        checks["exit_matches_rule"] = None
        if pd.isna(expected_exit_time):
            issues.append(
                "EXIT_RULE: No source candle satisfies the assumed "
                "4-to-5-hour exit rule."
            )

    # ----- Recalculate return from prices and direction -----
    normalized_direction = direction_raw

    if normalized_direction in {"long", "buy", "bullish"}:
        direction_sign = 1
    elif normalized_direction in {"short", "sell", "bearish"}:
        direction_sign = -1
    else:
        direction_sign = None

    recalculated_return = np.nan

    if (
        direction_sign is not None
        and pd.notna(entry_price)
        and pd.notna(exit_price)
        and float(entry_price) != 0
    ):
        recalculated_return = (
            direction_sign
            * (float(exit_price) / float(entry_price) - 1)
            * 100
        )

    if return_col and pd.notna(recalculated_return) and pd.notna(stored_return):
        return_difference = recalculated_return - float(stored_return)

        checks["return_matches_prices"] = (
            abs(return_difference) <= RETURN_TOLERANCE_PCT
        )

        add_issue(
            issues, "RETURN",
            not checks["return_matches_prices"],
            f"Stored={float(stored_return):.6f}%; "
            f"recalculated={recalculated_return:.6f}%; "
            f"difference={return_difference:.6f} percentage points."
        )
    else:
        return_difference = np.nan
        checks["return_matches_prices"] = None
        if direction_sign is None:
            issues.append(
                "RETURN: Direction not recognized; expected long/buy "
                "or short/sell."
            )

    # ----- Signal-to-entry timing -----
    # This is verified only if the signal timestamp was saved.
    if pd.notna(signal_time) and pd.notna(entry_time):
        next_candles = candles.loc[
            candles["datetime"] > signal_time, "datetime"
        ]

        expected_entry_time = (
            next_candles.iloc[0] if not next_candles.empty else pd.NaT
        )

        if pd.notna(expected_entry_time):
            checks["entry_is_next_candle"] = (
                entry_time == expected_entry_time
            )
            add_issue(
                issues, "ENTRY_TIMING",
                entry_time != expected_entry_time,
                f"Expected next candle {expected_entry_time}; "
                f"stored entry {entry_time}."
            )
        else:
            checks["entry_is_next_candle"] = None
    else:
        checks["entry_is_next_candle"] = None

    # ----- Record result -----
    audit_rows.append({
        "trade_number": trade_number + 1,
        "strategy": trade.get("strategy", ""),
        "direction": direction_raw,
        "signal_time": signal_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "elapsed_hours": elapsed_hours,
        "entry_price": entry_price,
        "expected_entry_open": expected_entry_open,
        "exit_price": exit_price,
        "expected_exit_close": expected_exit_close,
        "stored_return_pct": stored_return,
        "recalculated_return_pct": recalculated_return,
        "return_difference_pct_points": return_difference,
        "expected_exit_time": expected_exit_time,
        "issue_count": len(issues),
        "issues": " | ".join(issues) if issues else "No detected issue",
        **checks,
    })

# ============================================================
# 4. CHECK TRADE OVERLAPS
# ============================================================

audit = pd.DataFrame(audit_rows).sort_values("entry_time").reset_index(drop=True)

audit["overlaps_previous_trade"] = False

for i in range(1, len(audit)):
    previous_exit = audit.loc[i - 1, "exit_time"]
    current_entry = audit.loc[i, "entry_time"]

    if pd.notna(previous_exit) and pd.notna(current_entry):
        if current_entry < previous_exit:
            audit.loc[i, "overlaps_previous_trade"] = True
            audit.loc[i, "issues"] += (
                " | OVERLAP: Entry occurs before previous trade exit."
            )
            audit.loc[i, "issue_count"] += 1

# ============================================================
# 5. SAVE AUDIT OUTPUTS
# ============================================================

audit.to_csv(AUDIT_FILE, index=False)

check_columns = [
    col for col in audit.columns
    if col.startswith("entry_")
    or col.startswith("exit_")
    or col.startswith("return_")
    or col in [
        "entry_in_holdout",
        "entry_timestamp_is_candle",
        "exit_timestamp_is_candle",
        "exit_after_entry",
        "exit_within_5_hours",
        "exit_matches_rule",
        "overlaps_previous_trade",
    ]
]

summary_lines = [
    "BREAKOUT_20 TRADE AUDIT SUMMARY",
    "=" * 60,
    f"Input trade file: {TRADES_FILE.name}",
    f"Source candle file: {DATA_FILE.name}",
    f"Trades audited: {len(audit)}",
    f"Holdout starts: {HOLDOUT_START}",
    f"Audit CSV: {AUDIT_FILE.name}",
    "",
    "CHECK RESULTS",
    "-" * 60,
]

for col in check_columns:
    if audit[col].dtype == bool or audit[col].dropna().isin([True, False]).all():
        passed = int((audit[col] == True).sum())
        failed = int((audit[col] == False).sum())
        unknown = int(audit[col].isna().sum())
        summary_lines.append(
            f"{col}: pass={passed}, fail={failed}, unknown={unknown}"
        )

summary_lines += [
    "",
    "ISSUES",
    "-" * 60,
    f"Trades with one or more detected issues: "
    f"{int((audit['issue_count'] > 0).sum())}",
    f"Total issue flags: {int(audit['issue_count'].sum())}",
    "",
    "IMPORTANT LIMITATIONS",
    "-" * 60,
    "1. This audit uses the assumed next-candle-open entry and "
    "4-to-5-hour exit conventions.",
    "2. A failed price check may indicate a different original "
    "execution convention, not necessarily a bug.",
    "3. Signal-to-entry timing cannot be confirmed if signal_time "
    "was not saved in the trade file.",
    "4. The script cannot prove the original signal used no future data.",
    "5. Costs, spread, slippage, position sizing, and live execution "
    "are not validated here.",
    "6. No detected issues does not prove strategy profitability.",
]

SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("          BREAKOUT_20 TRADE AUDIT")
print("=" * 60)
print(f"Trades audited: {len(audit)}")
print(f"Trades with issue flags: {(audit['issue_count'] > 0).sum()}")
print(f"Total issue flags: {audit['issue_count'].sum()}")

print("\n--- CHECK COUNTS ---")
for col in check_columns:
    if audit[col].dtype == bool or audit[col].dropna().isin([True, False]).all():
        passed = int((audit[col] == True).sum())
        failed = int((audit[col] == False).sum())
        unknown = int(audit[col].isna().sum())
        print(
            f"{col}: PASS={passed}, FAIL={failed}, UNKNOWN={unknown}"
        )

print("\n--- TRADES WITH DETECTED ISSUES (first 20) ---")
problem_trades = audit[audit["issue_count"] > 0]

if problem_trades.empty:
    print("No issues detected by the checks that could be performed.")
else:
    display_columns = [
        "trade_number",
        "direction",
        "entry_time",
        "exit_time",
        "stored_return_pct",
        "recalculated_return_pct",
        "issues",
    ]
    print(problem_trades[display_columns].head(20).to_string(index=False))

print("\n--- AUDIT FILES CREATED ---")
print(AUDIT_FILE.name)
print(SUMMARY_FILE.name)

print("\nRemember: UNKNOWN means the available CSV fields were insufficient.")
print("Read the issues and limitations before drawing conclusions.")
print("=" * 60)
