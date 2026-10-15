
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "gold_data.csv"
TRADES_FILE = BASE_DIR / "sr_holdout_trades.csv"

OUTPUT_FILE = BASE_DIR / "breakout_regime_results.csv"
DETAIL_FILE = BASE_DIR / "breakout_trades_with_regimes.csv"

ATR_PERIOD = 14
EMA_PERIOD = 50
SLOPE_LOOKBACK = 5
TREND_THRESHOLD = 0.001

# ============================================================
# 2. LOAD AND CLEAN GOLD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Could not find: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

# Normalize all column names to lowercase
df.columns = df.columns.str.strip().str.lower()

required = {"datetime", "close", "high", "low"}

if not required.issubset(df.columns):
    raise ValueError(
        f"gold_data.csv must contain {sorted(required)}. "
        f"Found: {list(df.columns)}"
    )

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

for col in ["close", "high", "low"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["datetime", "close", "high", "low"])
    .sort_values("datetime")
    .drop_duplicates(subset=["datetime"])
    .reset_index(drop=True)
)

df = df.rename(
    columns={
        "datetime": "Datetime",
        "close": "Close",
        "high": "High",
        "low": "Low",
    }
)

if df.empty:
    raise ValueError("No valid gold candle data remains after cleaning.")

# ============================================================
# 3. CALCULATE TREND AND VOLATILITY INDICATORS
# ============================================================

previous_close = df["Close"].shift(1)

true_range = pd.concat(
    [
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs(),
    ],
    axis=1,
).max(axis=1)

# Simple rolling ATR
df["ATR"] = true_range.rolling(ATR_PERIOD).mean()

# Exponential moving average
df["EMA50"] = df["Close"].ewm(
    span=EMA_PERIOD,
    adjust=False,
    min_periods=EMA_PERIOD,
).mean()

# EMA slope over five candles
df["EMA_slope"] = (
    df["EMA50"] / df["EMA50"].shift(SLOPE_LOOKBACK) - 1
)

# ATR as a percentage of gold's closing price
df["ATR_pct"] = df["ATR"] / df["Close"]

# ============================================================
# 4. CLASSIFY TREND REGIMES
# ============================================================

df["regime"] = "sideways_or_unclear"

uptrend = (
    (df["Close"] > df["EMA50"])
    & (df["EMA_slope"] > TREND_THRESHOLD)
)

downtrend = (
    (df["Close"] < df["EMA50"])
    & (df["EMA_slope"] < -TREND_THRESHOLD)
)

df.loc[uptrend, "regime"] = "uptrend"
df.loc[downtrend, "regime"] = "downtrend"

# ============================================================
# 5. CLASSIFY VOLATILITY
# ============================================================

# Calculate a rolling median from past observations only.
# The shift prevents the current candle from setting its own threshold.
df["volatility_threshold"] = (
    df["ATR_pct"]
    .rolling(500, min_periods=100)
    .quantile(0.50)
    .shift(1)
)

df["volatility"] = "unknown"

known_volatility = (
    df["ATR_pct"].notna()
    & df["volatility_threshold"].notna()
)

high_volatility = (
    known_volatility
    & (df["ATR_pct"] > df["volatility_threshold"])
)

low_volatility = (
    known_volatility
    & (df["ATR_pct"] <= df["volatility_threshold"])
)

df.loc[high_volatility, "volatility"] = "high_volatility"
df.loc[low_volatility, "volatility"] = "low_volatility"

# ============================================================
# 6. LOAD THE EXISTING BACKTEST TRADES
# ============================================================

if not TRADES_FILE.exists():
    raise FileNotFoundError(f"Could not find: {TRADES_FILE}")

trades = pd.read_csv(TRADES_FILE)

trades.columns = trades.columns.str.strip().str.lower()

required_trade_columns = {
    "strategy",
    "entry_time",
    "exit_time",
    "strategy_return_pct",
}

if not required_trade_columns.issubset(trades.columns):
    raise ValueError(
        "Trade file is missing required columns. "
        f"Required: {sorted(required_trade_columns)}. "
        f"Found: {list(trades.columns)}"
    )

trades["entry_time"] = pd.to_datetime(
    trades["entry_time"], errors="coerce", utc=True
)

trades["exit_time"] = pd.to_datetime(
    trades["exit_time"], errors="coerce", utc=True
)

trades["strategy_return_pct"] = pd.to_numeric(
    trades["strategy_return_pct"], errors="coerce"
)

# Keep only breakout_20 trades
trades = trades[
    trades["strategy"].astype(str).str.lower() == "breakout_20"
].copy()

trades = trades.dropna(
    subset=["entry_time", "exit_time", "strategy_return_pct"]
)

if trades.empty:
    raise ValueError(
        "No valid breakout_20 trades found in sr_holdout_trades.csv."
    )

# ============================================================
# 7. MATCH EACH TRADE TO ITS ENTRY-TIME REGIME
# ============================================================

regime_data = df[
    [
        "Datetime",
        "regime",
        "volatility",
        "ATR_pct",
        "EMA_slope",
    ]
].sort_values("Datetime")

trades = trades.sort_values("entry_time")

# Use the latest available candle at or before entry time.
# No future regime candle is used to label an entry.
trades = pd.merge_asof(
    trades,
    regime_data,
    left_on="entry_time",
    right_on="Datetime",
    direction="backward",
)

trades = trades.drop(columns=["Datetime"])

# Remove trades for which no regime candle was available
trades = trades.dropna(subset=["regime"])

if trades.empty:
    raise ValueError(
        "No trades could be matched to a gold market regime."
    )

# ============================================================
# 8. SUMMARIZE RESULTS BY TREND AND VOLATILITY
# ============================================================

def win_rate_percent(returns):
    return (returns > 0).mean() * 100


def compounded_return_percent(returns):
    # Assumes each percentage return applies to the same
    # notional capital fraction. This is a descriptive metric,
    # not a portfolio backtest.
    return ((1 + returns / 100).prod() - 1) * 100


summary = (
    trades.groupby(["regime", "volatility"], dropna=False)
    .agg(
        trades=("strategy_return_pct", "size"),
        win_rate_pct=("strategy_return_pct", win_rate_percent),
        mean_return_pct=("strategy_return_pct", "mean"),
        median_return_pct=("strategy_return_pct", "median"),
        total_compounded_return_pct=(
            "strategy_return_pct",
            compounded_return_percent,
        ),
    )
    .reset_index()
    .sort_values(["regime", "volatility"])
)

# ============================================================
# 9. SUMMARIZE RESULTS BY TREND ONLY
# ============================================================

trend_summary = (
    trades.groupby("regime")
    .agg(
        trades=("strategy_return_pct", "size"),
        win_rate_pct=("strategy_return_pct", win_rate_percent),
        mean_return_pct=("strategy_return_pct", "mean"),
        median_return_pct=("strategy_return_pct", "median"),
    )
    .reset_index()
    .sort_values("regime")
)

# ============================================================
# 10. SAVE RESULTS
# ============================================================

trades.to_csv(DETAIL_FILE, index=False)
summary.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("       BREAKOUT_20 MARKET REGIME ANALYSIS")
print("=" * 60)

print(f"Gold candles: {len(df)}")
print(f"Matched breakout trades: {len(trades)}")
print(f"Gold data starts: {df['Datetime'].min()}")
print(f"Gold data ends:   {df['Datetime'].max()}")
print(f"Trade entries:    {trades['entry_time'].min()}")
print(f"Trade exits:      {trades['exit_time'].max()}")

print("\n--- RESULTS BY TREND AND VOLATILITY ---")
print(summary.round(4).to_string(index=False))

print("\n--- RESULTS BY TREND ONLY ---")
print(trend_summary.round(4).to_string(index=False))

print("\n--- REGIME COUNTS ---")
print(df["regime"].value_counts().to_string())

print("\n--- VOLATILITY COUNTS ---")
print(df["volatility"].value_counts().to_string())

print("\nFiles created:")
print(f"1. {OUTPUT_FILE.name}")
print(f"2. {DETAIL_FILE.name}")

print("\nIMPORTANT:")
print("- Returns are historical and before trading costs.")
print("- Regime definitions are simple research proxies.")
print("- Small groups may produce unreliable averages.")
print("- These results do not establish future profitability.")
print("- This analysis reuses the existing holdout trades.")
print("=" * 60)
