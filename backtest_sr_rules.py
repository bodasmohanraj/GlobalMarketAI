from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "gold_features_clean.csv"
PREDICTIONS = BASE / "sr_rules_predictions.csv"
RESULTS = BASE / "sr_rules_results.csv"

CAPITAL = 100_000.0
COSTS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
NEAR_LEVEL = 0.002  # 0.20% proximity to a level

df = pd.read_csv(SOURCE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

required = [
    "datetime", "target_time", "open", "high", "low", "close",
    "future_close_4h"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if not (df["target_time"] == df["datetime"] + pd.Timedelta(hours=4)).all():
    raise ValueError("The target horizon is not exactly four hours.")

# Each level uses only candles BEFORE the current prediction candle.
for w in (20, 50):
    df[f"resistance_{w}"] = df["high"].rolling(w, min_periods=w).max().shift(1)
    df[f"support_{w}"] = df["low"].rolling(w, min_periods=w).min().shift(1)

    # Bounce/rejection signals, confirmed by the current candle close.
    near_support = (
        (df["low"] - df[f"support_{w}"]).abs() / df["close"] <= NEAR_LEVEL
    )
    near_resistance = (
        (df["high"] - df[f"resistance_{w}"]).abs() / df["close"] <= NEAR_LEVEL
    )

    df[f"bounce_{w}"] = np.select(
        [
            near_support & (df["close"] > df["open"]) &
            (df["close"] > df[f"support_{w}"]),
            near_resistance & (df["close"] < df["open"]) &
            (df["close"] < df[f"resistance_{w}"])
        ],
        [1, -1], default=0
    )

    # Breakouts confirmed at candle close, compared with the previous candle.
    prev_close = df["close"].shift(1)
    prev_resistance = df[f"resistance_{w}"].shift(1)
    prev_support = df[f"support_{w}"].shift(1)

    broke_up = (
        (df["close"] > df[f"resistance_{w}"]) &
        (prev_close <= prev_resistance)
    )
    broke_down = (
        (df["close"] < df[f"support_{w}"]) &
        (prev_close >= prev_support)
    )
    df[f"breakout_{w}"] = np.select(
        [broke_up, broke_down], [1, -1], default=0
    )

df["forward_return"] = df["future_close_4h"] / df["close"] - 1
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[
    "open", "high", "low", "close", "future_close_4h",
    "forward_return", "support_20", "resistance_20",
    "support_50", "resistance_50"
]).reset_index(drop=True)

df = df[df["target_time"] == df["datetime"] + pd.Timedelta(hours=4)].copy()

# A combined signal is used only if the two rule types agree or only one fires.
for w in (20, 50):
    b = df[f"bounce_{w}"]
    br = df[f"breakout_{w}"]
    df[f"combined_{w}"] = np.where(
        (b == 0), br,
        np.where((br == 0) | (b == br), b, 0)
    )

# Select non-overlapping entries separately for each strategy.
rules = ["bounce_20", "breakout_20", "combined_20",
         "bounce_50", "breakout_50", "combined_50"]
all_rows = []
summaries = []

for rule in rules:
    next_allowed = None
    for _, row in df.iterrows():
        side = int(row[rule])
        if side == 0:
            continue
        if next_allowed is not None and row["datetime"] < next_allowed:
            continue

        all_rows.append({
            "strategy": rule,
            "datetime": row["datetime"],
            "target_time": row["target_time"],
            "signal": side,
            "close": row["close"],
            "future_close_4h": row["future_close_4h"],
            "forward_return": row["forward_return"]
        })
        next_allowed = row["target_time"]

trades = pd.DataFrame(all_rows)
trades.to_csv(PREDICTIONS, index=False)

for rule in rules:
    t = trades[trades["strategy"] == rule].sort_values("datetime")
    sides = t["signal"].to_numpy(dtype=float)
    returns = t["forward_return"].to_numpy(dtype=float)

    # Report a flat strategy when no signals occur.
    for cost in COSTS:
        equity = CAPITAL
        peak = equity
        max_dd = 0.0
        total_cost = 0.0
        for side, ret in zip(sides, returns):
            start = equity
            charge = start * cost
            total_cost += charge
            equity = start * (1 + side * ret) - charge
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1)

        if cost == 0.001:
            equity_at_standard_cost = equity
            drawdown_at_standard_cost = max_dd * 100
            costs_at_standard_cost = total_cost

    summaries.append({
        "strategy": rule,
        "trades": len(t),
        "long_trades": int((sides == 1).sum()),
        "short_trades": int((sides == -1).sum()),
        "return_0_cost_pct": (
            ((np.prod(1 + sides * returns) - 1) * 100) if len(t) else 0.0
        ),
        "return_0.05pct_cost_pct": None,
        "return_0.10pct_cost_pct": (
            (equity_at_standard_cost / CAPITAL - 1) * 100
            if len(t) else 0.0
        ),
        "max_drawdown_0.10pct_cost_pct": (
            drawdown_at_standard_cost if len(t) else 0.0
        ),
        "costs_at_0.10pct": costs_at_standard_cost if len(t) else 0.0
    })

# Recompute each cost explicitly with identical trade entries.
result = pd.DataFrame(summaries)
for idx, rule in enumerate(rules):
    t = trades[trades["strategy"] == rule]
    for cost, col in [
        (0.0005, "return_0.05pct_cost_pct"),
        (0.0015, "return_0.15pct_cost_pct"),
        (0.002, "return_0.20pct_cost_pct")
    ]:
        equity = CAPITAL
        for side, ret in zip(t["signal"], t["forward_return"]):
            start = equity
            equity = start * (1 + side * ret) - start * cost
        result.loc[idx, col] = (equity / CAPITAL - 1) * 100

result.to_csv(RESULTS, index=False)

print("\n=== SUPPORT/RESISTANCE RULE TEST ===")
print("Prepared candles:", len(df))
print("Test start:", df["datetime"].min())
print("Test end target:", df["target_time"].max())
print("Near-level threshold:", f"{NEAR_LEVEL*100:.2f}%")
print("Initial simulated capital:", f"{CAPITAL:,.2f}")
print("\n=== RESULTS ===")
print(result.to_string(index=False, formatters={
    "return_0_cost_pct": "{:.2f}%".format,
    "return_0.05pct_cost_pct": lambda x: f"{x:.2f}%",
    "return_0.10pct_cost_pct": "{:.2f}%".format,
    "return_0.15pct_cost_pct": lambda x: f"{x:.2f}%",
    "return_0.20pct_cost_pct": lambda x: f"{x:.2f}%",
    "max_drawdown_0.10pct_cost_pct": "{:.2f}%".format,
    "costs_at_0.10pct": "{:,.2f}".format
}))
print("\nSaved:", PREDICTIONS.name, "and", RESULTS.name)
print("Research only. Candle-close signals assume execution after the signal candle closes.")
