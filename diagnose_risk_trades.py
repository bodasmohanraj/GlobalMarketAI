
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "risk_managed_v2_trades.csv"

DETAILS_OUT = BASE / "trade_diagnostics_details.csv"
SUMMARY_OUT = BASE / "trade_diagnostics_summary.csv"
TEXT_OUT = BASE / "trade_diagnostics_report.txt"

COST_SCENARIOS = [0.0005, 0.001, 0.0015]


def load_trades():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE.name}. Run risk_managed_backtest_v2.py first."
        )

    df = pd.read_csv(SOURCE)
    df.columns = df.columns.str.strip().str.lower()

    required = [
        "fold", "direction", "entry_time", "exit_time",
        "entry_price", "exit_price", "exit_reason", "net_pnl",
        "gross_pnl", "total_cost", "atr_at_signal",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["entry_time", "exit_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in [
        "entry_price", "exit_price", "net_pnl", "gross_pnl",
        "total_cost", "atr_at_signal", "quantity_abstract_units",
        "entry_notional", "stop_price", "target_price",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "entry_time", "exit_time", "entry_price",
            "exit_price", "net_pnl", "gross_pnl",
        ]
    ).copy()

    df["direction"] = df["direction"].str.upper()
    df["exit_reason"] = df["exit_reason"].str.upper()
    df["holding_hours"] = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 3600

    df["outcome"] = np.select(
        [df["net_pnl"] > 0, df["net_pnl"] < 0],
        ["WIN", "LOSS"],
        default="BREAKEVEN",
    )

    df["net_R"] = np.nan
    if "planned_risk_amount" in df.columns:
        risk = pd.to_numeric(
            df["planned_risk_amount"], errors="coerce"
        )
        df["net_R"] = np.where(risk > 0, df["net_pnl"] / risk, np.nan)

    return df.sort_values(["fold", "entry_time"]).reset_index(drop=True)


def profit_factor(pnls):
    pnls = np.asarray(pnls, dtype=float)
    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan

    return gains / losses


def summarize_group(group, label_columns):
    pnl = group["net_pnl"].to_numpy(dtype=float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    row = {
        **label_columns,
        "trades": len(group),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else 0,
        "net_pnl": float(pnl.sum()),
        "mean_net_pnl": float(pnl.mean()) if len(pnl) else 0,
        "median_net_pnl": float(np.median(pnl)) if len(pnl) else 0,
        "gross_profit": float(winners.sum()),
        "gross_loss": float(-losers.sum()),
        "profit_factor": float(profit_factor(pnl)),
        "mean_holding_hours": float(group["holding_hours"].mean()),
    }

    if "net_R" in group:
        row["mean_net_R"] = float(group["net_R"].mean())

    return row


def main():
    df = load_trades()

    summaries = []

    # 1. Overall
    summaries.append(summarize_group(df, {"group_type": "overall", "group": "all"}))

    # 2. Long vs short
    for direction, group in df.groupby("direction"):
        summaries.append(
            summarize_group(
                group, {"group_type": "direction", "group": direction}
            )
        )

    # 3. Exit reason
    for reason, group in df.groupby("exit_reason"):
        summaries.append(
            summarize_group(
                group, {"group_type": "exit_reason", "group": reason}
            )
        )

    # 4. Fold-by-fold performance
    for fold, group in df.groupby("fold"):
        summaries.append(
            summarize_group(
                group, {"group_type": "fold", "group": str(fold)}
            )
        )

    # 5. Long/short within each fold
    for (fold, direction), group in df.groupby(["fold", "direction"]):
        summaries.append(
            summarize_group(
                group,
                {
                    "group_type": "fold_direction",
                    "group": f"fold_{fold}_{direction}",
                },
            )
        )

    # 6. Volatility buckets based on ATR/entry price.
    df["atr_pct"] = df["atr_at_signal"] / df["entry_price"] * 100
    try:
        df["volatility_bucket"] = pd.qcut(
            df["atr_pct"],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        df["volatility_bucket"] = "unavailable"

    for bucket, group in df.groupby("volatility_bucket", observed=True):
        summaries.append(
            summarize_group(
                group,
                {"group_type": "volatility", "group": str(bucket)},
            )
        )

    # 7. Cost sensitivity based on stored gross P&L and entry notional.
    if "entry_notional" in df.columns:
        for cost in COST_SCENARIOS:
            estimated_net = (
                df["gross_pnl"] - df["entry_notional"] * cost
            )
            summaries.append({
                "group_type": "cost_sensitivity",
                "group": f"{cost * 100:.2f}%_of_entry_notional",
                "trades": len(df),
                "wins": int((estimated_net > 0).sum()),
                "losses": int((estimated_net < 0).sum()),
                "win_rate_pct": float((estimated_net > 0).mean() * 100),
                "net_pnl": float(estimated_net.sum()),
                "mean_net_pnl": float(estimated_net.mean()),
                "median_net_pnl": float(estimated_net.median()),
                "gross_profit": float(estimated_net[estimated_net > 0].sum()),
                "gross_loss": float(-estimated_net[estimated_net < 0].sum()),
                "profit_factor": float(profit_factor(estimated_net)),
                "mean_holding_hours": float(df["holding_hours"].mean()),
                "mean_net_R": np.nan,
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(SUMMARY_OUT, index=False)

    # Detailed rows: worst and best trades are flagged, not removed.
    df["pnl_rank_ascending"] = df["net_pnl"].rank(method="first")
    df["pnl_rank_descending"] = df["net_pnl"].rank(
        ascending=False, method="first"
    )
    df["worst_20_flag"] = df["pnl_rank_ascending"] <= min(20, len(df))
    df["best_20_flag"] = df["pnl_rank_descending"] <= min(20, len(df))
    df.to_csv(DETAILS_OUT, index=False)

    overall = summary[summary["group_type"] == "overall"].iloc[0]
    worst = df.nsmallest(min(10, len(df)), "net_pnl")
    best = df.nlargest(min(10, len(df)), "net_pnl")

    lines = [
        "TRADE DIAGNOSTICS REPORT",
        "=" * 45,
        f"Trades analysed: {len(df)}",
        f"Total net P&L: {df['net_pnl'].sum():.2f}",
        f"Win rate: {(df['net_pnl'] > 0).mean() * 100:.2f}%",
        f"Profit factor: {profit_factor(df['net_pnl']):.3f}",
        "",
        "DIRECTION SUMMARY",
        summary[summary["group_type"] == "direction"].to_string(index=False),
        "",
        "EXIT REASON SUMMARY",
        summary[summary["group_type"] == "exit_reason"].to_string(index=False),
        "",
        "FOLD SUMMARY",
        summary[summary["group_type"] == "fold"].to_string(index=False),
        "",
        "VOLATILITY SUMMARY",
        summary[summary["group_type"] == "volatility"].to_string(index=False),
        "",
        "COST SENSITIVITY",
        summary[summary["group_type"] == "cost_sensitivity"].to_string(index=False),
        "",
        "10 WORST TRADES",
        worst[
            ["fold", "direction", "entry_time", "exit_time",
             "exit_reason", "gross_pnl", "total_cost", "net_pnl"]
        ].to_string(index=False),
        "",
        "10 BEST TRADES",
        best[
            ["fold", "direction", "entry_time", "exit_time",
             "exit_reason", "gross_pnl", "total_cost", "net_pnl"]
        ].to_string(index=False),
        "",
        "Interpretation note: these are descriptive summaries of the same historical test trades.",
        "They are exploratory and do not establish predictive power or future profitability.",
    ]

    TEXT_OUT.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved: {DETAILS_OUT.name}")
    print(f"Saved: {SUMMARY_OUT.name}")
    print(f"Saved: {TEXT_OUT.name}")


if __name__ == "__main__":
    main()
