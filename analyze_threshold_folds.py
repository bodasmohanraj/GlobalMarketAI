from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FILE = BASE / "threshold_oos_predictions.csv"

if not FILE.exists():
    raise FileNotFoundError(f"Missing {FILE}")

d = pd.read_csv(FILE)
for c in ["datetime", "target_time"]:
    d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")

thresholds = [0.52, 0.55, 0.60, 0.65]
costs = [0.0005, 0.001, 0.0015]
rows = []

for (model, fold), g in d.groupby(["model", "fold"], sort=True):
    g = g.sort_values("datetime").copy()
    for threshold in thresholds:
        p = pd.to_numeric(g["probability_up"], errors="coerce")
        signals = np.select(
            [p >= threshold, p <= 1 - threshold],
            [1, -1], default=0
        )
        active = signals != 0
        actual = pd.to_numeric(g["target_up"], errors="coerce").to_numpy()
        valid = active & np.isfinite(actual)
        accuracy = (
            np.mean((signals[valid] == 1) == (actual[valid] == 1)) * 100
            if valid.any() else np.nan
        )

        for cost in costs:
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            next_entry = None
            trade_returns = []

            for i, (_, r) in enumerate(g.iterrows()):
                signal = int(signals[i])
                if signal == 0:
                    continue

                entry = r["datetime"]
                exit_time = r["target_time"]
                if pd.isna(exit_time):
                    continue
                if next_entry is not None and entry < next_entry:
                    continue

                move = float(r["forward_return"])
                if not np.isfinite(move):
                    continue

                gross = move if signal == 1 else -move
                net = (1 + gross) * (1 - cost) - 1
                equity *= 1 + net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1)
                trade_returns.append(net)
                next_entry = exit_time

            t = np.array(trade_returns)
            wins = t[t > 0].sum()
            losses = -t[t < 0].sum()

            rows.append({
                "model": model,
                "fold": int(fold),
                "threshold": threshold,
                "cost_pct": cost * 100,
                "oos_rows": len(g),
                "long_signals": int((signals == 1).sum()),
                "short_signals": int((signals == -1).sum()),
                "no_signals": int((signals == 0).sum()),
                "directional_accuracy_pct": accuracy,
                "trades": len(t),
                "net_return_pct": (equity - 1) * 100,
                "profit_factor": wins / losses if losses else np.nan,
                "max_drawdown_pct": max_dd * 100,
            })

out = pd.DataFrame(rows)
out.to_csv(BASE / "threshold_fold_results.csv", index=False)

summary = (
    out.groupby(["model", "threshold", "cost_pct"])
    .agg(
        folds=("fold", "nunique"),
        positive_folds=("net_return_pct", lambda x: int((x > 0).sum())),
        median_fold_return_pct=("net_return_pct", "median"),
        mean_fold_return_pct=("net_return_pct", "mean"),
        total_trades=("trades", "sum"),
    )
    .reset_index()
)
summary.to_csv(BASE / "threshold_fold_summary.csv", index=False)

print("\nFOLD-BY-FOLD RESULTS")
print(out.round(3).to_string(index=False))
print("\nFOLD STABILITY SUMMARY")
print(summary.round(3).to_string(index=False))
print("\nSaved threshold_fold_results.csv")
print("Saved threshold_fold_summary.csv")
