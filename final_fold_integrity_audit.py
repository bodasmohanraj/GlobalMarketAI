from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
fold_file = root / "validation_audit_fold_metrics_v2.csv"
pred_file = root / "validation_audit_oos_predictions_v2.csv"

if not fold_file.exists() or not pred_file.exists():
    raise SystemExit("Missing validation report files.")

folds = pd.read_csv(fold_file)
preds = pd.read_csv(pred_file)

print("=" * 65)
print("FINAL FOLD-INTEGRITY AUDIT — RESEARCH ONLY")
print("=" * 65)

# Check required fields.
required = {
    "horizon", "fold", "task", "method", "rows",
    "train_rows", "train_end", "test_start", "test_end"
}
missing = required - set(folds.columns)
if missing:
    print("Missing fold columns:", sorted(missing))
    print("Available columns:", list(folds.columns))
    raise SystemExit("Cannot verify fold boundaries with this report.")

for col in ["train_end", "test_start", "test_end"]:
    folds[col] = pd.to_datetime(folds[col], utc=True, errors="coerce")

class_rows = folds[folds["task"] == "classification"].copy()
checks = []

for (horizon, fold), group in class_rows.groupby(["horizon", "fold"]):
    model_rows = group[group["method"] == "model_logistic"]
    if model_rows.empty:
        checks.append({
            "horizon": horizon, "fold": fold,
            "status": "FAIL", "reason": "Missing model row"
        })
        continue

    r = model_rows.iloc[0]
    start_ok = pd.notna(r["train_end"]) and pd.notna(r["test_start"])
    end_ok = pd.notna(r["test_end"]) and start_ok
    boundary_ok = start_ok and r["train_end"] < r["test_start"]
    range_ok = end_ok and r["test_start"] <= r["test_end"]

    checks.append({
        "horizon": horizon,
        "fold": fold,
        "train_rows": r["train_rows"],
        "test_rows": r["rows"],
        "train_end": r["train_end"],
        "test_start": r["test_start"],
        "test_end": r["test_end"],
        "boundary_ok": boundary_ok,
        "range_ok": range_ok,
        "status": "PASS" if boundary_ok and range_ok else "FAIL",
        "reason": "Chronological boundary check"
                  if boundary_ok and range_ok
                  else "Invalid or missing time boundary"
    })

report = pd.DataFrame(checks)
report.to_csv(root / "final_fold_integrity_report.csv", index=False)
print("\nFOLD BOUNDARY CHECKS")
print(report.to_string(index=False))

# Verify that methods in each classification fold report the same test size.
size_checks = []
for (horizon, fold), group in class_rows.groupby(["horizon", "fold"]):
    sizes = group["rows"].dropna().astype(int).unique()
    size_checks.append({
        "horizon": horizon,
        "fold": fold,
        "test_sizes": ",".join(map(str, sizes)),
        "same_test_size": len(sizes) == 1
    })

sizes = pd.DataFrame(size_checks)
print("\nTEST-SIZE CONSISTENCY")
print(sizes.to_string(index=False))

# Inspect OOS prediction coverage and duplicate keys.
print("\nOUT-OF-SAMPLE PREDICTIONS")
print("Rows:", len(preds))
keys = [c for c in ["horizon", "datetime"] if c in preds.columns]
if len(keys) == 2:
    print("Duplicate horizon/timestamp keys:", int(preds.duplicated(keys).sum()))
else:
    print("Cannot check duplicate keys; expected horizon and datetime columns.")

print("\nSaved:", root / "final_fold_integrity_report.csv")
print("NOTE: this verifies recorded fold boundaries, not target-time purging.")
