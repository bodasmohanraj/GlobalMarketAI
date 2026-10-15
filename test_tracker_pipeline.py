from pathlib import Path
import hashlib
import pandas as pd
import paper_trading_tracker as tracker

base = Path(__file__).resolve().parent
real_log = base / "paper_trading_log.csv"
test_log = base / "paper_trading_log_pipeline_test.csv"
clean_file = base / "gold_features_clean.csv"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert real_log.exists(), "Real log is missing."
original_hash = digest(real_log)

print("Loading historical data...")
raw = tracker.read_raw()
features = tracker.build_features(raw)
clean = pd.read_csv(clean_file)
tracker.validate_features(features, clean)

print("Running isolated pipeline...")
log = pd.DataFrame(columns=tracker.LOG_COLUMNS)
log = tracker.resolve_predictions(log, raw)
log = tracker.generate_predictions(features, clean, log)
log = tracker.resolve_predictions(log, raw)

log = log.sort_values(
    ["prediction_time", "model"]
).reset_index(drop=True)

log.to_csv(test_log, index=False)

assert digest(real_log) == original_hash, "REAL LOG CHANGED!"

print("\n--- PIPELINE TEST RESULTS ---")
print("Test records:", len(log))
print("Models:", sorted(log["model"].dropna().unique().tolist()))

audit_fields = [
    "model_version",
    "training_cutoff_time",
    "training_rows",
    "feature_set",
    "prediction_created_at",
]

if len(log):
    for field in audit_fields:
        passed = log[field].notna().all()
        print(f"{field}: {'PASS' if passed else 'FAIL'}")
        assert passed, f"Audit field missing: {field}"

    resolved = log["resolved"].astype(str).str.lower().eq("true").sum()
    print("Resolved:", int(resolved))
    print("Pending:", len(log) - int(resolved))

print("\nPASS: original prediction log unchanged.")
print("Test results saved to:", test_log.name)
