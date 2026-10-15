
import pandas as pd
from pathlib import Path

# Isolated test file — never modifies the real prediction log.
TEST_FILE = Path("paper_trading_log_audit_test.csv")

audit_fields = [
    "model_version",
    "training_cutoff_time",
    "training_rows",
    "feature_set",
    "prediction_created_at",
]

test_record = {
    "model": "AUDIT_TEST_ONLY",
    "prediction_time": "2099-01-01 00:00:00+00:00",
    "target_time": "2099-01-01 04:00:00+00:00",
    "prediction_close": 4000.0,
    "probability_up": 0.52,
    "probability_down": 0.48,
    "baseline_up_probability": 0.50,
    "signal": "NO_CLEAR_SIGNAL",
    "actual_close": None,
    "actual_up": None,
    "forward_return_pct": None,
    "resolved": False,
    "model_version": "TEST_VERSION_001",
    "training_cutoff_time": "2098-12-31 20:00:00+00:00",
    "training_rows": 100,
    "feature_set": "test_feature_a,test_feature_b",
    "prediction_created_at": pd.Timestamp.now(tz="UTC").isoformat(),
}

pd.DataFrame([test_record]).to_csv(TEST_FILE, index=False)

saved = pd.read_csv(TEST_FILE)
print("Test file:", TEST_FILE.resolve())
print("Audit field completeness:")
print(saved[audit_fields].notna().sum().to_string())

assert saved[audit_fields].notna().all().all()
assert saved.loc[0, "model"] == "AUDIT_TEST_ONLY"

print("\nPASS: all five audit fields were saved.")
print("This tests CSV logging only, not the live tracker.")
