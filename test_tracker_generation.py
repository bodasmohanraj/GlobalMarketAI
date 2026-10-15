
from pathlib import Path
import pandas as pd
import paper_trading_tracker as tracker

# All test outputs stay separate from the real prediction log.
TEST_LOG = Path("paper_trading_log_generation_test.csv")
REAL_LOG = Path("paper_trading_log.csv")

real_log_before = REAL_LOG.read_bytes()

# Load real historical data and build features in memory.
raw = tracker.read_raw()
features = tracker.build_features(raw)

clean = pd.read_csv("gold_features_clean.csv")
clean["datetime"] = pd.to_datetime(clean["datetime"], utc=True)
clean["target_time"] = pd.to_datetime(clean["target_time"], utc=True)

# Use a timestamp later than the current final candle so training
# labels can be selected strictly before the test prediction time.
latest = features.iloc[-1].copy()
test_time = latest["datetime"] + pd.Timedelta(hours=1)

# Copy features and set the test timestamp forward by one hour.
test_features = features.copy()
test_features.loc[test_features.index[-1], "datetime"] = test_time

# No existing records in the isolated test log.
empty_log = pd.DataFrame(columns=tracker.LOG_COLUMNS)

result = tracker.generate_predictions(
    test_features,
    clean,
    empty_log,
)

result.to_csv(TEST_LOG, index=False)
saved = pd.read_csv(TEST_LOG)

audit_fields = [
    "model_version",
    "training_cutoff_time",
    "training_rows",
    "feature_set",
    "prediction_created_at",
]

print("Generated rows:", len(saved))
print("Models:", saved["model"].tolist())
print("\nAudit field completeness:")
print(saved[audit_fields].notna().sum().to_string())

assert len(saved) == len(tracker.FEATURE_SETS), (
    "Unexpected number of model records."
)
assert saved[audit_fields].notna().all().all(), (
    "Some audit fields are missing."
)
assert REAL_LOG.read_bytes() == real_log_before, (
    "The real prediction log changed!"
)

print("\nPASS: actual generation function populated all audit fields.")
print("PASS: real prediction log remained unchanged.")
print("Test output:", TEST_LOG.resolve())

# Test duplicate protection using the generated test records.
before_count = len(result)

result_again = tracker.generate_predictions(
    test_features,
    clean,
    result,
)

after_count = len(result_again)

print("\nDuplicate protection test:")
print("Records before second call:", before_count)
print("Records after second call:", after_count)

assert before_count == 3
assert after_count == before_count

print("PASS: second call added no duplicate predictions.")
