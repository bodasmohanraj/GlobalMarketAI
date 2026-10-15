
from pathlib import Path
import pandas as pd
import paper_trading_tracker as tracker

# Verify the real prediction log is not modified by this test.
real_log_path = Path("paper_trading_log.csv")
before = real_log_path.read_bytes()

# Inspect the real function's implementation without executing it.
import inspect
source = inspect.getsource(tracker.generate_predictions)

required_fields = [
    '"model_version"',
    '"training_cutoff_time"',
    '"training_rows"',
    '"feature_set"',
    '"prediction_created_at"',
]

print("Actual function:", tracker.generate_predictions.__name__)

for field in required_fields:
    found = field in source
    print(f"{field}: {'FOUND' if found else 'MISSING'}")
    assert found, f"Missing field in generation function: {field}"

assert real_log_path.read_bytes() == before

print("\nPASS: actual generation function contains all five audit fields.")
print("Real prediction log remained unchanged.")
print("Note: this is a source inspection, not a generated prediction test.")
