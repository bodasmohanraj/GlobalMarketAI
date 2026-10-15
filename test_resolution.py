
import pandas as pd
import paper_trading_tracker as tracker

# Synthetic test timestamps, separate from real market records.
prediction_time = pd.Timestamp("2099-01-01 10:00:00", tz="UTC")
target_time = prediction_time + pd.Timedelta(hours=4)

test_log = pd.DataFrame([{
    "model": "RESOLUTION_TEST_ONLY",
    "prediction_time": prediction_time,
    "target_time": target_time,
    "prediction_close": 100.0,
    "probability_up": 0.6,
    "probability_down": 0.4,
    "baseline_up_probability": 0.5,
    "signal": "NO_CLEAR_SIGNAL",
    "actual_close": float("nan"),
    "actual_up": float("nan"),
    "forward_return_pct": float("nan"),
    "resolved": False,
    "model_version": "TEST_ONLY",
    "training_cutoff_time": prediction_time - pd.Timedelta(hours=1),
    "training_rows": 100,
    "feature_set": "test_feature",
    "prediction_created_at": pd.Timestamp.now(tz="UTC").isoformat(),
}])

# Case 1: target timestamp missing — must remain pending.
raw_without_target = pd.DataFrame({
    "datetime": [prediction_time],
    "close": [100.0],
})

pending = tracker.resolve_predictions(
    test_log.copy(), raw_without_target
)

assert not bool(pending.loc[0, "resolved"])
assert pd.isna(pending.loc[0, "actual_close"])

print("PASS: prediction remains pending without target candle.")

# Case 2: exact target timestamp exists at a higher price.
raw_with_target = pd.DataFrame({
    "datetime": [prediction_time, target_time],
    "close": [100.0, 102.0],
})

resolved = tracker.resolve_predictions(
    test_log.copy(), raw_with_target
)

assert bool(resolved.loc[0, "resolved"])
assert resolved.loc[0, "actual_close"] == 102.0
assert resolved.loc[0, "actual_up"] == 1
assert abs(resolved.loc[0, "forward_return_pct"] - 2.0) < 1e-9

print("PASS: exact target candle resolves the prediction correctly.")
print("Actual close:", resolved.loc[0, "actual_close"])
print("Actual direction:", resolved.loc[0, "actual_up"])
print("Forward return (%):", resolved.loc[0, "forward_return_pct"])
print("\nPASS: isolated resolution tests completed.")
