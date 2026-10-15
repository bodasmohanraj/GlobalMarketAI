
import pandas as pd

source = pd.read_csv("gold_features_timebased.csv")
ready = pd.read_csv("gold_model_ready.csv")

features = [
    "return_1h", "return_4h", "return_12h",
    "sma_5_gap", "sma_20_gap", "trend_5_20",
    "volatility_10", "volatility_20", "rsi_14",
]

# Match saved model-ready rows back to source timestamps.
ready_times = set(pd.to_datetime(ready["datetime"], utc=True))
source["datetime"] = pd.to_datetime(source["datetime"], utc=True)

valid_target = source["actual_horizon_hours"].between(4, 4.5)
valid_features = source[features].notna().all(axis=1)

print("\n--- ROW EXCLUSION CHECK ---")
print("Source rows:", len(source))
print("Valid targets:", int(valid_target.sum()))
print("Valid features:", int(valid_features.sum()))
print("Valid target AND features:",
      int((valid_target & valid_features).sum()))
print("Rows in model-ready file:", len(ready))
print("Source timestamps represented in ready file:",
      source["datetime"].isin(ready_times).sum())

print("\nFeature completeness among valid targets:")
print(source.loc[valid_target, features].notna().sum())

print("\nExamples of valid-target rows with missing features:")
print(
    source.loc[valid_target & ~valid_features,
               ["datetime", "actual_horizon_hours"] + features]
    .head(10)
    .to_string(index=False)
)
