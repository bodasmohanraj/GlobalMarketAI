
import pandas as pd
import numpy as np

FILE = "gold_features_clean.csv"

df = pd.read_csv(FILE)

features = [
    "return_1h",
    "return_4h",
    "return_12h",
    "sma_5_gap",
    "sma_20_gap",
    "trend_5_20",
    "volatility_10",
    "volatility_20",
    "rsi_14",
]

target = "target_up"

df = df.dropna(subset=features + [target]).copy()
df[target] = df[target].astype(int)

print("\n--- FEATURE SIGNAL CHECK ---")
print("Rows:", len(df))
print("Overall upward rate:", round(df[target].mean(), 4))

results = []

for feature in features:
    valid = df[[feature, target]].dropna()

    correlation = valid[feature].corr(valid[target])

    low = valid[feature] <= valid[feature].median()
    high = ~low

    low_up_rate = valid.loc[low, target].mean()
    high_up_rate = valid.loc[high, target].mean()

    results.append({
        "feature": feature,
        "correlation_with_target": correlation,
        "up_rate_low_half": low_up_rate,
        "up_rate_high_half": high_up_rate,
        "difference": high_up_rate - low_up_rate,
    })

result_df = pd.DataFrame(results)
result_df["abs_correlation"] = (
    result_df["correlation_with_target"].abs()
)

result_df = result_df.sort_values(
    "abs_correlation", ascending=False
)

print("\nFeature results:")
print(
    result_df.drop(columns=["abs_correlation"])
    .round(4)
    .to_string(index=False)
)
