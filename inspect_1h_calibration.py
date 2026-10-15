import pandas as pd
import numpy as np

df = pd.read_csv("validation_audit_oos_predictions_v2.csv")
df = df[df["horizon"].eq("1hour")].copy()

df["p_up_model"] = pd.to_numeric(df["p_up_model"], errors="coerce")
df["actual_up"] = pd.to_numeric(df["actual_up"], errors="coerce")
df = df.dropna(subset=["p_up_model", "actual_up"])

df["probability_band"] = pd.cut(
    df["p_up_model"],
    bins=[0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0],
    include_lowest=True
)

print("1-HOUR PROBABILITY DISTRIBUTION")
print(
    df.groupby(["fold", "probability_band"], observed=False)
      .agg(
          rows=("actual_up", "size"),
          mean_probability=("p_up_model", "mean"),
          actual_up_rate=("actual_up", "mean"),
      )
      .round(4)
      .to_string()
)

print("\nSaved no files; this is a diagnostic only.")
