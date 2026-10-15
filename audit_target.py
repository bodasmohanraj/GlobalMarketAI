
import pandas as pd
import numpy as np

DATA_FILE = "gold_data.csv"
MODEL_FILE = "gold_model_ready.csv"

data = pd.read_csv(DATA_FILE)
model = pd.read_csv(MODEL_FILE)

data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
model["datetime"] = pd.to_datetime(model["datetime"], utc=True)

data["close"] = pd.to_numeric(data["close"], errors="coerce")
model["target_up"] = pd.to_numeric(model["target_up"], errors="coerce")

data = (
    data.dropna(subset=["datetime", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
)

# Map each model timestamp to the first observation at or after
# exactly four elapsed hours later.
future_times = data["datetime"] + pd.Timedelta(hours=4)
future_positions = data["datetime"].searchsorted(
    future_times, side="left"
)

valid = future_positions < len(data)
rows = np.flatnonzero(valid)
positions = future_positions[valid]

elapsed = (
    data["datetime"].iloc[positions].reset_index(drop=True)
    - future_times.iloc[rows].reset_index(drop=True)
).dt.total_seconds() / 3600

# Accept only observations exactly four elapsed hours later.
exact = elapsed.abs() < 1e-9
rows = rows[exact.to_numpy()]
positions = positions[exact.to_numpy()]

future_close = data["close"].iloc[positions].to_numpy()
current_close = data["close"].iloc[rows].to_numpy()
expected_target = (future_close > current_close).astype(int)

expected = pd.DataFrame({
    "datetime": data["datetime"].iloc[rows].to_numpy(),
    "expected_target": expected_target,
    "expected_future_close": future_close,
})

checked = model.merge(expected, on="datetime", how="left")

comparable = checked["expected_target"].notna()
mismatches = (
    checked.loc[comparable, "target_up"].to_numpy()
    != checked.loc[comparable, "expected_target"].to_numpy()
)

print("\n--- FOUR-HOUR TARGET AUDIT ---")
print("Model-ready rows:", len(model))
print("Rows with independently verified 4-hour targets:",
      int(comparable.sum()))
print("Rows without an exact 4-hour observation:",
      int((~comparable).sum()))
print("Target mismatches:", int(mismatches.sum()))

if comparable.any():
    print("Verified target agreement:",
          f"{100 * (1 - mismatches.mean()):.2f}%")

print("\nSample mismatches:")
print(
    checked.loc[
        comparable
        & (checked["target_up"] != checked["expected_target"]),
        ["datetime", "target_up", "expected_target",
         "expected_future_close"]
    ].head(10).to_string(index=False)
)

print("\nAudit complete. No model was trained.")
