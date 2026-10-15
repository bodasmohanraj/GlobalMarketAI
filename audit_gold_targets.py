from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
file = base / "gold_features_clean.csv"

df = pd.read_csv(file)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
df["target_time"] = pd.to_datetime(df["target_time"], utc=True, errors="coerce")

for col in ["close", "future_close_4h", "target_up"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

expected_time = df["datetime"] + pd.Timedelta(hours=4)
expected_target = (df["future_close_4h"] > df["close"]).astype("Int64")

time_ok = df["target_time"].eq(expected_time)
target_ok = df["target_up"].eq(expected_target)

print("\n=== TARGET ALIGNMENT AUDIT ===")
print("Rows:", len(df))
print("Missing target timestamps:", df["target_time"].isna().sum())
print("Target timestamps exactly +4h:", int(time_ok.sum()), "/", len(df))
print("Target labels match future prices:", int(target_ok.sum()), "/", len(df))

bad = df.loc[
    ~(time_ok & target_ok),
    ["datetime", "close", "target_time", "future_close_4h", "target_up"]
]
print("\nFirst mismatches:")
print(bad.head(10).to_string(index=False) if not bad.empty else "None")

print("\n=== TEST LABEL OVERLAP AUDIT ===")
pred_file = base / "walk_forward_predictions.csv"
pred = pd.read_csv(pred_file)
pred["datetime"] = pd.to_datetime(pred["datetime"], utc=True, errors="coerce")

for fold in sorted(pred["fold"].dropna().unique()):
    test_start = pred.loc[pred["fold"] == fold, "datetime"].min()
    cutoff = test_start - pd.Timedelta(hours=4)
    training_rows = df[df["datetime"] < cutoff]
    latest_target_time = training_rows["target_time"].max()
    print(
        f"Fold {int(fold)} | test start={test_start} | "
        f"latest training target time={latest_target_time} | "
        f"strictly before test={latest_target_time < test_start}"
    )
