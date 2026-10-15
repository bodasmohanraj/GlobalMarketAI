import pandas as pd

df = pd.read_csv("gold_features_clean.csv")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").drop_duplicates("datetime")

gaps = df["datetime"].diff().dropna()
print("Total rows:", len(df))
print("\nTime interval distribution:")
print(gaps.value_counts().head(12).to_string())
print("\nLargest gaps:")
print(gaps[gaps > pd.Timedelta(hours=1)].sort_values(ascending=False).head(10).to_string())
