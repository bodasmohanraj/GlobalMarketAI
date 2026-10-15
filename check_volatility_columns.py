
import pandas as pd

for filename in [
    "gold_features_clean.csv",
    "gold_backtest_nonoverlap.csv",
]:
    df = pd.read_csv(filename)

    print(f"\n--- {filename} ---")
    print("Columns:")
    print(df.columns.tolist())

    matches = [
        col for col in df.columns
        if "volatility" in col.lower()
    ]
    print("Volatility columns:", matches)
