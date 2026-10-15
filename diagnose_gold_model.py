from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "gold_features_clean.csv"
PRED = BASE / "walk_forward_predictions.csv"

FEATURES = [
    "return_1h", "return_4h", "return_12h", "sma_5_gap",
    "sma_20_gap", "trend_5_20", "volatility_10",
    "volatility_20", "rsi_14"
]

def main():
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    for col in FEATURES + ["target_up"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (df.dropna(subset=["datetime", "target_up", *FEATURES])
            .sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True))
    df["target_up"] = df["target_up"].astype(int)

    pred = pd.read_csv(PRED)
    pred["datetime"] = pd.to_datetime(pred["datetime"], utc=True, errors="coerce")
    pred = pred.dropna(subset=["datetime"])

    print("\n=== 1. TARGET UP RATE BY YEAR ===")
    print(df.assign(year=df.datetime.dt.year)
            .groupby("year")["target_up"]
            .agg(["count", "mean"]).round(4).to_string())

    print("\n=== 2. FEATURE/TARGET CORRELATIONS ===")
    print(df[FEATURES + ["target_up"]].corr()["target_up"]
            .drop("target_up").sort_values().round(4).to_string())

    print("\n=== 3. MODEL COEFFICIENTS ===")
    fold = int(pred["fold"].max())
    test_start = pred.loc[pred.fold == fold, "datetime"].min()
    train = df[df.datetime < test_start - pd.Timedelta(hours=4)]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])
    model.fit(train[FEATURES], train["target_up"])

    coefficients = pd.Series(
        model.named_steps["classifier"].coef_[0], index=FEATURES
    ).sort_values()
    print("Training UP rate:", round(train["target_up"].mean(), 4))
    print(coefficients.round(4).to_string())

    print("\n=== 4. PREDICTED PROBABILITIES ===")
    p = pred["model_probability_up"].astype(float)
    print(p.describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())

    print("\n=== 5. THRESHOLD DIAGNOSTIC ONLY ===")
    actual = pred["actual_target_up"].astype(int)
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
        guess = (p >= threshold).astype(int)
        print(
            f"Threshold {threshold:.2f}: "
            f"accuracy={(guess == actual).mean():.4f}, "
            f"predicted UP rate={guess.mean():.4f}"
        )

    print("\nDo not choose a final threshold using these test results.")
    print("Next, verify target timestamps and feature construction.")

if __name__ == "__main__":
    main()
