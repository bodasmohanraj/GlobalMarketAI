from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gold_market_regimes.csv"
OUTPUT_TRANSITIONS = BASE_DIR / "gold_regime_transitions.csv"
OUTPUT_EPISODES = BASE_DIR / "gold_regime_episodes.csv"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "gold_market_regimes.csv not found. Run market_regime.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["datetime", "market_regime"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

    # Exclude warm-up rows because they do not have enough history.
    df = df[df["market_regime"] != "INSUFFICIENT_HISTORY"].copy()
    df = df.reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Not enough classified observations.")

    # Each episode is a consecutive run of the same regime.
    df["episode_id"] = (
        df["market_regime"] != df["market_regime"].shift()
    ).cumsum()

    episodes = (
        df.groupby(["episode_id", "market_regime"], as_index=False)
        .agg(
            start_time=("datetime", "min"),
            end_time=("datetime", "max"),
            observations=("datetime", "size"),
            start_close=("close", "first"),
            end_close=("close", "last"),
        )
    )

    # This is elapsed clock time, not the number of trading hours.
    episodes["elapsed_hours"] = (
        episodes["end_time"] - episodes["start_time"]
    ).dt.total_seconds() / 3600

    episodes["price_change_pct"] = (
        (episodes["end_close"] / episodes["start_close"]) - 1
    ) * 100

    episodes["next_regime"] = episodes["market_regime"].shift(-1)
    episodes["next_start_time"] = episodes["start_time"].shift(-1)

    episodes.to_csv(OUTPUT_EPISODES, index=False)

    # Count transitions between consecutive episodes.
    transitions = episodes.dropna(subset=["next_regime"]).copy()
    transition_counts = pd.crosstab(
        transitions["market_regime"],
        transitions["next_regime"]
    )

    # Convert counts into row-wise percentages.
    transition_probabilities = transition_counts.div(
        transition_counts.sum(axis=1).replace(0, pd.NA), axis=0
    ) * 100

    transition_counts.to_csv(OUTPUT_TRANSITIONS.with_name(
        "gold_regime_transition_counts.csv"
    ))
    transition_probabilities.to_csv(OUTPUT_TRANSITIONS)

    print("\n=== REGIME TRANSITION ANALYSIS ===")
    print("Classified observations:", len(df))
    print("Regime episodes:", len(episodes))
    print("Regime changes:", len(transitions))

    print("\n=== TRANSITION COUNTS ===")
    print(transition_counts.to_string())

    print("\n=== NEXT-REGIME FREQUENCY (%) ===")
    print(transition_probabilities.round(1).to_string())

    print("\n=== EPISODE DURATION SUMMARY ===")
    summary = episodes.groupby("market_regime").agg(
        episodes=("episode_id", "count"),
        median_observations=("observations", "median"),
        mean_observations=("observations", "mean"),
        median_elapsed_hours=("elapsed_hours", "median"),
    )
    print(summary.round(2).to_string())

    print("\n=== MOST RECENT EPISODES ===")
    print(
        episodes[
            ["start_time", "end_time", "market_regime",
             "observations", "elapsed_hours", "price_change_pct",
             "next_regime"]
        ].tail(10).to_string(index=False)
    )

    print("\nSaved:")
    print(" - gold_regime_transition_counts.csv")
    print(" - gold_regime_transitions.csv")
    print(" - gold_regime_episodes.csv")
    print("\nResearch statistics only; not trading signals.")

if __name__ == "__main__":
    main()
