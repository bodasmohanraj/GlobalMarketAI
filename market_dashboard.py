from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gold_market_regimes.csv"

st.set_page_config(
    page_title="Gold Market Intelligence",
    page_icon="📊",
    layout="wide"
)

REGIME_COLORS = {
    "UPTREND": "#16A34A",
    "DOWNTREND": "#DC2626",
    "SIDEWAYS_OR_MIXED": "#64748B",
    "HIGH_VOLATILITY": "#F59E0B",
    "INSUFFICIENT_HISTORY": "#94A3B8",
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for col in ["close", "high", "low", "open", "sma_5", "sma_20", "sma_50",
                "trend_strength_pct", "volatility_20"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime", "close"]).sort_values("datetime")

st.title("🌍 Global Market Intelligence")
st.caption("Gold market research dashboard | Historical analysis, not financial advice")

if not DATA_FILE.exists():
    st.error("gold_market_regimes.csv not found. Run: python market_regime.py")
    st.stop()

df = load_data()

st.sidebar.header("Research Filters")
min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()

start_date = st.sidebar.date_input(
    "Start date", value=max(min_date, max_date - pd.Timedelta(days=90)),
    min_value=min_date, max_value=max_date
)
end_date = st.sidebar.date_input(
    "End date", value=max_date,
    min_value=min_date, max_value=max_date
)

regimes = sorted(df["market_regime"].dropna().unique())
selected_regimes = st.sidebar.multiselect(
    "Market regimes",
    regimes,
    default=[r for r in regimes if r != "INSUFFICIENT_HISTORY"]
)

if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

filtered = df[
    (df["datetime"].dt.date >= start_date) &
    (df["datetime"].dt.date <= end_date) &
    (df["market_regime"].isin(selected_regimes))
].copy()

if filtered.empty:
    st.warning("No data matches these filters. Select another date range or regime.")
    st.stop()

latest = df.iloc[-1]
first_price = filtered.iloc[0]["close"]
last_price = filtered.iloc[-1]["close"]
change_pct = ((last_price / first_price) - 1) * 100 if first_price else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest recorded close", f"${latest['close']:,.2f}")
c2.metric("Latest recorded regime", str(latest["market_regime"]))
c3.metric("Filtered period change", f"{change_pct:+.2f}%")
c4.metric("Filtered observations", f"{len(filtered):,}")

st.caption(
    "Latest recorded candle: "
    + latest["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    + ". The last candle may be incomplete; values are historical data, not live quotes."
)

# ================= LATEST MODEL FORECAST =================
st.divider()
st.header("🤖 Latest Gold Forecast (Research Only)")

PREDICTION_FILE = BASE_DIR / "latest_gold_prediction.csv"

if not PREDICTION_FILE.exists():
    st.info(
        "No latest forecast file found. Run "
        "`python predict_latest_gold.py` to generate it."
    )
else:
    try:
        prediction_df = pd.read_csv(PREDICTION_FILE)

        if prediction_df.empty:
            st.warning("The latest forecast file exists but contains no rows.")
        else:
            forecast = prediction_df.iloc[-1]
            # Normalize headers so display is resilient to small naming differences.
            normalized = {
                str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
                for column in prediction_df.columns
            }

            def find_value(*names):
                for name in names:
                    key = name.strip().lower().replace(" ", "_").replace("-", "_")
                    if key in normalized:
                        value = forecast[normalized[key]]
                        if pd.notna(value):
                            return value
                return None

            signal = find_value("model_signal", "signal", "prediction", "forecast")
            prob_up = find_value(
                "probability_up", "prob_up", "model_probability_up",
                "up_probability", "probability_up_pct"
            )
            prob_down = find_value(
                "probability_down", "prob_down", "model_probability_down",
                "down_probability", "probability_down_pct"
            )
            holdout_accuracy = find_value(
                "holdout_accuracy", "test_accuracy", "accuracy"
            )
            baseline_accuracy = find_value(
                "majority_class_baseline", "baseline_accuracy",
                "holdout_baseline_accuracy"
            )
            forecast_time = find_value(
                "latest_feature_time_utc", "feature_time", "datetime",
                "timestamp", "latest_timestamp"
            )
            latest_close = find_value("latest_close", "close", "last_close")

            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Model signal", str(signal) if signal is not None else "See forecast data"
            )

            def probability_label(value):
                if value is None:
                    return "Not provided"
                try:
                    numeric = float(value)
                    return f"{numeric * 100:.2f}%" if 0 <= numeric <= 1 else f"{numeric:.2f}%"
                except (TypeError, ValueError):
                    return str(value)

            metric_columns[1].metric("Probability up", probability_label(prob_up))
            metric_columns[2].metric("Probability down", probability_label(prob_down))

            if latest_close is not None:
                try:
                    metric_columns[3].metric("Latest close", f"{float(latest_close):,.2f}")
                except (TypeError, ValueError):
                    metric_columns[3].metric("Latest close", str(latest_close))
            else:
                metric_columns[3].metric("Latest close", "Not provided")

            if forecast_time is not None:
                st.caption(f"Forecast feature timestamp: {forecast_time} (check source timezone).")

            accuracy_cols = st.columns(2)
            for col, label, value in [
                (accuracy_cols[0], "Holdout accuracy", holdout_accuracy),
                (accuracy_cols[1], "Baseline accuracy", baseline_accuracy),
            ]:
                if value is None:
                    col.metric(label, "Not provided")
                else:
                    try:
                        numeric = float(value)
                        display = f"{numeric * 100:.1f}%" if 0 <= numeric <= 1 else f"{numeric:.1f}%"
                        col.metric(label, display)
                    except (TypeError, ValueError):
                        col.metric(label, str(value))

            try:
                if holdout_accuracy is not None and baseline_accuracy is not None:
                    holdout_num = float(holdout_accuracy)
                    baseline_num = float(baseline_accuracy)
                    # Accuracy values may be stored as fractions or percentages.
                    if holdout_num > 1:
                        holdout_num /= 100
                    if baseline_num > 1:
                        baseline_num /= 100
                    if holdout_num <= baseline_num:
                        st.warning(
                            "The latest reported holdout accuracy does not beat the "
                            "baseline. The model has not demonstrated a predictive edge."
                        )
                    else:
                        st.info(
                            "Holdout accuracy is above the displayed baseline, but this "
                            "alone does not prove a durable or profitable predictive edge."
                        )
                else:
                    st.warning(
                        "Accuracy and baseline fields were not both found in the CSV. "
                        "Inspect the forecast data below before interpreting reliability."
                    )
            except (TypeError, ValueError):
                st.warning("Could not compare accuracy with baseline from the saved CSV.")

            st.subheader("Saved Forecast Data")
            st.dataframe(prediction_df, width="stretch", hide_index=True)

    except Exception as exc:
        st.error(f"Could not read the latest forecast file: {exc}")

st.caption(
    "Research only. Model probabilities may be uncalibrated, historical performance can "
    "change, and this panel is not a trading recommendation."
)
# ========================================================

st.divider()
st.subheader("Gold Price and Moving Averages")

price = filtered.sort_values("datetime")
fig = px.line(
    price,
    x="datetime",
    y="close",
    title="Gold close price",
    labels={"datetime": "Time (UTC)", "close": "Gold price (source units)"}
)
for col, color in [
    ("sma_5", "#14B8A6"),
    ("sma_20", "#3B82F6"),
    ("sma_50", "#A855F7")
]:
    if col in price.columns:
        fig.add_scatter(
            x=price["datetime"], y=price[col],
            mode="lines", name=col.upper(),
            line={"color": color, "width": 1.5}
        )
fig.update_layout(hovermode="x unified", legend_title_text="Series")
st.plotly_chart(fig, width="stretch", key="main_chart")

st.subheader("Market Regime Distribution")
counts = filtered["market_regime"].value_counts().rename_axis("regime").reset_index(name="count")
pie = px.pie(
    counts, names="regime", values="count", hole=0.45,
    color="regime", color_discrete_map=REGIME_COLORS,
    title="Observations by regime"
)
st.plotly_chart(pie, width="stretch", key="regime_pie_chart")

st.subheader("Volatility and Trend Strength")
left, right = st.columns(2)

with left:
    vol = px.line(
        filtered, x="datetime", y="volatility_20",
        title="20-observation rolling volatility",
        labels={"datetime": "Time (UTC)", "volatility_20": "Volatility (%)"}
    )
    st.plotly_chart(vol, width="stretch", key="volatility_chart")

with right:
    trend = px.line(
        filtered, x="datetime", y="trend_strength_pct",
        title="Short-term trend strength",
        labels={"datetime": "Time (UTC)", "trend_strength_pct": "Trend strength (%)"}
    )
    trend.add_hline(y=0, line_dash="dash")
    st.plotly_chart(trend, width="stretch", key="trend_strength_chart")

st.subheader("Historical Regime Timeline")
timeline = px.scatter(
    filtered, x="datetime", y="market_regime",
    color="market_regime", color_discrete_map=REGIME_COLORS,
    title="Regime labels over time",
    labels={"datetime": "Time (UTC)", "market_regime": "Regime"}
)
st.plotly_chart(timeline, width="stretch", key="regime_timeline_chart")

st.subheader("Recent Observations")
show_cols = [
    "datetime", "close", "sma_5", "sma_20", "sma_50",
    "trend_strength_pct", "volatility_20", "market_regime"
]
st.dataframe(
    filtered[show_cols].sort_values("datetime", ascending=False).head(100),
    width="stretch",
    hide_index=True
)

st.download_button(
    "Download filtered research data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="gold_market_research_export.csv",
    mime="text/csv",
    key="download_market_research_csv"
)

st.caption(
    "Research only. Regime classifications are rule-based and do not establish a profitable trading edge."
)

# ================= REGIME TRANSITION INTELLIGENCE =================
st.divider()
st.header("🔄 Regime Transition Intelligence")

EPISODES_FILE = BASE_DIR / "gold_regime_episodes.csv"

if not EPISODES_FILE.exists():
    st.warning(
        "Run `python regime_transitions.py` first to generate "
        "the regime episode data."
    )
else:
    episodes = pd.read_csv(EPISODES_FILE)
    episodes["start_time"] = pd.to_datetime(
        episodes["start_time"], utc=True, errors="coerce"
    )
    episodes["end_time"] = pd.to_datetime(
        episodes["end_time"], utc=True, errors="coerce"
    )

    episodes = episodes.dropna(
        subset=["start_time", "market_regime"]
    ).sort_values("start_time").reset_index(drop=True)

    # Recalculate transitions in chronological order.
    episodes["next_regime"] = episodes["market_regime"].shift(-1)

    valid_transitions = episodes.dropna(
        subset=["next_regime"]
    ).copy()

    transition_counts = pd.crosstab(
        valid_transitions["market_regime"],
        valid_transitions["next_regime"]
    )

    transition_pct = transition_counts.div(
        transition_counts.sum(axis=1).replace(0, float("nan")),
        axis=0
    ) * 100

    regime_order = [
        "UPTREND",
        "DOWNTREND",
        "SIDEWAYS_OR_MIXED",
        "HIGH_VOLATILITY"
    ]

    transition_pct = transition_pct.reindex(
        index=regime_order,
        columns=regime_order
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Transition Probability Heatmap")
        st.caption(
            "Each cell shows the historical percentage of episodes "
            "that transitioned from the row regime to the column regime."
        )

        heatmap = px.imshow(
            transition_pct,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Next regime",
                "y": "Current regime",
                "color": "Percentage (%)"
            },
            zmin=0,
            zmax=100
        )
        heatmap.update_layout(
            xaxis_title="Next observed regime",
            yaxis_title="Current regime"
        )
        st.plotly_chart(heatmap, width="stretch", key="transition_heatmap_chart")

    with right:
        st.subheader("Episode Duration")
        duration = episodes.groupby("market_regime", as_index=False).agg(
            episodes=("market_regime", "size"),
            median_observations=("observations", "median"),
            median_elapsed_hours=("elapsed_hours", "median")
        )

        duration = duration.sort_values(
            "median_observations", ascending=True
        )

        duration_chart = px.bar(
            duration,
            x="median_observations",
            y="market_regime",
            orientation="h",
            text="median_observations",
            labels={
                "median_observations": "Median observations per episode",
                "market_regime": "Regime"
            },
            title="Typical episode length"
        )
        st.plotly_chart(duration_chart, width="stretch", key="episode_duration_chart")

    st.subheader("Transition Counts")
    st.dataframe(transition_counts, width="stretch")

    st.subheader("Episode Duration Details")
    st.dataframe(
        duration.sort_values("episodes", ascending=False),
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "Download transition percentages",
        data=transition_pct.to_csv().encode("utf-8"),
        file_name="gold_regime_transition_percentages.csv",
        mime="text/csv",
        key="download_transition_percentages"
    )

    st.caption(
        "Historical descriptive statistics only. Transition percentages "
        "are not forecasts. Episode duration is based on observed candles, "
        "and elapsed clock hours may include market closures or data gaps."
    )
# ================================================================


# ================= SUPPORT & RESISTANCE INTELLIGENCE =================
st.divider()
st.header("🎯 Support & Resistance Intelligence")

SR_FILE = BASE_DIR / "gold_support_resistance.csv"

if not SR_FILE.exists():
    st.warning("Run `python support_resistance.py` first.")
else:
    sr = pd.read_csv(SR_FILE)
    sr["datetime"] = pd.to_datetime(
        sr["datetime"], utc=True, errors="coerce"
    )
    sr = sr.dropna(subset=["datetime", "close"]).sort_values("datetime")

    sr_start = sr["datetime"].min().date()
    sr_end = sr["datetime"].max().date()

    sr_recent_days = st.selectbox(
        "Support/resistance chart period",
        [7, 14, 30, 90, 180],
        index=2,
        key="sr_period_main"
    )

    sr_cutoff = sr["datetime"].max() - pd.Timedelta(days=sr_recent_days)
    sr_view = sr[sr["datetime"] >= sr_cutoff].copy()

    if sr_view.empty:
        st.info("No observations available for this period.")
    else:
        sr_latest = sr_view.iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest recorded close", f"{sr_latest['close']:,.2f}")
        c2.metric(
            "Nearest support",
            f"{sr_latest['nearest_support']:,.2f}"
            if pd.notna(sr_latest["nearest_support"]) else "Unavailable"
        )
        c3.metric(
            "Nearest resistance",
            f"{sr_latest['nearest_resistance']:,.2f}"
            if pd.notna(sr_latest["nearest_resistance"]) else "Unavailable"
        )

        fig_sr = px.line(
            sr_view,
            x="datetime",
            y="close",
            title="Gold Price with Estimated Support and Resistance",
            labels={"datetime": "Time (UTC)", "close": "Gold price"}
        )

        fig_sr.add_scatter(
            x=sr_view["datetime"],
            y=sr_view["nearest_support"],
            mode="lines",
            name="Estimated support",
            line={"dash": "dash", "color": "#16A34A"}
        )
        fig_sr.add_scatter(
            x=sr_view["datetime"],
            y=sr_view["nearest_resistance"],
            mode="lines",
            name="Estimated resistance",
            line={"dash": "dash", "color": "#DC2626"}
        )

        breakout_rows = sr_view[
            sr_view["possible_resistance_breakout"].astype(str).str.lower() == "true"
        ]
        breakdown_rows = sr_view[
            sr_view["possible_support_breakdown"].astype(str).str.lower() == "true"
        ]

        if not breakout_rows.empty:
            fig_sr.add_scatter(
                x=breakout_rows["datetime"],
                y=breakout_rows["close"],
                mode="markers",
                name="Possible resistance breakout",
                marker={"symbol": "triangle-up", "size": 10, "color": "#2563EB"}
            )

        if not breakdown_rows.empty:
            fig_sr.add_scatter(
                x=breakdown_rows["datetime"],
                y=breakdown_rows["close"],
                mode="markers",
                name="Possible support breakdown",
                marker={"symbol": "triangle-down", "size": 10, "color": "#F59E0B"}
            )

        fig_sr.update_layout(hovermode="x unified")
        st.plotly_chart(fig_sr, width="stretch", key="support_resistance_chart")

        st.subheader("Recent Support & Resistance Observations")
        display_cols = [
            "datetime", "close", "nearest_support", "support_touches",
            "nearest_resistance", "resistance_touches",
            "possible_resistance_breakout", "possible_support_breakdown"
        ]
        st.dataframe(
            sr_view[display_cols].sort_values(
                "datetime", ascending=False
            ).head(100),
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download support/resistance data",
            data=sr_view.to_csv(index=False).encode("utf-8"),
            file_name="gold_support_resistance_filtered.csv",
            mime="text/csv",
            key="download_support_resistance_csv"
        )

        st.caption(
            "Research visualization only. Zone changes can affect breakout flags; "
            "these events have not yet been independently validated."
        )
# ===================================================================
