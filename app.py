import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

st.set_page_config(
    page_title="Global Market Intelligence AI",
    page_icon="📈",
    layout="wide",
)

FEATURES = [
    "return_1h",
    "sma_5",
    "sma_20",
    "volatility_10",
    "rsi_14",
]


@st.cache_data(ttl=300)
def fetch_gold():
    data = yf.download(
        "GC=F",
        period="1mo",
        interval="1h",
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Normalize Yahoo Finance's timestamp column
    time_col = "Datetime" if "Datetime" in data.columns else "Date"
    data = data.rename(columns={time_col: "Datetime"})

    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    data = data[[c for c in required if c in data.columns]].copy()
    data = data.dropna(subset=["Close"])
    return data


def create_features(data):
    df = data.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["return_1h"] = df["Close"].pct_change()
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["volatility_10"] = df["return_1h"].rolling(10).std()

    change = df["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["future_close"] = df["Close"].shift(-4)
    df["target_up"] = np.where(
        df["future_close"].isna(),
        np.nan,
        (df["future_close"] > df["Close"]).astype(float),
    )

    return df


def train_and_evaluate(df):
    labeled = df.dropna(subset=["target_up"]).copy()

    if len(labeled) < 100:
        return None, None, None

    X = labeled[FEATURES]
    y = labeled["target_up"].astype(int)

    split = int(len(labeled) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2:
        return None, None, None

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "test_rows": len(y_test),
    }

    return model, metrics, labeled


st.title("🌍 Global Market Intelligence AI")
st.caption(
    "Experimental gold futures analysis · Hourly data · "
    "Research use only, not financial advice"
)

with st.sidebar:
    st.header("Market settings")
    st.write("**Instrument:** COMEX Gold Futures (`GC=F`)")
    st.write("**Data source:** Yahoo Finance")
    st.write("**Prediction horizon:** 4 available hourly candles")
    if st.button("🔄 Refresh market data"):
        fetch_gold.clear()
        st.rerun()

with st.spinner("Loading gold-market data..."):
    raw = fetch_gold()

if raw.empty:
    st.error(
        "No market data received. Check your internet connection "
        "or try again later."
    )
    st.stop()

df = create_features(raw)
latest = df.iloc[-1]

model, metrics, labeled = train_and_evaluate(df)

price = float(latest["Close"])
previous = float(df.iloc[-2]["Close"]) if len(df) > 1 else price
change_pct = (price / previous - 1) * 100 if previous else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest available close", f"${price:,.2f}")
c2.metric("Change vs prior candle", f"{change_pct:+.2f}%")
c3.metric(
    "RSI (14)",
    f"{latest['rsi_14']:.1f}"
    if pd.notna(latest["rsi_14"]) else "N/A",
)
c4.metric(
    "20-candle moving average",
    f"${latest['sma_20']:,.2f}"
    if pd.notna(latest["sma_20"]) else "N/A",
)

st.warning(
    "Yahoo Finance data may be delayed or incomplete. "
    "The latest candle may still be forming. Model predictions "
    "are experimental and do not imply profitable trades."
)

st.subheader("Gold futures price history")

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df["Datetime"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Gold futures",
    )
)
fig.add_trace(
    go.Scatter(
        x=df["Datetime"],
        y=df["sma_5"],
        mode="lines",
        name="SMA 5",
    )
)
fig.add_trace(
    go.Scatter(
        x=df["Datetime"],
        y=df["sma_20"],
        mode="lines",
        name="SMA 20",
    )
)
fig.update_layout(
    height=520,
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)

with left:
    st.subheader("Momentum and volatility")

    indicator = go.Figure()
    indicator.add_trace(
        go.Scatter(
            x=df["Datetime"],
            y=df["rsi_14"],
            mode="lines",
            name="RSI 14",
        )
    )
    indicator.add_hline(y=70, line_dash="dash")
    indicator.add_hline(y=30, line_dash="dash")
    indicator.update_layout(
        height=300,
        yaxis_title="RSI",
        template="plotly_white",
    )
    st.plotly_chart(indicator, width="stretch")
    st.line_chart(
        df.set_index("Datetime")["volatility_10"],
        height=200,
    )

with right:
    st.subheader("Experimental direction model")

    if model is None:
        st.info("Not enough labeled data to train the model yet.")
    else:
        # Latest row has no known target and is used only for prediction
        latest_features = df[FEATURES].iloc[[-1]]
        probabilities = model.predict_proba(latest_features)[0]
        classes = list(model.named_steps["logisticregression"].classes_)
        up_probability = float(probabilities[classes.index(1)])

        direction = "UP" if up_probability >= 0.5 else "DOWN / NOT UP"
        st.metric("Model direction estimate", direction)
        st.metric("Estimated probability of UP", f"{up_probability:.1%}")

        st.caption(
            "Probability is not necessarily calibrated and is not "
            "a probability of profit."
        )

        st.write("**Chronological holdout evaluation**")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric(
            "Balanced accuracy",
            f"{metrics['balanced_accuracy']:.1%}",
        )
        st.caption(f"Test examples: {metrics['test_rows']}")

st.subheader("Latest market data")
st.dataframe(
    df[
        [
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "rsi_14",
            "sma_20",
        ]
    ].tail(20),
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download analyzed market data",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="gold_market_analysis.csv",
    mime="text/csv",
)

st.caption(
    "Prototype only. No orders are placed. Validate data quality, "
    "out-of-sample performance, transaction costs and slippage "
    "before considering any trading strategy."
)