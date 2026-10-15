from pathlib import Path
import yfinance as yf
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "gold_5m_data.csv"

print("Downloading five-minute gold futures data...")
data = yf.download(
    "GC=F",
    period="5d",
    interval="5m",
    progress=False,
    auto_adjust=False
)

if data.empty:
    raise RuntimeError("No five-minute candles returned.")

# Handle Yahoo Finance's MultiIndex columns.
if isinstance(data.columns, pd.MultiIndex):
    if "GC=F" in data.columns.get_level_values(-1):
        data = data.xs("GC=F", axis=1, level=-1)
    else:
        data.columns = data.columns.get_level_values(0)

data = data.reset_index()
time_col = next(
    (c for c in data.columns if str(c).lower() in ("datetime", "date")),
    None
)

if time_col is None:
    raise ValueError(f"Cannot identify timestamp column: {data.columns.tolist()}")

data = data.rename(columns={time_col: "datetime"})
data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
data = data.sort_values("datetime").drop_duplicates("datetime", keep="last")

data.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Rows:", len(data))
print("First candle (UTC):", data["datetime"].min())
print("Last candle (UTC):", data["datetime"].max())
print("Columns:", data.columns.tolist())
print(data.tail(3).to_string(index=False))
