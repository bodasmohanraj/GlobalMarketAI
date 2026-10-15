from pathlib import Path
import yfinance as yf
import pandas as pd

base = Path(__file__).resolve().parent
output = base / "gold_5m_data.csv"

print("Downloading up to 60 days of five-minute gold candles...")

data = yf.download(
    "GC=F",
    period="60d",
    interval="5m",
    progress=False,
    auto_adjust=False
)

if data.empty:
    raise RuntimeError("No data returned. Existing files were not changed.")

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
    raise ValueError(f"Timestamp column not found: {data.columns.tolist()}")

data = data.rename(columns={time_col: "datetime"})
data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
data = data.sort_values("datetime").drop_duplicates("datetime", keep="last")

temp = output.with_suffix(".tmp.csv")
data.to_csv(temp, index=False)
temp.replace(output)

print("Saved:", output)
print("Rows:", len(data))
print("First candle UTC:", data["datetime"].min())
print("Last candle UTC:", data["datetime"].max())
print("Hourly gold_data.csv was not modified.")
