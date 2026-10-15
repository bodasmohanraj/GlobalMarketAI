
import yfinance as yf
import pandas as pd

TICKER = "GC=F"
PERIOD = "1mo"
INTERVAL = "1h"


def download_gold_data():
    print(f"Downloading gold data for {TICKER}...")

    data = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if data.empty:
        print("Yahoo Finance returned no data.")
        print("Try running the script again in a few minutes.")
        return None

    # Handle yfinance's newer multi-level column format.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data.columns = [
        str(column).lower().replace(" ", "_")
        for column in data.columns
    ]

    data = data.loc[:, ~data.columns.duplicated()]

    data.to_csv("gold_data.csv", index=False)

    print(f"Saved {len(data)} rows to gold_data.csv")
    print(data.tail())

    return data


if __name__ == "__main__":
    download_gold_data()
