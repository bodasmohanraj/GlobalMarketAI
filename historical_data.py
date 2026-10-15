
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "gold_data.csv"
TICKER = "GC=F"

PERIODS = [
    ("1h", "730d"),
    ("1d", "5y"),
]


def download_data(interval, period):
    print(
        f"\nDownloading {TICKER}: "
        f"interval={interval}, period={period}"
    )

    for attempt in range(1, 4):
        try:
            data = yf.download(
                TICKER,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data is not None and not data.empty:
                return data

            print(f"Attempt {attempt}: no data returned.")

        except Exception as error:
            print(f"Attempt {attempt} failed: {error}")

        if attempt < 3:
            time.sleep(2)

    return pd.DataFrame()


def normalize_data(data):
    if data.empty:
        return pd.DataFrame()

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    data.index.name = "datetime"
    data = data.reset_index()

    if "datetime" not in data.columns:
        data = data.rename(
            columns={data.columns[0]: "datetime"}
        )

    data["datetime"] = pd.to_datetime(
        data["datetime"], utc=True, errors="coerce"
    )

    if "close" not in data.columns:
        raise ValueError(
            f"Missing close column. Columns: {list(data.columns)}"
        )

    data["close"] = pd.to_numeric(
        data["close"], errors="coerce"
    )

    data = data.dropna(subset=["datetime", "close"])
    data = data.sort_values("datetime")
    data = data.drop_duplicates("datetime", keep="last")

    return data


def main():
    downloaded = pd.DataFrame()
    selected_interval = None

    for interval, period in PERIODS:
        downloaded = normalize_data(
            download_data(interval, period)
        )

        if not downloaded.empty:
            selected_interval = interval
            break

    if downloaded.empty:
        print("\nDownload failed. Existing data was preserved.")
        print("Check your internet connection and Yahoo Finance.")
        return

    # Do not mix daily bars into the hourly model's dataset.
    if selected_interval != "1h":
        print(
            "\nYahoo did not return hourly data. Daily data was "
            "not saved because your current model uses hourly candles."
        )
        print("Existing gold_data.csv was preserved.")
        return

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)

        if "datetime" not in existing.columns:
            raise ValueError(
                "Existing CSV has no datetime column. "
                "It was not modified."
            )

        existing["datetime"] = pd.to_datetime(
            existing["datetime"], utc=True, errors="coerce"
        )

        if "close" not in existing.columns:
            raise ValueError(
                "Existing CSV has no close column. "
                "It was not modified."
            )

        existing["close"] = pd.to_numeric(
            existing["close"], errors="coerce"
        )

        existing = existing.dropna(
            subset=["datetime", "close"]
        )

        # Combine old and downloaded records. New downloaded rows
        # take priority when the same timestamp appears in both.
        combined = pd.concat(
            [existing, downloaded],
            ignore_index=True,
            sort=False,
        )

        combined = combined.sort_values("datetime")
        combined = combined.drop_duplicates(
            subset=["datetime"], keep="last"
        )
    else:
        combined = downloaded.copy()

    combined = combined.sort_values("datetime").reset_index(drop=True)

    temporary_file = OUTPUT_FILE.with_suffix(".tmp.csv")
    combined.to_csv(temporary_file, index=False)
    temporary_file.replace(OUTPUT_FILE)

    latest_time = combined["datetime"].iloc[-1]
    latest_close = combined["close"].iloc[-1]

    print("\nHourly data download completed.")
    print(f"Interval used: {selected_interval}")
    print(f"Downloaded rows: {len(downloaded)}")
    print(f"Total merged rows: {len(combined)}")
    print(f"Latest candle (UTC): {latest_time}")
    print(f"Latest close: {latest_close:.2f}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
