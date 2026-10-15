from pathlib import Path

p = Path("audit_multi_horizon.py")
s = p.read_text(encoding="utf-8-sig")

start = s.index("def add_labels(")
end = s.index("\ndef pipeline(", start)

replacement = '''def add_labels(features, minutes):
    """Join each candle to the price at its exact future timestamp."""
    out = features.copy()

    # Keep the original timestamps of the actual future prices.
    future = features[["datetime", "close"]].rename(
        columns={
            "datetime": "target_time",
            "close": "future_close",
        }
    )

    # For a candle at t, the required target is the candle at t + horizon.
    out["target_time"] = (
        out["datetime"] + pd.Timedelta(minutes=minutes)
    )

    # Match t + horizon to the actual candle timestamp, never t - horizon.
    out = out.merge(
        future,
        on="target_time",
        how="left",
        validate="many_to_one",
    )

    out["price_change"] = out["future_close"] - out["close"]
    out["target_up"] = np.where(
        out["future_close"].notna(),
        (out["price_change"] > 0).astype(int),
        np.nan,
    )

    return out

'''

s = s[:start] + replacement + s[end:]
p.write_text(s, encoding="utf-8")
print("Fixed add_labels(): targets now match exact future timestamps.")
