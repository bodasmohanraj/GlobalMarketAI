from pathlib import Path

p = Path("audit_multi_horizon.py")
s = p.read_text(encoding="utf-8-sig")

old = '''    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
'''

new = '''    returns = close.pct_change()
    x["volatility_12"] = returns.rolling(12, min_periods=12).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
'''

if 'x["volatility_12"]' in s:
    print("volatility_12 already exists.")
elif old in s:
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print("SUCCESS: Added volatility_12.")
else:
    print("Could not find the expected code section. No changes made.")
