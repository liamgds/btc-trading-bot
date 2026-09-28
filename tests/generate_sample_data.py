"""Generate synthetic 15m BTC/USDT-like OHLCV data for offline pipeline testing.

This is SYNTHETIC data for exercising the code — not real market data.
For a real demo, use fetch_historical_data.py instead (needs internet).

Run from anywhere; writes to sample_data/sample_btcusdt_15m.csv relative to
the project root (parent of this tests/ folder).
"""
import os
import numpy as np
import pandas as pd

n = 20000  # ~208 days of 15m candles
dt = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")

# Build distinct regime segments: alternating strong trend / chop / strong trend
# so the strategy's strict multi-filter entry actually gets qualifying setups
# (real BTC has this kind of regime structure; a pure random walk mostly doesn't).
returns = np.zeros(n)
volume_mult = np.ones(n)
i = 0
rng = np.random.default_rng(42)
while i < n:
    seg_len = rng.integers(150, 500)
    seg_len = min(seg_len, n - i)
    kind = rng.choice(["uptrend", "downtrend", "chop"], p=[0.35, 0.25, 0.40])
    if kind == "uptrend":
        drift = rng.uniform(0.0004, 0.0012)
        noise = 0.0025
        volume_mult[i:i+seg_len] = rng.uniform(1.3, 2.2)
    elif kind == "downtrend":
        drift = -rng.uniform(0.0004, 0.0012)
        noise = 0.0030
        volume_mult[i:i+seg_len] = rng.uniform(1.2, 2.0)
    else:
        drift = rng.uniform(-0.0001, 0.0001)
        noise = 0.0018
        volume_mult[i:i+seg_len] = rng.uniform(0.6, 1.1)
    returns[i:i+seg_len] = rng.normal(drift, noise, seg_len)
    i += seg_len

price = 60000 * np.exp(np.cumsum(returns))

high = price * (1 + np.abs(rng.normal(0, 0.0015, n)))
low = price * (1 - np.abs(rng.normal(0, 0.0015, n)))
open_ = np.roll(price, 1)
open_[0] = price[0]
volume = rng.lognormal(mean=3, sigma=0.4, size=n) * 10 * volume_mult

df = pd.DataFrame({
    "timestamp": dt,
    "open": open_,
    "high": np.maximum.reduce([open_, high, price]),
    "low": np.minimum.reduce([open_, low, price]),
    "close": price,
    "volume": volume,
})

out_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sample_btcusdt_15m.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} candles, price range ${df['close'].min():.0f}-${df['close'].max():.0f}")
print(f"Saved to {out_path}")
