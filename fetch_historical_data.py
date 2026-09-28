"""
fetch_historical_data.py — Pull real historical candles from Binance and
save them to CSV for backtest.py.

Needs real internet access to Binance's public API (no API key required —
klines are a public endpoint). Run this on your own machine, not in a
sandboxed environment.

Usage:
    python fetch_historical_data.py --symbol BTCUSDT --interval 15m --days 180 --out btcusdt_15m.csv
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from requests.exceptions import RequestException


def main():
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV candles from Binance")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m",
                       help="1m, 5m, 15m, 1h, 4h, 1d, etc.")
    parser.add_argument("--days", type=int, default=180, help="How many days of history to fetch")
    parser.add_argument("--out", default="historical_data.csv")
    args = parser.parse_args()

    print(f"Connecting to Binance public API (no key needed for historical data)...")
    try:
        client = Client()  # public endpoints only
    except (RequestException, BinanceAPIException, BinanceRequestException) as e:
        print(f"\nCouldn't connect to Binance: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    start = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
    print(f"Fetching {args.symbol} {args.interval} candles since {start}...")

    all_klines = []
    current_start = start
    try:
        while True:
            klines = client.get_historical_klines(
                symbol=args.symbol, interval=args.interval,
                start_str=current_start, limit=1000,
            )
            if not klines:
                break
            all_klines.extend(klines)
            if len(klines) < 1000:
                break
            current_start = str(klines[-1][0] + 1)
            print(f"  ...{len(all_klines)} candles so far")
    except BinanceAPIException as e:
        if e.code == -1121:
            print(f"\n'{args.symbol}' isn't a valid Binance symbol. Check the spelling (e.g. BTCUSDT, ETHUSDT).")
        else:
            print(f"\nBinance rejected the request: {e}")
        sys.exit(1)
    except (RequestException, BinanceRequestException) as e:
        print(f"\nNetwork error while fetching data: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    if not all_klines:
        print("No data returned. Check symbol/interval/days.")
        return

    import pandas as pd
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} candles to {args.out}")
    print(f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
