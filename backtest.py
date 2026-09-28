"""
backtest.py — Run strategy.py's Phase 8 logic against historical OHLCV data
and produce a full performance report + equity curve chart.

This is the safest thing to demo live at the exhibition: it needs no
internet connection and no exchange account. Feed it a CSV of historical
candles (from fetch_historical_data.py, or your own source) and it runs
entirely offline.

Usage:
    python backtest.py --data path/to/candles.csv --capital 10000
    python backtest.py --data path/to/candles.csv --capital 10000 --plot
"""

import argparse
import sys

import pandas as pd

import strategy
from performance_analytics import PerformanceAnalytics


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Data file is missing required columns: {missing}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def run_backtest(df: pd.DataFrame, capital: float, cfg=None) -> dict:
    return strategy.run(df, capital=capital, cfg=cfg)


def build_report(result: dict, capital: float) -> str:
    analytics = PerformanceAnalytics(initial_capital=capital)
    for t in result["trades"]:
        analytics.record_trade({
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "timestamp": t["exit_time"],
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pnl": t["pnl"],
            "pnl_percent": t["pnl_pct"],  # strategy.py uses pnl_pct; analytics expects pnl_percent
            "side": "LONG",
            "symbol": "BTCUSDT",
        })
    return analytics.generate_performance_report(include_trades=True), analytics


def plot_equity_curve(result: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ec = pd.DataFrame(result["equity_curve"])
    if ec.empty:
        print("No equity curve data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ec["timestamp"], ec["equity"], linewidth=1.2, color="#2563eb")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Equity curve saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest the strategy against historical candles")
    parser.add_argument("--data", required=True, help="Path to CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital")
    parser.add_argument("--plot", action="store_true", help="Save an equity curve PNG alongside the report")
    parser.add_argument("--out", default="backtest_report.txt", help="Path to save the text report")
    parser.add_argument(
        "--relaxed-demo", action="store_true",
        help=("DEMO/TESTING ONLY. Loosens entry filters (ADX/RSI/volume/session "
             "thresholds) so the pipeline produces visible trades even on "
             "synthetic or non-BTC data. Do NOT use this for evaluating real "
             "strategy performance or allocating real capital — it exists so "
             "you can prove the pipeline runs end-to-end without needing a "
             "live internet connection to fetch real BTC history."),
    )
    args = parser.parse_args()

    cfg = dict(strategy.DEFAULT_CFG)
    if args.relaxed_demo:
        print("*** --relaxed-demo active: entry filters loosened for demo purposes only ***")
        cfg.update({
            "adx_min": 12.0, "di_spread_min": 1.0, "sus_vol_threshold": 0.9,
            "min_session_score": 0, "rsi_floor": 35.0, "rsi_ceil": 70.0,
        })

    print(f"Loading candles from {args.data} ...")
    df = load_candles(args.data)
    print(f"Loaded {len(df)} candles, {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("Running backtest...")
    result = run_backtest(df, capital=args.capital, cfg=cfg)

    report, analytics = build_report(result, args.capital)
    print(report)

    with open(args.out, "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.out}")

    if args.plot:
        plot_equity_curve(result, args.out.replace(".txt", "_equity.png"))


if __name__ == "__main__":
    sys.exit(main())
