"""
main.py — Single entry point for the trading bot.

    python main.py backtest --data historical_data.csv --capital 10000
    python main.py backtest --data historical_data.csv --relaxed-demo --plot
    python main.py live

For 'live' (the subcommand — meaning "run the polling loop", not a trading
mode), the actual trading mode (paper/demo/testnet) and symbol/timeframe
come from your .env file (see .env.example) via config.Config. This build
is demo-only: setting TRADING_MODE=live in .env is refused at startup —
see config.py.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="BTC/USDT trading bot — Phase 8 strategy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bt = subparsers.add_parser("backtest", help="Run an offline historical backtest")
    bt.add_argument("--data", required=True, help="CSV file with timestamp,open,high,low,close,volume")
    bt.add_argument("--capital", type=float, default=10_000.0)
    bt.add_argument("--plot", action="store_true")
    bt.add_argument("--out", default="backtest_report.txt")
    bt.add_argument("--relaxed-demo", action="store_true",
                    help="DEMO ONLY: loosen entry filters so non-BTC/synthetic data still produces trades")

    subparsers.add_parser("live", help="Run the polling trading loop (mode set via TRADING_MODE in .env: paper/demo/testnet)")

    args = parser.parse_args()

    if args.command == "backtest":
        import backtest
        sys.argv = ["backtest.py", "--data", args.data, "--capital", str(args.capital),
                   "--out", args.out]
        if args.plot:
            sys.argv.append("--plot")
        if args.relaxed_demo:
            sys.argv.append("--relaxed-demo")
        backtest.main()

    elif args.command == "live":
        from bot import TradingBot
        try:
            bot = TradingBot()
        except ValueError as e:
            # Config.validate() raises this for missing/invalid .env settings,
            # including a deliberate refusal if TRADING_MODE=live is set.
            print(f"Configuration problem:\n{e}")
            print("\nCheck your .env file (see .env.example) and try again.")
            sys.exit(1)
        except Exception as e:
            print(f"Couldn't start the bot: {e}")
            print("If you're in 'demo' or 'testnet' mode, check your API keys and "
                 "internet connection. 'paper' mode still needs internet access "
                 "to fetch live prices, even though it needs no API keys.")
            sys.exit(1)
        bot.start()


if __name__ == "__main__":
    main()
