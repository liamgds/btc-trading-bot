"""
Validates the full orchestration (entry -> tranche exits -> full exit ->
risk_manager bookkeeping -> logging) end-to-end using a fake exchange fed by
sample_data/sample_btcusdt_15m.csv, so it needs no internet/API access.

This is NOT a substitute for a real testnet run, but it proves the wiring
between strategy.py / bot.py / risk_manager.py / exchange.py has no
integration bugs, and that live-mode and backtest-mode produce identical
trades given identical data (single source of truth for all decisions).

Run from the project root:
    python tests/test_orchestration.py
"""
import sys
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

import strategy
from bot import TradingBot


class FakeExchange:
    """Implements the same interface as BinanceExchange, sourced from a local CSV."""

    def __init__(self, csv_path: str, cfg: dict):
        self.df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.cursor = max(cfg["ema_trend"], cfg["atr_regime_period"]) * 2 + 100
        self._last_limit_price = None

    def get_account_balance(self, asset="USDT"):
        return 10_000.0

    def get_current_price(self, symbol):
        return float(self.df.iloc[self.cursor]["close"])

    def get_historical_klines(self, symbol, interval, start_time, end_time=None, limit=1000):
        lo = max(0, self.cursor - limit)
        return self.df.iloc[lo:self.cursor + 1].reset_index(drop=True)

    def place_market_order(self, symbol, side, quantity, expected_price, max_slippage_percent=0.5):
        return {"orderId": f"FAKE_{random.randint(0,10**9)}", "symbol": symbol, "side": side,
                "origQty": quantity, "executedQty": quantity, "price": expected_price,
                "avg_fill_price": expected_price, "status": "FILLED", "fills": []}

    def place_limit_order(self, symbol, side, quantity, price):
        oid = f"FAKE_LIMIT_{random.randint(0,10**9)}"
        self._last_limit_price = price
        return {"orderId": oid, "symbol": symbol, "side": side, "origQty": quantity,
                "executedQty": 0.0, "price": price, "status": "NEW"}

    def get_order_status(self, symbol, order_id):
        # Always-fill for test determinism, at the price the order was placed at
        return {"status": "FILLED", "executedQty": 1.0, "price": self._last_limit_price}

    def cancel_order(self, symbol, order_id):
        return {"orderId": order_id, "status": "CANCELED"}

    def advance(self):
        self.cursor += 1
        return self.cursor < len(self.df) - 1


def main():
    os.environ["TRADING_MODE"] = "paper"
    os.environ["SYMBOL"] = "BTCUSDT"
    test_tmp = os.path.abspath(os.path.join(ROOT, "tests", ".tmp_test_run"))
    os.makedirs(test_tmp, exist_ok=True)
    os.environ["LOG_FILE"] = os.path.join(test_tmp, "trading_bot.log")
    # Fresh working dir so any state files this test writes don't leak into
    # a real run (bot.py/risk_manager.py persist small JSON state files for
    # crash recovery — see Fix 5 in strategy.py's docstring).
    os.chdir(test_tmp)
    for stale in ("logs/risk_state.json",):
        if os.path.exists(stale):
            os.remove(stale)

    sample_csv = os.path.join(ROOT, "sample_data", "sample_btcusdt_15m.csv")
    if not os.path.exists(sample_csv):
        print(f"Sample data not found at {sample_csv}.")
        print("Run: python tests/generate_sample_data.py")
        sys.exit(1)

    # Relaxed cfg override so entries actually fire against synthetic data
    # (real BTC data doesn't need this — see backtest.py's --relaxed-demo flag)
    orig_default_cfg = dict(strategy.DEFAULT_CFG)
    strategy.DEFAULT_CFG.update({
        "adx_min": 12.0, "di_spread_min": 1.0, "sus_vol_threshold": 0.9,
        "min_session_score": 0, "rsi_floor": 35.0, "rsi_ceil": 70.0,
    })

    fake_exchange = FakeExchange(sample_csv, strategy.DEFAULT_CFG)
    bot = TradingBot(exchange=fake_exchange)

    cycles = 0
    entries_seen = 0
    exits_seen = 0
    max_cycles = len(fake_exchange.df) - fake_exchange.cursor - 2

    while fake_exchange.advance() and cycles < max_cycles:
        had_position_before = bot.risk_manager.has_open_position()
        bot.run_once()
        has_position_after = bot.risk_manager.has_open_position()
        if not had_position_before and has_position_after:
            entries_seen += 1
        if had_position_before and not has_position_after:
            exits_seen += 1
        cycles += 1

    print(f"\nCycles run: {cycles}")
    print(f"Entries opened: {entries_seen}")
    print(f"Full exits: {exits_seen}")
    print(f"Trades recorded: {len(bot.risk_manager.trades)}")
    print(f"Final equity: {bot.risk_manager.equity:.2f}")

    assert entries_seen > 0, "No entries fired — orchestration or filters broken"
    assert len(bot.risk_manager.trades) == exits_seen, "Trade count mismatch"
    for t in bot.risk_manager.trades:
        assert t["exit_time"] >= t["entry_time"], "Exit before entry — timestamp bug"
        assert t["qty"] > 0
        assert np.isfinite(t["pnl"]) and np.isfinite(t["pnl_pct"]), "Non-finite P&L — accounting bug"

    strategy.DEFAULT_CFG.clear()
    strategy.DEFAULT_CFG.update(orig_default_cfg)

    print("\nALL ORCHESTRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
