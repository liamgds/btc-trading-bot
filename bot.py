"""
bot.py — Main trading bot orchestration loop.

This is the piece that never got finished in earlier phases: the strategy
logic (strategy.py, Phase 8) kept improving, but nothing wired it up to an
actual exchange connection. This module is that missing wiring.

Design principle: this file contains NO trading-decision logic of its own.
Every decision (entry filter, position size, stop-loss level, exit trigger)
comes from strategy.py's pure functions. bot.py only:
  1. fetches market data from exchange.py
  2. asks strategy.py what to do
  3. tells exchange.py to place/cancel orders
  4. tells risk_manager.py to update the ledger
  5. logs everything

Run modes (see config.py):
  paper   - simulated fills against real market data, no API keys needed
  testnet - real orders against Binance Spot Testnet (fake funds)
  live    - real orders against a real account (REAL MONEY — off by default)
"""

import time
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from logger import get_logger
from exchange import BinanceExchange
from risk_manager import RiskManager
import strategy


class TradingBot:
    """Coordinates strategy, exchange, and risk management for live/paper trading."""

    def __init__(self, exchange: Optional[BinanceExchange] = None):
        """
        `exchange` can be injected for testing (e.g. a fake exchange fed by
        historical/synthetic data with no network access). Production use
        (main.py) leaves this as None and gets a real BinanceExchange.
        """
        Config.validate()

        self.logger = get_logger(Config.LOG_LEVEL, Config.LOG_FILE,
                                 Config.LOG_MAX_BYTES, Config.LOG_BACKUP_COUNT)
        self.logger.info(Config.mode_banner())  # first thing logged — unmissable, and recorded to file
        self.logger.info("Initializing Trading Bot...")

        self.cfg = dict(strategy.DEFAULT_CFG)  # single source of truth for strategy params
        self.exchange = exchange or BinanceExchange(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET,
            mode=Config.TRADING_MODE,
        )

        try:
            initial_balance = self.exchange.get_account_balance("USDT")
        except Exception as e:
            self.logger.warning(f"Could not get account balance, using default: {e}")
            initial_balance = Config.BACKTEST_INITIAL_CAPITAL

        self.risk_manager = RiskManager(initial_capital=initial_balance, cfg=self.cfg)
        self._vov_halted = False
        self._pending_entry_order: Optional[dict] = None  # tracks an unfilled limit BUY order

        self.running = False
        self.logger.info(Config.display())

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.logger.info("Shutdown signal received. Stopping bot...")
        self.stop()
        sys.exit(0)

    # ------------------------------------------------------------------ loop

    def start(self):
        self.logger.info("=" * 50)
        self.logger.info(f"Starting {Config.TRADING_MODE.upper()} Trading Bot")
        self.logger.info(f"Symbol: {Config.SYMBOL} | Timeframe: {Config.TIMEFRAME}")
        self.logger.info("=" * 50)
        self.running = True

        try:
            while self.running:
                try:
                    self.run_once()
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def run_once(self):
        """Execute a single trading cycle. Exposed separately so it can be tested in isolation."""
        self.logger.debug(f"Running trading cycle at {datetime.now(timezone.utc)}")

        df_raw = self._get_market_data()
        d = strategy.build_indicators(df_raw, self.cfg)
        if d.empty:
            self.logger.warning("No indicator rows after warmup — not enough candle history yet")
            return
        row = d.iloc[-1]
        today = row["timestamp"].date()

        # Update VoV crash-filter halt state (Fix 2)
        vov = float(row["vov"]) if not np.isnan(row["vov"]) else 1.0
        if vov > self.cfg["vov_halt_threshold"]:
            self._vov_halted = True
        elif vov < self.cfg["vov_resume_threshold"]:
            self._vov_halted = False

        if self.risk_manager.has_open_position():
            self._manage_open_position(row, today)
        else:
            self._pending_entry_order = None
            can_open, why = self.risk_manager.can_open_new_position(today)
            if not can_open:
                self.logger.debug(f"Not opening new position: {why}")
                return
            ok, reason = strategy.check_entry(row, self.cfg, self._vov_halted)
            indicators = {
                "Price": float(row["close"]), "ADX": float(row["adx"]),
                "RSI": float(row["rsi"]), "VoV": round(vov, 3),
                "VoV_Halt": self._vov_halted,
            }
            indicators["Reason" if ok else "Blocked"] = reason
            self.logger.log_signal("BUY" if ok else "HOLD", Config.SYMBOL, float(row["close"]), indicators)
            if ok:
                self._execute_entry(row, d, today)

    # -------------------------------------------------------------- position management

    def _manage_open_position(self, row: pd.Series, today):
        position = self.risk_manager.position
        fee_maker = self.cfg["fee_maker"] / 100
        fee_slip = self.cfg["slippage_normal"] / 100

        position, partial_exits, full_exit = strategy.evaluate_position_candle(
            row, position, self.cfg, fee_maker
        )

        for pe in partial_exits:
            self._execute_exit_order(pe["qty"], pe["price"], f"Partial-{pe['label']}")
            pnl = self.risk_manager.record_partial_exit(pe, today)
            self.logger.log_trade("SELL", Config.SYMBOL, pe["qty"], pe["price"],
                                  pe["qty"] * pe["price"], reason=f"Tranche {pe['label']}", pnl=pnl)

        if full_exit is not None:
            qty_remaining = position["qty"]
            self._execute_exit_order(qty_remaining, full_exit["price"], full_exit["reason"])
            trade = self.risk_manager.record_full_exit(
                full_exit["price"], full_exit["reason"], row["timestamp"], today,
                fee_maker, fee_slip,
            )
            self.logger.log_trade("SELL", Config.SYMBOL, qty_remaining, full_exit["price"],
                                  qty_remaining * full_exit["price"], reason=full_exit["reason"],
                                  pnl=trade["pnl"], pnl_percent=trade["pnl_pct"])
            self.logger.info(self.risk_manager.display_statistics())
        else:
            self.risk_manager.update_position_state(position)

    def _execute_exit_order(self, qty: float, expected_price: float, reason: str):
        """
        Place the actual sell order for an exit. Simplification for this
        polling-interval bot: since evaluate_position_candle already tells us
        a level was crossed on the last CLOSED candle, we execute at market
        with a slippage check against that level, rather than trying to have
        placed a resting limit order at the exact tick it triggered. This is
        a reasonable approximation at a 15m polling interval, but is not
        tick-perfect execution.
        """
        try:
            self.exchange.place_market_order(
                symbol=Config.SYMBOL, side="SELL", quantity=qty,
                expected_price=expected_price, max_slippage_percent=Config.MAX_SLIPPAGE_PERCENT,
            )
        except Exception as e:
            self.logger.error(f"Exit order failed ({reason}): {e}")

    def _execute_entry(self, row: pd.Series, d: pd.DataFrame, today):
        entry = float(row["close"])
        idx = len(d) - 1
        sl = strategy.compute_stop_loss(d, idx, entry, self.cfg)
        conf = strategy.confidence_score(row, self.cfg)

        equity, recent_returns, consec_losses, dd_pct, day_pnl = self.risk_manager.sizing_inputs(today)
        qty, sizing = strategy.compute_size(
            equity, entry, sl, conf, recent_returns, consec_losses, dd_pct, day_pnl, self.cfg
        )

        if qty <= 0 or "blocked" in sizing:
            self.logger.warning(f"Entry signal fired but sizing blocked it: {sizing}")
            return

        qty = round(qty, 6)
        self.logger.info(f"Attempting BUY {qty} {Config.SYMBOL} @ ${entry:.2f} | sizing={sizing}")

        try:
            if self.cfg["use_limit_orders"]:
                order = self.exchange.place_limit_order(Config.SYMBOL, "BUY", qty, entry)
                status = self.exchange.get_order_status(Config.SYMBOL, order["orderId"])
                if status["status"] != "FILLED":
                    self.logger.info(f"Limit BUY order {order['orderId']} not yet filled, will check next cycle")
                    self._pending_entry_order = {"order_id": order["orderId"], "candles_waited": 0}
                    return
                fill_price = float(status.get("price") or 0.0)
                if fill_price <= 0:
                    # Defensive: a filled limit order should execute at/better than
                    # the limit price. If the exchange response is missing/zero
                    # (glitch, unexpected API shape), fall back to the intended
                    # limit price rather than silently building a position on a
                    # $0 entry, which would corrupt every downstream calculation.
                    self.logger.warning(
                        f"Exchange returned invalid fill price ({status.get('price')}) "
                        f"for order {order['orderId']}; using limit price ${entry:.2f} instead"
                    )
                    fill_price = entry
            else:
                order = self.exchange.place_market_order(
                    Config.SYMBOL, "BUY", qty, expected_price=entry,
                    max_slippage_percent=Config.MAX_SLIPPAGE_PERCENT,
                )
                fill_price = float(order.get("avg_fill_price") or order.get("price") or 0.0)
                if fill_price <= 0:
                    self.logger.warning(
                        f"Exchange returned invalid fill price for market order "
                        f"{order.get('orderId')}; using expected price ${entry:.2f} instead"
                    )
                    fill_price = entry
        except Exception as e:
            self.logger.error(f"Buy order failed: {e}")
            return

        atr_v = float(row["atr"])
        position = {
            "entry": fill_price, "qty": qty, "qty_original": qty,
            "sl": sl, "tp_t1": fill_price + self.cfg["tp_t1_atr_mult"] * atr_v,
            "trail_stop": fill_price - 2.0 * atr_v, "atr_entry": atr_v,
            "in_profit": False, "t1_done": False, "t2_done": False,
            "candles_held": 0, "partial_pnl": 0.0,
            "entry_time": row["timestamp"],
            "vov_at_entry": float(row["vov"]) if not np.isnan(row["vov"]) else 1.0,
            "confidence": conf,
        }
        self.risk_manager.open_position(position)
        self.logger.log_trade("BUY", Config.SYMBOL, qty, fill_price, qty * fill_price,
                              reason="Entry: all filters passed")
        self.logger.info(f"Position opened: SL=${sl:.2f}, T1=${position['tp_t1']:.2f}")

    # ------------------------------------------------------------------ data

    def _get_market_data(self) -> pd.DataFrame:
        lookback_candles = max(self.cfg["ema_trend"], self.cfg["atr_regime_period"]) * 2 + 100
        tf_minutes = self._timeframe_to_minutes(Config.TIMEFRAME)
        start_time = datetime.now(timezone.utc) - timedelta(minutes=tf_minutes * lookback_candles)
        return self.exchange.get_historical_klines(
            symbol=Config.SYMBOL, interval=Config.TIMEFRAME,
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            limit=1000,
        )

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return {"m": value, "h": value * 60, "d": value * 1440, "w": value * 10080}.get(unit, 15)

    # ------------------------------------------------------------------ lifecycle

    def stop(self):
        self.logger.info("Stopping trading bot...")
        self.running = False
        self.logger.info(self.risk_manager.display_statistics())
        self.logger.info("Bot stopped successfully")

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "mode": Config.TRADING_MODE,
            "symbol": Config.SYMBOL,
            "has_position": self.risk_manager.has_open_position(),
            "equity": self.risk_manager.equity,
            "trades": len(self.risk_manager.trades),
        }
