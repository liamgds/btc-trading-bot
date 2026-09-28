"""
config.py — Centralized configuration.

Strategy parameters (EMA/RSI/ADX thresholds, sizing, stops, targets) live
in strategy.DEFAULT_CFG, NOT here — that keeps one source of truth for
anything that affects trading decisions. This file only holds settings
about HOW the bot runs: which exchange account, which symbol, how often
to poll, logging, and backtest I/O.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized runtime configuration, loaded from .env"""

    # ==================== EXCHANGE API ====================
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # ==================== TRADING SETTINGS ====================
    # 'paper'   -> simulated fills using live public market prices, no real orders,
    #              no API keys needed
    # 'demo'    -> real orders against Binance's Spot Demo Mode (simulated funds,
    #              realistic live market data/rules)
    # 'testnet' -> real orders against Binance's Spot Testnet (simulated funds,
    #              independent/synthetic market data)
    # 'live'    -> DISABLED in this build. This project is demo-only — see
    #              Config.validate(), which refuses to start if this is set.
    TRADING_MODE = os.getenv("TRADING_MODE", "paper")
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "15m")

    # ==================== OPERATIONAL ====================
    CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "5"))
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    MAX_SLIPPAGE_PERCENT = float(os.getenv("MAX_SLIPPAGE_PERCENT", "0.5"))

    # ==================== BACKTEST ====================
    BACKTEST_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000"))
    BACKTEST_DATA_FILE = os.getenv("BACKTEST_DATA_FILE", "")

    # ==================== LOGGING ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    VALID_TIMEFRAMES = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M",
    }

    @classmethod
    def _sanitize_string(cls, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\-_.]", "", value)

    @classmethod
    def _mask_sensitive(cls, value: str, visible_chars: int = 4) -> str:
        if not value or len(value) <= visible_chars:
            return "***"
        return value[:visible_chars] + "***" + value[-visible_chars:]

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration. Raises ValueError listing every problem found."""
        errors = []

        # This build is scoped to demo-only trading. 'live' is refused outright,
        # with its own message, rather than folded into the generic invalid-value
        # error below — this should read as a deliberate restriction, not a typo.
        if cls.TRADING_MODE == "live":
            raise ValueError(
                "TRADING_MODE=live is disabled in this build. This project is "
                "scoped to demo-only trading — use 'paper', 'demo', or 'testnet' "
                "instead. No code path in this codebase can place a real-money order."
            )

        if cls.TRADING_MODE not in ("testnet", "paper", "demo"):
            errors.append(f"TRADING_MODE must be 'paper', 'demo', or 'testnet', got '{cls.TRADING_MODE}'")

        if cls.TRADING_MODE in ("demo", "testnet"):
            if not cls.BINANCE_API_KEY:
                errors.append(f"BINANCE_API_KEY required for '{cls.TRADING_MODE}' mode")
            if not cls.BINANCE_API_SECRET:
                errors.append(f"BINANCE_API_SECRET required for '{cls.TRADING_MODE}' mode")

        cls.SYMBOL = cls._sanitize_string(cls.SYMBOL).upper()
        if not cls.SYMBOL:
            errors.append("SYMBOL cannot be empty")

        if cls.TIMEFRAME not in cls.VALID_TIMEFRAMES:
            errors.append(f"TIMEFRAME '{cls.TIMEFRAME}' invalid. Valid: {sorted(cls.VALID_TIMEFRAMES)}")

        if cls.CHECK_INTERVAL_SECONDS < 5:
            errors.append(f"CHECK_INTERVAL_SECONDS too low, got {cls.CHECK_INTERVAL_SECONDS}")

        if cls.BACKTEST_INITIAL_CAPITAL <= 0:
            errors.append(f"BACKTEST_INITIAL_CAPITAL must be positive, got {cls.BACKTEST_INITIAL_CAPITAL}")

        if cls.LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"LOG_LEVEL invalid: '{cls.LOG_LEVEL}'")

        if errors:
            msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

        return True

    @classmethod
    def mode_banner(cls) -> str:
        """A short, unmissable banner stating exactly which mode is active and
        what that means for real money, shown once at startup before anything
        else runs."""
        labels = {
            "paper": ("PAPER", "Fully simulated. No exchange orders are ever submitted."),
            "demo": ("DEMO", "Binance Spot Demo Mode — simulated funds, real orders submitted to Binance's demo endpoint only."),
            "testnet": ("TESTNET", "Binance Spot Testnet — simulated funds, real orders submitted to Binance's testnet endpoint only."),
        }
        label, detail = labels.get(cls.TRADING_MODE, (cls.TRADING_MODE.upper(), "Unrecognized mode."))
        line = f"  MODE: {label} — {detail}  "
        border = "*" * len(line)
        return f"\n{border}\n{line}\n{border}\n"

    @classmethod
    def display(cls) -> str:
        api_key_display = cls._mask_sensitive(cls.BINANCE_API_KEY) if cls.BINANCE_API_KEY else "Not Set"
        return f"""
========================================
TRADING BOT CONFIGURATION
========================================
Trading Mode:     {cls.TRADING_MODE.upper()}
Symbol:           {cls.SYMBOL}
Timeframe:        {cls.TIMEFRAME}
Check Interval:   {cls.CHECK_INTERVAL_SECONDS}s
Max Retries:      {cls.MAX_RETRIES}
Circuit Breaker:  {cls.CIRCUIT_BREAKER_THRESHOLD} failures
Binance API Key:  {api_key_display}
Log Level:        {cls.LOG_LEVEL}
========================================
        """
