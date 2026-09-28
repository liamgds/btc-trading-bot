"""
logger.py — Console + rotating file logging, trade CSV log, and API-key masking.

Fix applied here vs. the earlier draft: log_signal() used to assume every
indicator value was a plain int/float and format it with `:.2f`. strategy.py's
FinalStrategy returns booleans (VoV_Halt) and strings (Reason/Blocked) in the
same indicators dict, which crashed that formatter (`bool.__format__` doesn't
accept '.2f'). Now every value type is handled explicitly.
"""

import logging
import os
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


class SensitiveDataFilter(logging.Filter):
    """Masks API keys/secrets/tokens/passwords before they hit any log output."""

    SENSITIVE_PATTERNS = [
        (r'([\"\']?api[_-]?key[\"\']?\s*[:=]\s*[\"\']?)([a-zA-Z0-9]{20,})', r'\1***MASKED***'),
        (r'([\"\']?secret[\"\']?\s*[:=]\s*[\"\']?)([a-zA-Z0-9]{20,})', r'\1***MASKED***'),
        (r'([\"\']?token[\"\']?\s*[:=]\s*[\"\']?)([a-zA-Z0-9]{20,})', r'\1***MASKED***'),
        (r'([\"\']?password[\"\']?\s*[:=]\s*[\"\']?)([^\s\"\']+)', r'\1***MASKED***'),
        (r'\b([a-zA-Z0-9]{64})\b', r'***MASKED_API_KEY***'),
    ]

    def filter(self, record):
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = re.sub(pattern, replacement, record.msg, flags=re.IGNORECASE)
        return True


def _format_value(v) -> str:
    """Format any indicator value type safely for log output."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


class BotLogger:
    """Console + rotating file logger, plus a structured trades.csv log."""

    def __init__(self, name: str = "TradingBot", log_file: str = "logs/trading_bot.log",
                 log_level: str = "INFO", max_bytes: int = 10_485_760, backup_count: int = 5):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        if self.logger.handlers:
            return  # already configured (singleton reuse)

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sensitive_filter = SensitiveDataFilter()

        if _HAS_COLORLOG:
            console_handler = colorlog.StreamHandler()
            console_format = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow",
                            "ERROR": "red", "CRITICAL": "red,bg_white"},
            )
        else:
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_format)
        console_handler.addFilter(sensitive_filter)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes,
                                           backupCount=backup_count, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        file_handler.addFilter(sensitive_filter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.trade_log_file = os.path.join(log_dir or "logs", "trades.csv")
        self._init_trade_log()

    def _init_trade_log(self):
        if not os.path.exists(self.trade_log_file):
            log_dir = os.path.dirname(self.trade_log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(self.trade_log_file, "w", encoding="utf-8") as f:
                f.write("timestamp,action,symbol,quantity,price,total_value,reason,pnl,pnl_percent\n")

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

    def log_trade(self, action: str, symbol: str, quantity: float, price: float,
                  total_value: float, reason: str = "", pnl: float = 0, pnl_percent: float = 0):
        message = f"TRADE EXECUTED: {action} {quantity:.8f} {symbol} @ ${price:.2f} (${total_value:.2f})"
        if reason:
            message += f" | Reason: {reason}"
        if pnl != 0:
            message += f" | P&L: ${pnl:.2f} ({pnl_percent:.2f}%)"
        self.info(message)

        try:
            with open(self.trade_log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                reason_clean = reason.replace(",", ";")
                f.write(f"{timestamp},{action},{symbol},{quantity:.8f},{price:.8f},"
                       f"{total_value:.2f},{reason_clean},{pnl:.2f},{pnl_percent:.4f}\n")
        except Exception as e:
            self.error(f"Failed to write to trade log: {e}")

    def log_signal(self, signal_type: str, symbol: str, price: float, indicators: dict):
        indicator_str = ", ".join(f"{k}={_format_value(v)}" for k, v in indicators.items())
        self.info(f"SIGNAL: {signal_type} | {symbol} @ ${price:.2f} | {indicator_str}")

    def log_error_with_context(self, error: Exception, context: str):
        self.error(f"{context} - Error: {type(error).__name__}: {error}")


_logger_instance = None


def get_logger(log_level: str = "INFO", log_file: str = "logs/trading_bot.log",
              max_bytes: int = 10_485_760, backup_count: int = 5) -> BotLogger:
    """Get or create the singleton bot logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = BotLogger(log_level=log_level, log_file=log_file,
                                     max_bytes=max_bytes, backup_count=backup_count)
    return _logger_instance
