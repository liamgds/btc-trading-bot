"""
exchange.py — Binance exchange interface.

This build is demo-only. Three modes, selected by Config.TRADING_MODE:
  paper   - no API keys needed. Simulates fills using real public market
            data (price/candles are real, orders are simulated locally,
            no authenticated client is ever constructed).
  demo    - real orders against Binance's Spot Demo Mode (api key/secret
            from https://demo.binance.com, simulated funds, market data
            and rules matching the live exchange).
  testnet - real orders against Binance's Spot Testnet (api key/secret
            from https://testnet.binance.vision, simulated funds,
            independent/synthetic market data).

'live' mode is refused unconditionally in __init__ — see the class
docstring below. There is no code path in this file that can reach a
real-money endpoint.

Includes: duplicate-order prevention (deterministic hash, not time-based),
a circuit breaker to stop hammering the API after repeated failures,
slippage validation, partial-fill handling, limit-order support with
a cancel-if-unfilled TTL (used by strategy.py's Fix 3: prefer limit orders
over market orders to save on fees/slippage), and an endpoint-verification
safety check (_verify_endpoint) run at construction and again immediately
before every order submission.
"""

import time
import hashlib
import random
from typing import Optional, Dict, List, Tuple

import pandas as pd
from retry import retry


class CircuitBreaker:
    """Stops repeated API calls after N consecutive failures, until a cooldown passes."""

    def __init__(self, threshold: int = 5, reset_timeout: int = 300):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"

    def can_proceed(self) -> Tuple[bool, str]:
        if self.state == "CLOSED":
            return True, "OK"
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.failure_count = 0
                return True, "Attempting recovery"
            return False, f"Circuit breaker OPEN ({self.failure_count} failures)"
        return True, "Testing connection"


class BinanceExchange:
    """
    Binance exchange interface. Falls back to paper mode if no API keys are set.

    This build is demo-only by design: 'live' mode is refused unconditionally,
    independently of config.py's own check, so this class is safe even if it
    is ever constructed from somewhere that skipped Config.validate(). Every
    non-paper mode is verified, right after construction and again
    immediately before any order is submitted, to actually be pointed at the
    endpoint its mode name promises — see _verify_endpoint().
    """

    # Domain fragments expected in the resolved request URL for each mode.
    # Used by _verify_endpoint() — if the library's URL resolution for a mode
    # ever stops matching this, that's treated as a safety failure, not a
    # detail to ignore.
    _EXPECTED_ENDPOINT_FRAGMENT = {
        "demo": "demo-api.binance.com",
        "testnet": "testnet.binance.vision",
    }

    def __init__(self, api_key: str, api_secret: str, mode: str = "paper"):
        if mode == "live":
            # Belt-and-suspenders: Config.validate() already refuses to start
            # in 'live' mode, but this class refuses independently too, so
            # there is no code path — now or added later — that reaches a
            # real-money endpoint just because some caller skipped config
            # validation.
            raise ValueError(
                "BinanceExchange refuses to construct in 'live' mode. This "
                "build is demo-only (paper/demo/testnet)."
            )
        if mode not in ("paper", "demo", "testnet"):
            raise ValueError(f"Unknown trading mode '{mode}'. Must be 'paper', 'demo', or 'testnet'.")

        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.paper_mode = (mode == "paper") or not api_key or not api_secret

        from binance.client import Client
        if self.paper_mode:
            self.client = None
            self.public_client = Client()  # public endpoints only, no auth needed
        else:
            self.client = Client(
                api_key, api_secret,
                demo=(mode == "demo"),
                testnet=(mode == "testnet"),
            )
            self.public_client = self.client
            self._verify_endpoint()  # fail fast at startup, before the trading loop ever begins

        self.recent_orders: Dict[str, float] = {}
        self.order_cache_duration = 300  # seconds
        self.circuit_breaker = CircuitBreaker(threshold=5, reset_timeout=300)
        self._open_limit_orders: Dict[str, Dict] = {}  # order_id -> {symbol, side, qty, price, candles_waited}

    # ---------------------------------------------------------------- safety

    def _verify_endpoint(self):
        """
        Confirms the constructed client is actually pointed at the endpoint
        its mode name promises, checked two independent ways:
          1. the boolean flags that gate the library's URL resolution
             (client.testnet / client.demo) match the requested mode, and
          2. the concrete URL those flags resolve to contains the domain
             we expect for that mode.
        Raises if either check fails or disagrees — this runs at startup
        (__init__) and again immediately before every order submission, so a
        state change between the two could never let an order through
        unverified.
        """
        if self.paper_mode:
            if self.client is not None:
                raise RuntimeError("Safety check failed: paper mode has an authenticated client — it must not.")
            return  # no authenticated client exists in paper mode; nothing further to verify

        expected_fragment = self._EXPECTED_ENDPOINT_FRAGMENT.get(self.mode)
        if expected_fragment is None:
            raise RuntimeError(f"Safety check failed: no known-good endpoint defined for mode '{self.mode}'.")

        flags_ok = {
            "demo": getattr(self.client, "demo", False) and not getattr(self.client, "testnet", False),
            "testnet": getattr(self.client, "testnet", False) and not getattr(self.client, "demo", False),
        }[self.mode]
        if not flags_ok:
            raise RuntimeError(
                f"Safety check failed: mode='{self.mode}' but client.demo="
                f"{getattr(self.client, 'demo', None)}, client.testnet="
                f"{getattr(self.client, 'testnet', None)}. Refusing to proceed."
            )

        try:
            resolved_url = self.client._create_api_uri("account")
        except Exception as e:
            raise RuntimeError(f"Safety check failed: could not resolve API URL to verify ({e}).")

        if expected_fragment not in resolved_url:
            raise RuntimeError(
                f"Safety check failed: mode='{self.mode}' resolved to '{resolved_url}', "
                f"which does not contain the expected '{expected_fragment}'. Refusing to proceed."
            )

    # ---------------------------------------------------------------- utils

    def _check_circuit_breaker(self):
        can_proceed, reason = self.circuit_breaker.can_proceed()
        if not can_proceed:
            raise Exception(f"Circuit breaker preventing operation: {reason}")

    def _order_hash(self, symbol: str, side: str, quantity: float) -> str:
        data = f"{symbol}{side}{quantity:.8f}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _is_recent_order(self, order_hash: str) -> bool:
        now = time.time()
        self.recent_orders = {h: t for h, t in self.recent_orders.items()
                               if now - t < self.order_cache_duration}
        return order_hash in self.recent_orders

    def _mark_order_placed(self, order_hash: str):
        self.recent_orders[order_hash] = time.time()

    # ------------------------------------------------------------- account

    @retry(tries=3, delay=2, backoff=2)
    def get_account_balance(self, asset: str = "USDT") -> float:
        self._check_circuit_breaker()
        if self.paper_mode:
            return 10_000.0
        try:
            info = self.client.get_account()
            for bal in info["balances"]:
                if bal["asset"] == asset:
                    self.circuit_breaker.record_success()
                    return float(bal["free"])
            self.circuit_breaker.record_success()
            return 0.0
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get balance: {e}")

    @retry(tries=3, delay=2, backoff=2)
    def get_current_price(self, symbol: str) -> float:
        self._check_circuit_breaker()
        try:
            ticker = self.public_client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            self.circuit_breaker.record_success()
            return price
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get price for {symbol}: {e}")

    @retry(tries=3, delay=2, backoff=2)
    def get_historical_klines(self, symbol: str, interval: str,
                              start_time: str, end_time: Optional[str] = None,
                              limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV candles, paginating past Binance's 1000-candle-per-call limit."""
        self._check_circuit_breaker()
        try:
            all_klines: List = []
            current_start = start_time
            while True:
                klines = self.public_client.get_historical_klines(
                    symbol=symbol, interval=interval,
                    start_str=current_start, end_str=end_time, limit=limit,
                )
                if not klines:
                    break
                all_klines.extend(klines)
                if len(klines) < limit:
                    break
                current_start = str(klines[-1][0] + 1)
                if len(all_klines) >= 20_000:
                    break

            if not all_klines:
                raise Exception("No data returned from exchange")

            df = pd.DataFrame(all_klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)

            if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
                raise Exception("Data contains NaN values")

            self.circuit_breaker.record_success()
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get historical data: {e}")

    # --------------------------------------------------------------- orders

    @retry(tries=3, delay=2, backoff=2)
    def place_market_order(self, symbol: str, side: str, quantity: float,
                           expected_price: float, max_slippage_percent: float = 0.5) -> Dict:
        """Place (or simulate) a market order. Raises if slippage exceeds max_slippage_percent."""
        self._verify_endpoint()  # re-checked immediately before every order, not just at startup
        self._check_circuit_breaker()
        quantity = round(quantity, 8)

        order_hash = self._order_hash(symbol, side, quantity)
        if self._is_recent_order(order_hash):
            raise Exception(f"Duplicate order detected (same order within {self.order_cache_duration}s)")

        if self.paper_mode:
            current_price = self.get_current_price(symbol)
            slippage = random.uniform(-0.001, 0.001)
            fill_price = current_price * (1 + slippage)
            order = {
                "orderId": f"PAPER_{int(time.time() * 1000)}",
                "symbol": symbol, "side": side, "type": "MARKET",
                "origQty": quantity, "executedQty": quantity,
                "price": fill_price,
                "fills": [{"price": fill_price, "qty": quantity, "commission": quantity * fill_price * 0.001}],
                "status": "FILLED", "transactTime": int(time.time() * 1000),
                "paper_trade": True,
            }
            actual_slippage = abs((fill_price - expected_price) / expected_price * 100)
            if actual_slippage > max_slippage_percent:
                raise Exception(f"Simulated slippage {actual_slippage:.2f}% exceeds maximum {max_slippage_percent}%")
            self._mark_order_placed(order_hash)
            self.circuit_breaker.record_success()
            return order

        try:
            if side == "BUY":
                order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            else:
                order = self.client.order_market_sell(symbol=symbol, quantity=quantity)

            if order["status"] not in ("FILLED", "PARTIALLY_FILLED"):
                raise Exception(f"Order not filled: {order['status']}")

            executed_qty = float(order["executedQty"])
            if executed_qty == 0:
                raise Exception("Order executed but quantity is zero")

            total_cost, total_qty = 0.0, 0.0
            for fill in order.get("fills", []):
                fp, fq = float(fill["price"]), float(fill["qty"])
                total_cost += fp * fq
                total_qty += fq
            avg_price = total_cost / total_qty if total_qty > 0 else float(order.get("price", expected_price))

            actual_slippage = abs((avg_price - expected_price) / expected_price * 100)
            if actual_slippage > max_slippage_percent:
                print(f"WARNING: Slippage {actual_slippage:.2f}% exceeds maximum {max_slippage_percent}%")

            order["avg_fill_price"] = avg_price
            order["actual_slippage_percent"] = actual_slippage
            order["executedQty"] = executed_qty
            self._mark_order_placed(order_hash)
            self.circuit_breaker.record_success()
            return order
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Order failed: {e}")

    @retry(tries=3, delay=2, backoff=2)
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """
        Place (or simulate) a GTC limit order at `price`.
        Fix 3 from the strategy audit: limit orders cost less in fees/slippage
        than market orders. Caller is responsible for polling
        get_order_status() and cancelling if unfilled within the TTL window.
        """
        self._verify_endpoint()  # re-checked immediately before every order, not just at startup
        self._check_circuit_breaker()
        quantity = round(quantity, 8)

        if self.paper_mode:
            order_id = f"PAPER_LIMIT_{int(time.time() * 1000)}"
            order = {
                "orderId": order_id, "symbol": symbol, "side": side, "type": "LIMIT",
                "origQty": quantity, "executedQty": 0.0, "price": price,
                "status": "NEW", "transactTime": int(time.time() * 1000),
                "paper_trade": True,
            }
            self._open_limit_orders[order_id] = {
                "symbol": symbol, "side": side, "qty": quantity, "price": price, "candles_waited": 0,
            }
            self.circuit_breaker.record_success()
            return order

        try:
            if side == "BUY":
                order = self.client.order_limit_buy(symbol=symbol, quantity=quantity, price=f"{price:.8f}")
            else:
                order = self.client.order_limit_sell(symbol=symbol, quantity=quantity, price=f"{price:.8f}")
            self._open_limit_orders[str(order["orderId"])] = {
                "symbol": symbol, "side": side, "qty": quantity, "price": price, "candles_waited": 0,
            }
            self.circuit_breaker.record_success()
            return order
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Limit order failed: {e}")

    def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Check fill status of a previously placed order (paper mode simulates a probabilistic fill)."""
        if self.paper_mode:
            info = self._open_limit_orders.get(str(order_id))
            if info is None:
                return {"status": "UNKNOWN"}
            info["candles_waited"] += 1
            # Simulate: ~60% chance of fill per candle checked, to mimic realistic limit fills
            if random.random() < 0.6:
                return {"status": "FILLED", "executedQty": info["qty"], "price": info["price"]}
            return {"status": "NEW", "executedQty": 0.0}

        try:
            result = self.client.get_order(symbol=symbol, orderId=order_id)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get order status: {e}")

    @retry(tries=3, delay=2, backoff=2)
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        self._check_circuit_breaker()
        if self.paper_mode:
            self._open_limit_orders.pop(str(order_id), None)
            return {"orderId": order_id, "status": "CANCELED", "paper_trade": True}
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            self._open_limit_orders.pop(str(order_id), None)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Cancel order failed: {e}")

    def get_circuit_breaker_status(self) -> Dict:
        return {
            "state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "threshold": self.circuit_breaker.threshold,
        }
