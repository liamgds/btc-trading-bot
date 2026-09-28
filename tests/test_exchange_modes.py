"""
Tests for the paper/demo/testnet mode routing added to exchange.py and
config.py, and for the safety guarantees around them:
  - demo mode always resolves to Binance's demo endpoint
  - testnet mode always resolves to Binance's testnet endpoint
  - paper mode never constructs an authenticated client, so it structurally
    cannot submit an exchange order
  - TRADING_MODE=live is refused at both the config layer and the exchange
    layer independently
  - the endpoint-verification safety check genuinely catches a mismatch,
    not just always passing

No real network access is used. Client construction makes one network call
(a startup ping) in the real python-binance library, which this sandbox
cannot reach — so these tests patch only that one call, and otherwise run
the REAL library's own URL-resolution code, not a hand-written substitute.
This matters: it proves demo=True/testnet=True actually resolve to the
right domains in the real dependency, not just in a mock that assumes they do.

Run from the project root:
    python tests/test_exchange_modes.py
"""
import sys
import os
from unittest.mock import patch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import config
import exchange as exchange_module
from binance.client import Client as RealBinanceClient


def _no_network_client(mode_kwargs_check=None):
    """Context manager-style helper: patches only Client.ping (the one call
    that needs network) so construction otherwise runs the real library's
    own logic — everything else (URL resolution, attribute storage) is real."""
    return patch.object(RealBinanceClient, "ping", lambda self: None)


def test_paper_mode_has_no_authenticated_client():
    ex = exchange_module.BinanceExchange("", "", mode="paper")
    assert ex.client is None, "paper mode must never construct an authenticated client"
    assert ex.paper_mode is True
    # Confirm order placement still works (simulated) without ever touching self.client.
    # get_current_price() is mocked here purely because it calls Binance's public
    # ticker endpoint for real market data, which needs network this sandbox
    # doesn't have — everything else in the simulated-order path below is real.
    with patch.object(ex, "get_current_price", return_value=50000.0):
        order = ex.place_market_order("BTCUSDT", "BUY", 0.01, expected_price=50000, max_slippage_percent=5)
    assert order["paper_trade"] is True
    print("PASS: paper mode has no authenticated client, and simulated orders work")


def test_demo_mode_resolves_to_demo_endpoint():
    with _no_network_client():
        ex = exchange_module.BinanceExchange("fakekey", "fakesecret", mode="demo")
    assert ex.client.demo is True, "demo mode must set client.demo=True"
    assert ex.client.testnet is False, "demo mode must not also set testnet=True"
    resolved = ex.client._create_api_uri("account")
    assert "demo-api.binance.com" in resolved, f"demo mode resolved to unexpected URL: {resolved}"
    print("PASS: demo mode resolves to demo-api.binance.com (verified against the real library)")


def test_testnet_mode_resolves_to_testnet_endpoint():
    with _no_network_client():
        ex = exchange_module.BinanceExchange("fakekey", "fakesecret", mode="testnet")
    assert ex.client.testnet is True, "testnet mode must set client.testnet=True"
    assert ex.client.demo is False, "testnet mode must not also set demo=True"
    resolved = ex.client._create_api_uri("account")
    assert "testnet.binance.vision" in resolved, f"testnet mode resolved to unexpected URL: {resolved}"
    print("PASS: testnet mode resolves to testnet.binance.vision (verified against the real library)")


def test_demo_mode_without_keys_falls_back_to_paper():
    ex = exchange_module.BinanceExchange("", "", mode="demo")
    assert ex.paper_mode is True
    assert ex.client is None
    print("PASS: demo mode with missing keys safely falls back to paper (no authenticated client)")


def test_live_mode_refused_by_exchange_layer():
    with _no_network_client():
        try:
            exchange_module.BinanceExchange("fakekey", "fakesecret", mode="live")
            raise AssertionError("live mode should have raised ValueError")
        except ValueError as e:
            assert "live" in str(e).lower()
    print("PASS: exchange.py refuses to construct in live mode")


def test_live_mode_refused_by_config_layer():
    original = config.Config.TRADING_MODE
    try:
        config.Config.TRADING_MODE = "live"
        try:
            config.Config.validate()
            raise AssertionError("Config.validate() should have raised for TRADING_MODE=live")
        except ValueError as e:
            assert "demo-only" in str(e) or "disabled" in str(e).lower()
    finally:
        config.Config.TRADING_MODE = original
    print("PASS: config.py refuses TRADING_MODE=live with a clear message")


def test_demo_mode_requires_api_keys_at_config_level():
    original_mode = config.Config.TRADING_MODE
    original_key = config.Config.BINANCE_API_KEY
    original_secret = config.Config.BINANCE_API_SECRET
    try:
        config.Config.TRADING_MODE = "demo"
        config.Config.BINANCE_API_KEY = ""
        config.Config.BINANCE_API_SECRET = ""
        try:
            config.Config.validate()
            raise AssertionError("Config.validate() should require keys for demo mode")
        except ValueError as e:
            assert "BINANCE_API_KEY" in str(e)

        config.Config.BINANCE_API_KEY = "somekey"
        config.Config.BINANCE_API_SECRET = "somesecret"
        assert config.Config.validate() is True
    finally:
        config.Config.TRADING_MODE = original_mode
        config.Config.BINANCE_API_KEY = original_key
        config.Config.BINANCE_API_SECRET = original_secret
    print("PASS: config.py requires API keys for demo mode, and validates cleanly once provided")


def test_endpoint_safety_check_catches_url_mismatch():
    """Proves _verify_endpoint() would actually catch a broken/mismatched
    client, not just always pass — using a deliberately wrong fake."""
    class BrokenClient:
        def __init__(self, api_key=None, api_secret=None, demo=False, testnet=False, **kwargs):
            self.demo = demo
            self.testnet = testnet
        def _create_api_uri(self, path, signed=True, version="v3"):
            return "https://api.binance.com/api/v3/" + path  # wrong: real endpoint regardless of flags

    with patch("binance.client.Client", BrokenClient):
        try:
            exchange_module.BinanceExchange("fakekey", "fakesecret", mode="demo")
            raise AssertionError("should have caught the endpoint mismatch")
        except RuntimeError as e:
            assert "does not contain the expected" in str(e)
    print("PASS: endpoint safety check catches a genuine URL mismatch, not just rubber-stamping")


def test_endpoint_safety_check_catches_flag_mismatch():
    class WrongFlagsClient:
        def __init__(self, api_key=None, api_secret=None, demo=False, testnet=False, **kwargs):
            self.demo = demo
            self.testnet = True  # bug: ignores the requested flag
        def _create_api_uri(self, path, signed=True, version="v3"):
            return "https://testnet.binance.vision/api/v3/" + path

    with patch("binance.client.Client", WrongFlagsClient):
        try:
            exchange_module.BinanceExchange("fakekey", "fakesecret", mode="demo")
            raise AssertionError("should have caught the flag mismatch")
        except RuntimeError as e:
            assert "client.demo=" in str(e) and "client.testnet=" in str(e)
    print("PASS: endpoint safety check catches a boolean-flag mismatch")


def test_invalid_mode_string_rejected():
    try:
        exchange_module.BinanceExchange("fakekey", "fakesecret", mode="not_a_real_mode")
        raise AssertionError("should have rejected an unknown mode string")
    except ValueError:
        pass
    print("PASS: unrecognized mode strings are rejected")


def test_mode_banner_names_the_active_mode():
    for mode in ("paper", "demo", "testnet"):
        config.Config.TRADING_MODE = mode
        banner = config.Config.mode_banner()
        assert mode.upper() in banner
    print("PASS: mode_banner() clearly names whichever mode is active")


def main():
    tests = [
        test_paper_mode_has_no_authenticated_client,
        test_demo_mode_resolves_to_demo_endpoint,
        test_testnet_mode_resolves_to_testnet_endpoint,
        test_demo_mode_without_keys_falls_back_to_paper,
        test_live_mode_refused_by_exchange_layer,
        test_live_mode_refused_by_config_layer,
        test_demo_mode_requires_api_keys_at_config_level,
        test_endpoint_safety_check_catches_url_mismatch,
        test_endpoint_safety_check_catches_flag_mismatch,
        test_invalid_mode_string_rejected,
        test_mode_banner_names_the_active_mode,
    ]
    # Every mode here — including paper's unauthenticated public client —
    # constructs a real python-binance Client, which pings Binance at
    # construction time. That network call is the only thing patched below;
    # everything else (URL resolution, attribute handling) runs the library's
    # real code, which is the point of testing against it rather than a fake.
    with patch.object(RealBinanceClient, "ping", lambda self: None):
        for t in tests:
            t()
    print(f"\nALL {len(tests)} DEMO-MODE SAFETY CHECKS PASSED")


if __name__ == "__main__":
    main()
