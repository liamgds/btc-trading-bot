"""
risk_manager.py — Live portfolio bookkeeping.

Deliberately thin. Position sizing (compute_size), stop-loss placement
(compute_stop_loss), and exit-trigger logic (evaluate_position_candle) all
live in strategy.py — this module does NOT recompute or duplicate any of
that. Its only job is tracking state ACROSS polling cycles that strategy.py's
pure functions need as inputs: current equity, today's P&L, the rolling
return history, consecutive win/loss streaks, and drawdown. This is exactly
the state that backtest.run() keeps in local variables inside its loop —
here it's kept on an object so bot.py's live loop can maintain it between
calls to the exchange.
"""

import json
import math
import os
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple


class RiskManager:
    def __init__(self, initial_capital: float, cfg: Dict,
                state_path: str = "logs/risk_state.json"):
        self.cfg = cfg
        self.state_path = state_path

        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.position: Optional[Dict] = None
        self.daily_pnl: Dict[str, float] = {}
        self.recent_returns: List[float] = []
        self.consec_losses = 0
        self.consec_wins = 0
        self.trades: List[Dict] = []

        self._load_state()

    # ------------------------------------------------------------- sizing inputs

    @property
    def dd_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    def today_pnl(self, today: date_type) -> float:
        return self.daily_pnl.get(str(today), 0.0)

    def sizing_inputs(self, today: date_type) -> Tuple[float, List[float], int, float, float]:
        """Returns exactly the arguments strategy.compute_size() needs."""
        return (self.equity, self.recent_returns, self.consec_losses,
                self.dd_pct, self.today_pnl(today))

    def can_open_new_position(self, today: date_type) -> Tuple[bool, str]:
        if self.position is not None:
            return False, "position_already_open"
        if self.today_pnl(today) < -(self.equity * self.cfg["max_daily_loss_pct"] / 100):
            return False, "daily_loss_limit_hit"
        return True, "ok"

    def has_open_position(self) -> bool:
        return self.position is not None

    # ------------------------------------------------------------- mutation

    def open_position(self, position: Dict):
        self.position = position
        self._save_state()

    def record_partial_exit(self, pe: Dict, today: date_type) -> float:
        """pe: {'qty','price','fee_rate','label'} from evaluate_position_candle()."""
        entry = self.position["entry"]
        p_gross = (pe["price"] - entry) * pe["qty"]
        p_fee = (entry + pe["price"]) * pe["qty"] * pe["fee_rate"]
        p_pnl = p_gross - p_fee
        self.equity += p_pnl
        self.daily_pnl[str(today)] = self.today_pnl(today) + p_pnl
        self.position["partial_pnl"] = self.position.get("partial_pnl", 0) + p_pnl
        self._save_state()
        return p_pnl

    def record_full_exit(self, exit_price: float, reason: str, exit_time,
                         today: date_type, fee_maker: float, fee_slip: float) -> Dict:
        """Closes self.position, updates equity/streaks/rolling-returns, returns the trade dict."""
        entry = self.position["entry"]
        gross = (exit_price - entry) * self.position["qty"]
        cost = (entry + exit_price) * self.position["qty"] * (fee_maker + fee_slip)
        pnl = gross - cost + self.position.get("partial_pnl", 0)

        self.equity += (gross - cost)
        self.daily_pnl[str(today)] = self.today_pnl(today) + (gross - cost)
        self.peak_equity = max(self.peak_equity, self.equity)

        pnl_pct = (pnl / (self.position["qty_original"] * entry) * 100
                  if entry > 0 and self.position["qty_original"] > 0 else 0.0)
        if pnl > 0:
            self.consec_wins += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            self.consec_wins = 0

        self.recent_returns.append(pnl_pct / 100)
        if len(self.recent_returns) > self.cfg["vol_lookback_trades"] * 2:
            self.recent_returns.pop(0)

        trade = {
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "entry_price": entry,
            "exit_price": exit_price,
            "qty": self.position["qty_original"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
        }
        self.trades.append(trade)
        self.position = None
        self._save_state()
        return trade

    def update_position_state(self, position: Dict):
        """Persist in-place mutations made by evaluate_position_candle (trailing stop, t1_done, etc)."""
        self.position = position
        self._save_state()

    # ------------------------------------------------------------- persistence

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        state = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "position": self.position,
            "daily_pnl": self.daily_pnl,
            "recent_returns": self.recent_returns,
            "consec_losses": self.consec_losses,
            "consec_wins": self.consec_wins,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)

            equity = state.get("equity", self.equity)
            peak_equity = state.get("peak_equity", self.peak_equity)
            self.equity = equity if math.isfinite(equity) else self.equity
            self.peak_equity = peak_equity if math.isfinite(peak_equity) else self.peak_equity
            self.position = state.get("position")
            self.daily_pnl = {k: v for k, v in state.get("daily_pnl", {}).items() if math.isfinite(v)}
            # Defensive: a non-finite return here would silently poison
            # compute_size()'s volatility-targeting layer for every future
            # trade, so any corrupted entries (e.g. from an older bug, a
            # crash mid-write) are dropped rather than loaded.
            self.recent_returns = [r for r in state.get("recent_returns", []) if math.isfinite(r)]
            self.consec_losses = state.get("consec_losses", 0)
            self.consec_wins = state.get("consec_wins", 0)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass  # start fresh if the state file is corrupt/unreadable

    def display_statistics(self) -> str:
        wins = [t for t in self.trades if t["pnl"] > 0]
        wr = (len(wins) / len(self.trades) * 100) if self.trades else 0.0
        return (f"Equity: ${self.equity:,.2f} | Trades: {len(self.trades)} | "
               f"Win rate: {wr:.1f}% | Drawdown: {self.dd_pct:.2f}% | "
               f"Consecutive losses: {self.consec_losses}")
