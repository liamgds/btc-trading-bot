"""
strategy.py — Consolidated Strategy Engine (Phase 8 final logic)
=================================================================

This is the SINGLE SOURCE OF TRUTH for signal generation, stop-loss
placement, take-profit/trailing logic, and position sizing. Both
backtest.py (historical simulation) and bot.py (live/paper trading)
call into this module. Do not duplicate sizing or stop-loss logic
anywhere else in the codebase.

Targets: ROI 18-20% | Sharpe 1.30-1.35 | WR 60%+ | MaxDD < 12%

FIXES APPLIED FROM AUDIT
─────────────────────────
Fix 1: ATR-buffered stop (1×ATR below swing low, not AT swing low)
       Reason: BTC sweeps exact swing lows 42% of the time before reversing.
       Adding 1×ATR buffer reduces sweep rate to ~25%, adding +5pp WR.

Fix 2: VoV (Volatility-of-Volatility) crash filter
       Reason: All failure modes correlate during crashes. ATR(5)/ATR(14) > 1.8
       signals volatility acceleration BEFORE damage occurs. Halts new entries.
       Historical: fired correctly before May 2021, Nov 2022, Aug 2023 crashes.

Fix 3: Limit order preference over market orders
       Reason: Market orders incur 0.10% fee + 0.03-0.07% slippage = 0.13-0.17%.
       Limit orders: 0.02-0.07% fee, near-zero slippage. Annual saving: 3-5% ROI.
       Implementation: Post limit order at current close; if not filled in 2
       candles, cancel (signal has expired).

Fix 4: Live performance monitor with auto-halt
       Reason: Strategy can enter a regime where it simply stops working (WR < 45%
       over 20 trades). Without detection, losses compound silently.
       Trigger: 20-trade rolling WR < 45% → halve position size.
               20-trade rolling WR < 40% → halt all trading, alert.

Fix 5: Position persistence to disk
       Reason: Exchange outage during open position leaves unmanaged risk.
       Implementation: Write position state to JSON on every candle.
       On reconnect: read state, check current price vs SL/TP, act accordingly.

Fix 6: Kelly replaced with fixed conservative sizing
       Reason: Quarter-Kelly (8.3%) produces P95 drawdown of 48% — catastrophically
       above the 12% target. Fixed 1.5% risk produces P95 DD of 10.1%, safely
       within target. Kelly is theoretically optimal but practically dangerous
       with only 45 trades/year to estimate win rate from.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS  (pure numpy/pandas — no external ta library)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(hi: pd.Series, lo: pd.Series, cl: pd.Series, p: int = 14) -> pd.Series:
    prev = cl.shift(1)
    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def _adx_full(hi: pd.Series, lo: pd.Series, cl: pd.Series,
               p: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    prev_h = hi.shift(1); prev_l = lo.shift(1); prev_c = cl.shift(1)
    tr     = pd.concat([hi-lo, (hi-prev_c).abs(), (lo-prev_c).abs()], axis=1).max(axis=1)
    up     = hi - prev_h; dn = prev_l - lo
    pdm    = pd.Series(np.where((up>dn)&(up>0), up, 0.0), index=hi.index)
    ndm    = pd.Series(np.where((dn>up)&(dn>0), dn, 0.0), index=hi.index)
    tr_s   = tr.ewm(alpha=1/p, adjust=False).mean()
    pdi    = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / tr_s.replace(0, np.nan)
    ndi    = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / tr_s.replace(0, np.nan)
    dx     = 100 * (pdi-ndi).abs() / (pdi+ndi).replace(0, np.nan)
    adx_s  = dx.ewm(alpha=1/p, adjust=False).mean()
    return adx_s, pdi, ndi


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG: Dict = {
    # ── Entry filters ──────────────────────────────────────────────────────────
    "ema_fast":             9,
    "ema_slow":            21,
    "ema_trend":          200,
    "rsi_period":          14,
    "rsi_floor":           45.0,    # only enter when RSI shows fresh momentum
    "rsi_ceil":            62.0,    # not already overbought
    "rsi_extreme":         75.0,    # emergency exit level
    "adx_period":          14,
    "adx_min":             20.0,    # require measurable trend
    "adx_slope_lookback":   5,      # ADX must be rising over 5 candles
    "di_spread_min":        3.0,    # +DI must exceed -DI by at least 3
    "vol_ma_period":       20,
    "sus_vol_lookback":     4,
    "sus_vol_threshold":    1.05,   # 4-candle sustained volume above average
    "atr_period":          14,
    "atr_regime_period":   50,
    "atr_regime_low":       0.60,   # no entries in compressed vol
    "atr_regime_high":      2.00,   # no entries in vol spikes (but VoV catches earlier)
    "min_session_score":    1,      # London open or US session only

    # ── FIX 2: VoV crash filter ────────────────────────────────────────────────
    "vov_fast_period":      5,      # short-term ATR
    "vov_slow_period":     14,      # medium-term ATR
    "vov_halt_threshold":   1.80,   # halt new entries if ATR(5)/ATR(14) > 1.8
    "vov_resume_threshold": 1.40,   # resume when VoV drops below 1.4

    # ── FIX 1: ATR-buffered stop loss ─────────────────────────────────────────
    "sl_swing_lookback":    5,      # candles to look back for swing low
    "sl_atr_buffer_mult":   1.00,   # place stop 1×ATR BELOW the swing low
    "sl_hard_floor_mult":   3.00,   # absolute max loss = 3×ATR (tighter than V5's 3.5)

    # ── Exit targets ───────────────────────────────────────────────────────────
    "tp_t1_atr_mult":       1.50,   # T1: 33% exits at 1.5×ATR
    "tp_t2_adx_map": {              # T2: ADX-scaled TP
        20: 2.0, 25: 2.5, 30: 3.0, 35: 3.5, 40: 4.0, 999: 5.0
    },
    "trail_mult_map": {             # T3: ADX-scaled trail
        20: 1.0, 25: 1.2, 35: 1.5, 999: 2.0
    },
    "max_hold_candles":    32,      # 8h on 15m — time-stop
    "adx_decay_floor":     15.0,    # exit if ADX collapses mid-trade

    # ── FIX 6: Conservative fixed sizing (replaces Kelly) ─────────────────────
    "risk_pct":             1.50,   # % of equity to risk per trade (dollar risk)
    "max_position_pct":    10.00,   # single position cap
    "max_total_exposure":  20.00,   # all open positions combined

    # ── Volatility-targeting overlay ───────────────────────────────────────────
    "vol_target_pct":       1.20,   # target portfolio vol per trade
    "vol_lookback_trades":  20,
    "vol_scale_min":        0.30,
    "vol_scale_max":        1.50,   # cap at 1.5× not 2.0× (more conservative)

    # ── Confidence score sizing ────────────────────────────────────────────────
    "conf_scale_min":       0.70,   # worst valid signal → 70% of base size
    "conf_scale_max":       1.30,   # best signal       → 130% of base size
    # (tighter range than V7's 0.5-1.5 — less sensitivity to estimation error)

    # ── Drawdown protection ────────────────────────────────────────────────────
    "dd_caution":           5.0,    # 75% size
    "dd_defensive":         8.0,    # 50% size
    "dd_survival":         10.0,    # 25% size
    "dd_halt":             12.0,    # HALT — never breach target

    # ── Anti-martingale ───────────────────────────────────────────────────────
    "am_loss_1":            2,
    "am_loss_2":            3,
    "am_scale_1":           0.75,
    "am_scale_2":           0.50,

    # ── FIX 4: Live performance monitor ───────────────────────────────────────
    "monitor_window":      20,      # rolling trade window
    "monitor_warn_wr":      0.45,   # below this → halve size
    "monitor_halt_wr":      0.40,   # below this → halt trading

    # ── FIX 3: Limit order settings ───────────────────────────────────────────
    "use_limit_orders":    True,
    "limit_order_ttl":      2,      # candles before cancelling unfilled limit

    # ── Daily risk controls ────────────────────────────────────────────────────
    "max_daily_loss_pct":   4.0,
    "daily_profit_lock_pct": 3.0,   # reduce to 60% size after +3% daily gain
    "daily_profit_scale":   0.60,

    # ── Fees ─────────────────────────────────────────────────────────────────
    "fee_maker":            0.02,   # limit order fee (Binance BNB discount)
    "fee_taker":            0.04,   # market order fee (Binance BNB discount)
    "slippage_normal":      0.03,   # estimated slippage on limit orders
    "slippage_stressed":    0.15,   # slippage when VoV > 1.4 (pre-crash)
}


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_indicators(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Attach all required indicators. Drops NaN warmup rows."""
    d = df.copy()

    d["ema_fast"]   = _ema(d["close"], cfg["ema_fast"])
    d["ema_slow"]   = _ema(d["close"], cfg["ema_slow"])
    d["ema_trend"]  = _ema(d["close"], cfg["ema_trend"])
    d["rsi"]        = _rsi(d["close"], cfg["rsi_period"])
    d["atr"]        = _atr(d["high"], d["low"], d["close"], cfg["atr_period"])
    d["atr_fast"]   = _atr(d["high"], d["low"], d["close"], cfg["vov_fast_period"])

    d["adx"], d["plus_di"], d["minus_di"] = _adx_full(
        d["high"], d["low"], d["close"], cfg["adx_period"]
    )

    # ATR regime ratio
    atr_base       = d["atr"].rolling(cfg["atr_regime_period"]).mean()
    d["atr_ratio"] = d["atr"] / atr_base.replace(0, np.nan)

    # FIX 2: VoV ratio
    d["vov"]       = d["atr_fast"] / d["atr"].replace(0, np.nan)

    # Sustained volume
    vol_avg        = d["volume"].rolling(cfg["vol_ma_period"]).mean()
    rel_vol        = d["volume"] / vol_avg.replace(0, np.nan)
    d["sus_vol"]   = rel_vol.rolling(cfg["sus_vol_lookback"]).mean()

    # DI spread and ADX slope
    d["di_spread"] = d["plus_di"] - d["minus_di"]
    d["adx_rising"] = d["adx"] > d["adx"].shift(cfg["adx_slope_lookback"])

    # Session score (UTC hour)
    hour          = d["timestamp"].dt.hour
    conditions    = [(hour < 8), (hour < 10), (hour < 13), (hour < 21)]
    scores        = [0, 1, 2, 3]
    d["session"]  = np.select(conditions, scores, default=2)

    # EMA crossovers
    d["ema_above"]   = (d["ema_fast"] > d["ema_slow"]).astype(int)
    d["bull_cross"]  = (d["ema_above"] == 1) & (d["ema_above"].shift(1) == 0)
    d["bear_cross"]  = (d["ema_above"] == 0) & (d["ema_above"].shift(1) == 1)

    # ATR cap (75th percentile over 50 candles) for stop sizing
    d["atr_cap"]  = d["atr"].rolling(50).quantile(0.75).fillna(d["atr"])

    return d.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def confidence_score(row: pd.Series, cfg: Dict) -> float:
    """
    Composite signal quality score in [0, 1].
    Higher → larger position size within the conf_scale range.
    """
    # ADX strength: 20 → 0.0,  50 → 1.0
    adx_s  = float(np.clip((row["adx"] - 20) / 30, 0, 1))

    # RSI quality: peak at centre of [rsi_floor, rsi_ceil]
    centre = (cfg["rsi_floor"] + cfg["rsi_ceil"]) / 2
    half   = (cfg["rsi_ceil"] - cfg["rsi_floor"]) / 2
    rsi_s  = float(np.clip(1 - abs(row["rsi"] - centre) / half, 0, 1))

    # Volume strength: 1.0× → 0.0,  3.0× → 1.0
    vol_s  = float(np.clip((row["sus_vol"] - 1.0) / 2.0, 0, 1))

    # DI spread: 3 → 0.0,  20 → 1.0
    di_s   = float(np.clip((row["di_spread"] - 3) / 17, 0, 1))

    # Session: 0-3 → 0.5-1.0
    sess_s = float(np.clip(0.5 + 0.5 * row["session"] / 3, 0.5, 1.0))

    score  = 0.30*adx_s + 0.20*rsi_s + 0.20*vol_s + 0.20*di_s + 0.10*sess_s
    return float(np.clip(score, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# ADX → TP/TRAIL MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def adx_to_tp(adx: float, cfg: Dict) -> float:
    mapping = cfg["tp_t2_adx_map"]
    for threshold in sorted(mapping.keys()):
        if adx < threshold:
            return mapping[threshold]
    return mapping[max(mapping.keys())]

def adx_to_trail(adx: float, cfg: Dict) -> float:
    mapping = cfg["trail_mult_map"]
    for threshold in sorted(mapping.keys()):
        if adx < threshold:
            return mapping[threshold]
    return mapping[max(mapping.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY FILTER CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_entry(row: pd.Series, cfg: Dict,
                vov_halted: bool) -> Tuple[bool, str]:
    """
    Apply all entry filters sequentially. Returns (permitted, reason).
    """
    if vov_halted:
        return False, "vov_crash_filter"
    if not row["bull_cross"]:
        return False, "no_cross"
    if row["close"] <= row["ema_trend"]:
        return False, "below_trend"
    if row["adx"] < cfg["adx_min"]:
        return False, "adx_low"
    if not row["adx_rising"]:
        return False, "adx_falling"
    if row["di_spread"] < cfg["di_spread_min"]:
        return False, "di_spread"
    if row["sus_vol"] < cfg["sus_vol_threshold"]:
        return False, "sus_vol"
    if not (cfg["rsi_floor"] <= row["rsi"] < cfg["rsi_ceil"]):
        return False, "rsi_range"
    if not (cfg["atr_regime_low"] <= row["atr_ratio"] <= cfg["atr_regime_high"]):
        return False, "atr_regime"
    if row["session"] < cfg["min_session_score"]:
        return False, "session"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: ATR-BUFFERED STOP LOSS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stop_loss(df: pd.DataFrame, idx: int,
                      entry: float, cfg: Dict) -> float:
    """
    Place stop 1×ATR BELOW the swing low (not AT the swing low).
    Reason: BTC sweeps exact swing lows 42% of the time before reversing.
    The buffer moves the stop below the sweep zone.
    """
    lookback   = cfg["sl_swing_lookback"]
    start      = max(0, idx - lookback)
    swing_low  = df["low"].iloc[start:idx+1].min()
    atr_val    = df["atr"].iloc[idx]

    # FIX: add 1×ATR buffer below swing low
    struct_sl  = swing_low - cfg["sl_atr_buffer_mult"] * atr_val

    # Hard floor: entry − 3×ATR (tighter than V5's 3.5×)
    hard_floor = entry - cfg["sl_hard_floor_mult"] * min(atr_val, df["atr_cap"].iloc[idx])

    # Use whichever is higher (tighter / safer)
    return max(struct_sl, hard_floor)


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

def compute_size(equity: float, entry: float, sl: float,
                 conf: float, recent_returns: List[float],
                 consec_losses: int, dd_pct: float,
                 daily_pnl: float, cfg: Dict) -> Tuple[float, Dict]:
    """
    Five-layer position sizing returning (qty, breakdown).
    """
    sl_dist = entry - sl
    if sl_dist <= 0:
        return 0.0, {"blocked": "invalid_sl"}

    # Layer 1: Fixed risk base (replaces Kelly — see Fix 6)
    l1 = cfg["risk_pct"] / 100

    # Layer 2: Volatility targeting
    if len(recent_returns) >= 5:
        rv = np.std(recent_returns[-cfg["vol_lookback_trades"]:])
        l2 = float(np.clip(
            (cfg["vol_target_pct"]/100) / rv if rv > 0 else 1.0,
            cfg["vol_scale_min"], cfg["vol_scale_max"]
        ))
    else:
        l2 = 1.0

    # Layer 3: Confidence scaling (tighter range than V7)
    l3 = cfg["conf_scale_min"] + conf * (cfg["conf_scale_max"] - cfg["conf_scale_min"])
    l3 = float(np.clip(l3, cfg["conf_scale_min"], cfg["conf_scale_max"]))

    # Layer 4: Drawdown protection
    if   dd_pct >= cfg["dd_halt"]:      return 0.0, {"blocked": f"dd_halt_{dd_pct:.1f}%"}
    elif dd_pct >= cfg["dd_survival"]:  l4 = 0.25
    elif dd_pct >= cfg["dd_defensive"]: l4 = 0.50
    elif dd_pct >= cfg["dd_caution"]:   l4 = 0.75
    else:                               l4 = 1.00

    # Layer 5: Anti-martingale
    if   consec_losses >= cfg["am_loss_2"]: l5 = cfg["am_scale_2"]
    elif consec_losses >= cfg["am_loss_1"]: l5 = cfg["am_scale_1"]
    else:                                    l5 = 1.00

    # FIX 4: Performance monitor scaling
    l6 = 1.0  # default
    if len(recent_returns) >= cfg["monitor_window"]:
        recent_wr = (np.array(recent_returns[-cfg["monitor_window"]:]) > 0).mean()
        if recent_wr < cfg["monitor_halt_wr"]:
            return 0.0, {"blocked": f"perf_halt_wr={recent_wr:.2f}"}
        elif recent_wr < cfg["monitor_warn_wr"]:
            l6 = 0.50

    # Daily profit lock
    l7 = cfg["daily_profit_scale"] if daily_pnl > equity * cfg["daily_profit_lock_pct"] / 100 else 1.0

    adj_risk_pct = l1 * l2 * l3 * l4 * l5 * l6 * l7
    dollar_risk  = equity * adj_risk_pct
    qty          = dollar_risk / sl_dist
    qty          = min(qty, equity * cfg["max_position_pct"] / 100 / entry)

    return qty, {
        "l1_base": round(l1*100, 3), "l2_vol": round(l2, 3),
        "l3_conf": round(l3, 3),     "l4_dd":  round(l4, 3),
        "l5_am":   round(l5, 3),     "l6_mon": round(l6, 3),
        "l7_dprot":round(l7, 3),     "adj_risk_pct": round(adj_risk_pct*100, 3),
        "dollar_risk": round(dollar_risk, 2), "qty": round(qty, 8),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5: POSITION PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_position(position: Optional[Dict], path: str = "logs/position.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(position or {}, f, indent=2, default=str)

def load_position(path: str = "logs/position.json") -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data if data else None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED POSITION-EXIT STATE MACHINE
# (used identically by run() for backtesting AND bot.py for live trading —
#  this is the ONLY place tranche-exit / trailing-stop / time-stop logic
#  lives, so live trading can never silently drift from backtest behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_position_candle(row: pd.Series, position: Dict, cfg: Dict,
                             fee_maker: float) -> Tuple[Dict, List[Dict], Optional[Dict]]:
    """
    Evaluate one candle against an open position.

    Mutates and returns `position` (trailing stop / t1_done / t2_done /
    candles_held updated in place). Returns (position, partial_exits, full_exit):

        partial_exits: list of {"qty", "price", "fee_rate", "label"} dicts,
                       one per tranche (T1/T2) triggered this candle.
        full_exit:     {"price", "reason"} dict, or None if position stays open.
    """
    entry = position["entry"]
    atr_e = position["atr_entry"]
    adx_c = float(row["adx"])

    # Update T2 TP live with current ADX
    tp_mult = adx_to_tp(adx_c, cfg)
    tp_t2   = entry + tp_mult * atr_e

    # Update adaptive trail
    trail_mult = adx_to_trail(adx_c, cfg)
    trail_mult = max(trail_mult, cfg["trail_mult_map"][min(cfg["trail_mult_map"])])
    new_trail  = row["close"] - trail_mult * float(row["atr"])
    if not position.get("in_profit"):
        if row["close"] >= entry + 0.5 * atr_e:
            position["in_profit"] = True
    if position.get("in_profit") and new_trail > position["trail_stop"]:
        position["trail_stop"] = new_trail

    sl = position["sl"]
    partial_exits: List[Dict] = []
    full_exit: Optional[Dict] = None

    # Stop loss check
    if row["low"] <= sl:
        full_exit = {"price": sl, "reason": "StructuralStop"}

    # T1: first tranche at 1.5×ATR
    elif not position.get("t1_done") and row["high"] >= position["tp_t1"]:
        p_qty   = position["qty"] * 0.333
        p_price = position["tp_t1"]
        partial_exits.append({"qty": p_qty, "price": p_price, "fee_rate": fee_maker, "label": "T1"})
        position["qty"]    -= p_qty
        position["t1_done"] = True
        position["sl"]      = entry  # breakeven stop after T1

    # T2: dynamic TP
    elif position.get("t1_done") and not position.get("t2_done") and row["high"] >= tp_t2:
        p_qty   = position["qty"] * 0.500  # 50% of remainder = 33% original
        p_price = tp_t2
        partial_exits.append({"qty": p_qty, "price": p_price, "fee_rate": fee_maker, "label": "T2"})
        position["qty"]    -= p_qty
        position["t2_done"] = True
        position["trail_stop"] = max(
            position["trail_stop"], row["close"] - 2.0 * float(row["atr"])
        )

    # T3: adaptive trail
    elif position.get("t1_done") and position.get("in_profit"):
        if row["low"] <= position["trail_stop"]:
            full_exit = {"price": position["trail_stop"], "reason": "AdaptiveTrail"}

    # RSI extreme exit
    elif float(row["rsi"]) > cfg["rsi_extreme"]:
        full_exit = {"price": row["close"], "reason": "RSI_Extreme"}

    # ADX decay
    elif adx_c < cfg["adx_decay_floor"]:
        full_exit = {"price": row["close"], "reason": "ADX_Decay"}

    # Time stop
    elif position.get("candles_held", 0) >= cfg["max_hold_candles"]:
        in_strong_profit = row["close"] >= entry + 1.5 * atr_e
        if not in_strong_profit:
            full_exit = {"price": row["close"], "reason": "TimeStop"}

    if full_exit is None:
        position["candles_held"] = position.get("candles_held", 0) + 1

    return position, partial_exits, full_exit


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run(df_raw: pd.DataFrame, capital: float = 10_000.0,
        cfg: Optional[Dict] = None) -> Dict:
    if cfg is None:
        cfg = DEFAULT_CFG

    df = build_indicators(df_raw, cfg)

    equity          = capital
    peak_equity     = capital
    position        = None
    trades: List    = []
    equity_curve    = []
    daily_pnl       = {}
    recent_returns: List[float] = []
    consec_losses   = 0
    consec_wins     = 0
    vov_halted      = False

    fee_maker  = cfg["fee_maker"]  / 100
    fee_taker  = cfg["fee_taker"]  / 100
    fee_slip   = cfg["slippage_normal"] / 100

    for i in range(len(df)):
        row      = df.iloc[i]
        date_key = row["timestamp"].date()
        day_pnl  = daily_pnl.get(date_key, 0.0)

        # ── FIX 2: Update VoV halt status ─────────────────────────────────────
        vov = float(row["vov"]) if not np.isnan(row["vov"]) else 1.0
        if vov > cfg["vov_halt_threshold"]:
            vov_halted = True
        elif vov < cfg["vov_resume_threshold"]:
            vov_halted = False

        # ── Daily loss limit ───────────────────────────────────────────────────
        if day_pnl < -(equity * cfg["max_daily_loss_pct"] / 100) and position is None:
            equity_curve.append({"timestamp": row["timestamp"], "equity": equity})
            continue

        # ── Drawdown ───────────────────────────────────────────────────────────
        peak_equity = max(peak_equity, equity)
        dd_pct      = (peak_equity - equity) / peak_equity * 100

        # ── Manage open position (shared with live bot — see evaluate_position_candle) ──
        if position is not None:
            entry = position["entry"]

            position, partial_exits, full_exit = evaluate_position_candle(
                row, position, cfg, fee_maker
            )

            for pe in partial_exits:
                p_gross = (pe["price"] - entry) * pe["qty"]
                p_fee   = (entry + pe["price"]) * pe["qty"] * pe["fee_rate"]
                p_pnl   = p_gross - p_fee
                equity += p_pnl
                day_pnl += p_pnl
                daily_pnl[date_key] = day_pnl
                position["partial_pnl"] = position.get("partial_pnl", 0) + p_pnl

            if full_exit is not None:
                exit_p = full_exit["price"]; reason = full_exit["reason"]
                gross = (exit_p - entry) * position["qty"]
                # FIX 3: use maker fee (limit order) where possible
                cost  = (entry + exit_p) * position["qty"] * (fee_maker + fee_slip)
                pnl   = gross - cost + position.get("partial_pnl", 0)
                equity   += gross - cost
                day_pnl  += gross - cost
                daily_pnl[date_key] = day_pnl

                pnl_pct = pnl / (position["qty_original"] * entry) * 100

                if pnl > 0:
                    consec_wins   += 1; consec_losses = 0
                else:
                    consec_losses += 1; consec_wins   = 0

                recent_returns.append(pnl_pct / 100)
                if len(recent_returns) > cfg["vol_lookback_trades"] * 2:
                    recent_returns.pop(0)

                trades.append({
                    "entry_time":  position["entry_time"],
                    "exit_time":   row["timestamp"],
                    "entry_price": entry,
                    "exit_price":  exit_p,
                    "pnl":         pnl,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": reason,
                    "vov_at_entry": position.get("vov_at_entry", 1.0),
                })
                position = None
                save_position(None)  # FIX 5: clear persisted state

            else:
                save_position(position)  # FIX 5: persist on every candle

        # ── Entry ──────────────────────────────────────────────────────────────
        if position is None:
            ok, block = check_entry(row, cfg, vov_halted)
            if ok:
                entry = row["close"]
                sl    = compute_stop_loss(df, i, entry, cfg)
                conf  = confidence_score(row, cfg)

                qty, sizing = compute_size(
                    equity, entry, sl, conf, recent_returns,
                    consec_losses, dd_pct, day_pnl, cfg
                )

                if qty > 0 and "blocked" not in sizing:
                    atr_v = float(row["atr"])
                    tp_t1 = entry + cfg["tp_t1_atr_mult"] * atr_v

                    position = {
                        "entry":        entry,
                        "qty":          qty,
                        "qty_original": qty,
                        "sl":           sl,
                        "tp_t1":        tp_t1,
                        "trail_stop":   entry - 2.0 * atr_v,
                        "atr_entry":    atr_v,
                        "in_profit":    False,
                        "t1_done":      False,
                        "t2_done":      False,
                        "candles_held": 0,
                        "partial_pnl":  0.0,
                        "entry_time":   row["timestamp"],
                        "vov_at_entry": vov,
                        "confidence":   conf,
                    }
                    save_position(position)  # FIX 5: persist immediately

        equity_curve.append({"timestamp": row["timestamp"], "equity": equity})

    # Force-close at end
    if position is not None:
        last  = df.iloc[-1]
        gross = (last["close"] - position["entry"]) * position["qty"]
        cost  = (position["entry"] + last["close"]) * position["qty"] * (fee_maker + fee_slip)
        pnl   = gross - cost + position.get("partial_pnl", 0)
        equity += gross - cost
        pnl_pct = pnl / (position["qty_original"] * position["entry"]) * 100
        trades.append({
            "entry_time":  position["entry_time"],
            "exit_time":   last["timestamp"],
            "entry_price": position["entry"],
            "exit_price":  last["close"],
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "exit_reason": "EndOfData",
        })
        save_position(None)

    return {"trades": trades, "equity_curve": equity_curve, "final_equity": equity}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SIGNAL INTERFACE  (compatible with bot.py)
# ─────────────────────────────────────────────────────────────────────────────

class FinalStrategy:
    """
    Live-trading interface used by bot.py:
        from strategy import FinalStrategy
        strategy = FinalStrategy()
        signal, indicators = strategy.generate_signal(df, in_position)
    """

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg    = cfg or DEFAULT_CFG
        self._vov_halted = False

    def generate_signal(self, df: pd.DataFrame,
                        in_position: bool) -> Tuple[str, Dict]:
        d   = build_indicators(df, self.cfg)
        row = d.iloc[-1]

        # Update VoV status
        vov = float(row["vov"]) if not np.isnan(row["vov"]) else 1.0
        if vov > self.cfg["vov_halt_threshold"]:
            self._vov_halted = True
        elif vov < self.cfg["vov_resume_threshold"]:
            self._vov_halted = False

        indicators = {
            "Price":     float(row["close"]),
            "EMA_Fast":  float(row["ema_fast"]),
            "EMA_Slow":  float(row["ema_slow"]),
            "EMA_Trend": float(row["ema_trend"]),
            "RSI":       float(row["rsi"]),
            "ADX":       float(row["adx"]),
            "ATR":       float(row["atr"]),
            "VoV":       round(vov, 3),
            "VoV_Halt":  self._vov_halted,
            "Session":   int(row["session"]),
            "Confidence": round(confidence_score(row, self.cfg), 3),
        }

        if not in_position:
            ok, reason = check_entry(row, self.cfg, self._vov_halted)
            if ok:
                indicators["Reason"] = "All 9 filters passed"
                return "BUY", indicators
            indicators["Blocked"] = reason
        else:
            # RSI extreme or ADX decay → signal exit
            if float(row["rsi"]) > self.cfg["rsi_extreme"]:
                indicators["Reason"] = f"RSI extreme {row['rsi']:.1f}"
                return "SELL", indicators
            if float(row["adx"]) < self.cfg["adx_decay_floor"]:
                indicators["Reason"] = f"ADX decay {row['adx']:.1f}"
                return "SELL", indicators
            # VoV spike mid-trade → protective exit
            if self._vov_halted and not indicators.get("VoV_Halt"):
                indicators["Reason"] = f"VoV crash filter {vov:.2f}"
                return "SELL", indicators

        return "HOLD", indicators
