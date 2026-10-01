"""End-of-day 125-day breakout (BUY) and breakdown (SELL) scans.

Implements the Chartink "Bullish" and "Bearish" guide scans exactly as the
saved Chartink conditions read, on completed daily bars:

    BUY : Close > 1 day ago Max(125, High)
          Volume > 1 day ago SMA(Volume, 125) x 2
          RSI(14) < 70
    SELL: Close < 1 day ago Min(125, Low)
          Volume < 1 day ago SMA(Volume, 125)       (as saved in the guide; configurable)
          RSI(14) > 30

"1 day ago Max(125, High)" is the highest high of the 125 sessions ending
yesterday, so today's bar is compared with a range it is not part of.
A liquidity floor (average turnover) is added because the guide runs on
the Nifty 500, where every stock is reasonably liquid.

Run after the close (16:00 IST onward); signals are for the next session.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from config.config import BreakoutConfig
from screeners.screener import atr, rsi

logger = logging.getLogger(__name__)


def breakout_frame(df: pd.DataFrame, cfg: BreakoutConfig, atr_period: int = 14) -> pd.DataFrame:
    """Per-bar rule components and BUY/SELL flags for one symbol's daily OHLCV."""
    L = cfg.lookback
    c, h, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
    prior_high = h.shift(1).rolling(L, min_periods=L).max()
    prior_low = lo.shift(1).rolling(L, min_periods=L).min()
    vol_avg = v.shift(1).rolling(L, min_periods=L).mean()
    rsi14 = rsi(c, 14)
    turnover_cr = (c * v).rolling(20, min_periods=20).mean() / 1e7
    liquid = turnover_cr >= cfg.min_avg_turnover_cr

    bull = (c > prior_high) & (v > cfg.volume_multiple * vol_avg) & (rsi14 < cfg.bull_rsi_max)
    bear_vol = (v < vol_avg) if cfg.bear_volume_rule == "below_avg" else (v > cfg.volume_multiple * vol_avg)
    bear = (c < prior_low) & bear_vol & (rsi14 > cfg.bear_rsi_min)

    return pd.DataFrame({
        "close": c, "prior_high": prior_high, "prior_low": prior_low,
        "volume_ratio": v / vol_avg, "rsi": rsi14, "atr": atr(h, lo, c, atr_period),
        "turnover_cr": turnover_cr, "buy": (bull & liquid).fillna(False).astype(bool),
        "sell": (bear & liquid).fillna(False).astype(bool),
    }, index=df.index)


def scan_breakouts(histories: Mapping[str, pd.DataFrame], cfg: BreakoutConfig) -> pd.DataFrame:
    """Evaluate the latest completed bar of every symbol. One row per symbol."""
    rows = []
    for symbol, df in histories.items():
        if len(df) < cfg.lookback + 20:
            logger.debug("%s: only %d bars; needs %d", symbol, len(df), cfg.lookback + 20)
            continue
        last = breakout_frame(df, cfg).iloc[-1]
        signal = "BUY" if last["buy"] else "SELL" if last["sell"] else ""
        rows.append({
            "Ticker": symbol, "Signal": signal, "Session": df.index[-1].date().isoformat(),
            "Close": round(float(last["close"]), 2),
            "125D High (prior)": round(float(last["prior_high"]), 2),
            "125D Low (prior)": round(float(last["prior_low"]), 2),
            "Beyond Range %": round(100 * (last["close"] / last["prior_high"] - 1) if signal != "SELL"
                                    else 100 * (last["close"] / last["prior_low"] - 1), 2),
            "Volume x Avg": round(float(last["volume_ratio"]), 2),
            "RSI": round(float(last["rsi"]), 1), "ATR": round(float(last["atr"]), 2),
            "Turnover (Cr)": round(float(last["turnover_cr"]), 1),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = out["Signal"].map({"BUY": 0, "SELL": 1, "": 2})
    return out.assign(_o=order).sort_values(["_o", "Volume x Avg"], ascending=[True, False]).drop(columns="_o") \
              .reset_index(drop=True)
