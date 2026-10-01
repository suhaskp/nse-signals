"""Trade-by-trade backtest of the breakout/breakdown rules.

For every signal on day *t* (rules evaluated on completed bars only):
    entry  = open of day t+1
    stop   = entry -/+ atr_mult x ATR(t)       (long / short)
    target = entry +/- reward_risk x stop distance
    exit   = first of stop, target, or the close after ``max_hold`` sessions

Conservative conventions: a gap beyond the stop exits at that day's open;
if stop and target are both inside one bar, the stop is assumed first.
Only one open trade per symbol at a time. Estimated round-trip costs are
deducted from every trade (reported in R). SELL trades are simulated as
shorts; in India an overnight short needs F&O, since cash-segment shorts
must be squared off the same day.

Bias warning: backtesting on today's ticker list ignores stocks that were
delisted or dropped from the index (survivorship bias), which flatters results.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from config.config import BreakoutConfig
from screeners.breakout import breakout_frame


def _simulate(df: pd.DataFrame, flags: pd.Series, atr_s: pd.Series, side: str, atr_mult: float,
              rr: float, max_hold: int, cost_pct: float) -> list[dict]:
    o, h, lo, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    sig_idx = np.flatnonzero(flags.to_numpy())
    trades, busy_until = [], -1
    for t in sig_idx:
        e_i = t + 1
        if e_i >= len(df) or t <= busy_until or not np.isfinite(atr_s.iloc[t]) or atr_s.iloc[t] <= 0:
            continue
        entry = o[e_i]
        dist = atr_mult * atr_s.iloc[t]
        long = side == "long"
        stop = entry - dist if long else entry + dist
        target = entry + rr * dist if long else entry - rr * dist
        exit_px, outcome, x_i = None, "time", min(e_i + max_hold - 1, len(df) - 1)
        for i in range(e_i, x_i + 1):
            if i > e_i and ((long and o[i] <= stop) or (not long and o[i] >= stop)):
                exit_px, outcome, x_i = o[i], "gap_stop", i
                break
            hit_stop = lo[i] <= stop if long else h[i] >= stop
            hit_tgt = h[i] >= target if long else lo[i] <= target
            if hit_stop:
                exit_px, outcome, x_i = stop, "stop", i
                break
            if hit_tgt:
                exit_px, outcome, x_i = target, "target", i
                break
        if exit_px is None:
            exit_px = c[x_i]
            if x_i < e_i + max_hold - 1:
                outcome = "open"  # data ended before the time exit
        pnl = (exit_px - entry) if long else (entry - exit_px)
        r = (pnl - cost_pct * entry) / dist
        trades.append({"signal_date": df.index[t].date(), "entry_date": df.index[e_i].date(),
                       "exit_date": df.index[x_i].date(), "side": side, "entry": entry, "exit": exit_px,
                       "outcome": outcome, "r_multiple": r, "return_pct": 100 * (pnl / entry - cost_pct),
                       "hold_days": x_i - e_i + 1})
        busy_until = x_i
    return trades


def backtest_breakouts(histories: Mapping[str, pd.DataFrame], cfg: BreakoutConfig,
                       atr_mult: float = 1.5, reward_risk: float = 2.0) -> pd.DataFrame:
    """All simulated trades across symbols (closed trades only)."""
    rows = []
    for symbol, df in histories.items():
        if len(df) < cfg.lookback + 30:
            continue
        f = breakout_frame(df, cfg)
        for flag, side in (("buy", "long"), ("sell", "short")):
            for tr in _simulate(df, f[flag], f["atr"], side, atr_mult, reward_risk,
                                cfg.backtest_max_hold_days, cfg.est_round_trip_cost_pct):
                rows.append({"Ticker": symbol, **tr})
    trades = pd.DataFrame(rows)
    return trades[trades["outcome"] != "open"].reset_index(drop=True) if not trades.empty else trades


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-side statistics: count, win rate, expectancy (avg R), profit factor, exits."""
    if trades.empty:
        return pd.DataFrame()
    out = []
    for side, g in trades.groupby("side"):
        wins, losses = g.loc[g["r_multiple"] > 0, "r_multiple"], g.loc[g["r_multiple"] <= 0, "r_multiple"]
        out.append({
            "Rule": "BUY (125D breakout)" if side == "long" else "SELL (125D breakdown)",
            "Trades": len(g), "Symbols": g["Ticker"].nunique(),
            "Win rate %": round(100 * len(wins) / len(g), 1),
            "Avg R": round(g["r_multiple"].mean(), 3), "Median R": round(g["r_multiple"].median(), 3),
            "Avg return %": round(g["return_pct"].mean(), 2),
            "Profit factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.inf,
            "Target %": round(100 * (g["outcome"] == "target").mean(), 1),
            "Stop %": round(100 * g["outcome"].isin(["stop", "gap_stop"]).mean(), 1),
            "Time exit %": round(100 * (g["outcome"] == "time").mean(), 1),
            "Avg hold (days)": round(g["hold_days"].mean(), 1),
            "First signal": str(g["signal_date"].min()), "Last signal": str(g["signal_date"].max()),
        })
    return pd.DataFrame(out)
