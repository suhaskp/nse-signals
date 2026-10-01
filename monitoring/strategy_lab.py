"""Strategy lab: the Chartink breakout rules versus improved variants, year by year.

Variants (BUY side):
    original      Chartink guide rules; fixed 2R target / 20-session time exit
    regime        + only when the Nifty closed above its 200-day average
    trailing      + regime, exit on a close below the prior 20-day low (lets trends run)
    no_rsi_cap    + regime + trailing, without the RSI < 70 cap (keeps strong breakouts)

Every variant: entry at the next open, initial stop 1.5 x ATR, delivery costs deducted,
one trade per stock at a time. SELL (breakdown) results are reported for the original rule.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from config.config import BreakoutConfig
from monitoring.breakout_backtest import _simulate
from screeners.breakout import breakout_frame

VARIANTS = {
    "original": "Chartink rules as in the guide (2R target / 20-day exit)",
    "regime": "+ only when Nifty is above its 200-day average",
    "trailing": "+ regime filter + trailing exit (close below 20-day low)",
    "no_rsi_cap": "+ regime + trailing, without the RSI < 70 cap",
    "quality": "Quality BUY: breakout + volume + trend + ATR stop + regime + next-day entry + 2:1 R/R",
    "quality_no_regime": "Same quality BUY rules but WITHOUT the market-regime check (buys in downtrends too)",
    "quality_3regime": "Quality BUY rules with three regimes: full size risk-on, HALF size neutral, none risk-off",
}


def _simulate_trailing(df: pd.DataFrame, flags: pd.Series, atr_s: pd.Series, atr_mult: float,
                       trail_days: int, max_hold: int, cost_pct: float) -> list[dict]:
    o, h, lo, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    trail = df["Low"].shift(1).rolling(trail_days, min_periods=trail_days).min().to_numpy()
    trades, busy = [], -1
    for t in np.flatnonzero(flags.to_numpy()):
        e = t + 1
        if e >= len(df) or t <= busy or not np.isfinite(atr_s.iloc[t]) or atr_s.iloc[t] <= 0:
            continue
        entry, dist = o[e], atr_mult * atr_s.iloc[t]
        stop = entry - dist
        exit_px, outcome, x = None, "time", min(e + max_hold - 1, len(df) - 1)
        for i in range(e, x + 1):
            if i > e and o[i] <= stop:
                exit_px, outcome, x = o[i], "gap_stop", i
                break
            if lo[i] <= stop:
                exit_px, outcome, x = stop, "stop", i
                break
            if i > e and np.isfinite(trail[i]) and c[i] < trail[i]:
                exit_px, outcome, x = c[i], "trail", i
                break
        if exit_px is None:
            exit_px = c[x]
            if x < e + max_hold - 1:
                outcome = "open"
        trades.append({"signal_date": df.index[t].date(), "exit_date": df.index[x].date(), "side": "long",
                       "outcome": outcome, "r_multiple": (exit_px - entry - cost_pct * entry) / dist,
                       "return_pct": 100 * ((exit_px - entry) / entry - cost_pct), "hold_days": x - e + 1})
        busy = x
    return trades


def run_lab(histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None, cfg: BreakoutConfig,
            atr_mult: float = 1.5, reward_risk: float = 2.0, trail_days: int = 20,
            max_hold_trailing: int = 120) -> pd.DataFrame:
    """All closed trades for every variant (column ``variant``)."""
    risk_on = None
    if benchmark is not None:
        bc = benchmark["Close"]
        risk_on = bc > bc.rolling(200, min_periods=200).mean()
    no_cap = replace(cfg, bull_rsi_max=100.0)
    rows = []
    for sym, df in histories.items():
        if len(df) < cfg.lookback + 30:
            continue
        f = breakout_frame(df, cfg)
        f_nc = breakout_frame(df, no_cap)
        ro = risk_on.reindex(df.index).fillna(False) if risk_on is not None else pd.Series(True, index=df.index)
        runs = {
            "original": _simulate(df, f["buy"], f["atr"], "long", atr_mult, reward_risk,
                                  cfg.backtest_max_hold_days, cfg.est_round_trip_cost_pct),
            "regime": _simulate(df, f["buy"] & ro, f["atr"], "long", atr_mult, reward_risk,
                                cfg.backtest_max_hold_days, cfg.est_round_trip_cost_pct),
            "trailing": _simulate_trailing(df, f["buy"] & ro, f["atr"], atr_mult, trail_days,
                                           max_hold_trailing, cfg.est_round_trip_cost_pct),
            "no_rsi_cap": _simulate_trailing(df, f_nc["buy"] & ro, f_nc["atr"], atr_mult, trail_days,
                                             max_hold_trailing, cfg.est_round_trip_cost_pct),
            "sell_original": _simulate(df, f["sell"], f["atr"], "short", atr_mult, reward_risk,
                                       cfg.backtest_max_hold_days, cfg.est_round_trip_cost_pct),
        }
        for variant, trades in runs.items():
            rows += [{"Ticker": sym, "variant": variant, **t} for t in trades if t["outcome"] != "open"]
    return pd.DataFrame(rows)


def lab_summary(trades: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    """One row per variant: trades, win rate, avg R, profit factor, share of positive years, verdict."""
    if trades.empty:
        return pd.DataFrame()
    out = []
    names = {**VARIANTS, "sell_original": "SELL: Chartink breakdown rule (short)"}
    for v, g in trades.groupby("variant", sort=False):
        wins, losses = g.loc[g["r_multiple"] > 0, "r_multiple"], g.loc[g["r_multiple"] <= 0, "r_multiple"]
        yr = g.assign(year=pd.to_datetime(g["exit_date"]).dt.year).groupby("year")["r_multiple"].mean()
        avg_r = g["r_multiple"].mean()
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
        if len(g) < min_trades:
            verdict = "Too few trades"
        elif avg_r > 0.1 and pf > 1.3 and (yr > 0).mean() >= 0.6:
            verdict = "Passes"
        elif avg_r > 0:
            verdict = "Marginal"
        else:
            verdict = "Fails"
        out.append({"Variant": v, "Description": names.get(v, v), "Trades": len(g),
                    "Win %": round(100 * len(wins) / len(g), 1), "Avg R": round(avg_r, 3),
                    "Profit factor": round(pf, 2), "Avg hold (days)": round(g["hold_days"].mean(), 1),
                    "Positive years %": round(100 * (yr > 0).mean(), 0), "Years": len(yr), "Verdict": verdict})
    order = list(VARIANTS) + ["sell_original"]
    return pd.DataFrame(out).set_index("Variant").reindex([v for v in order if v in set(trades["variant"])]) \
                            .reset_index()


def lab_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    """Average R per variant per year (pivot)."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.assign(year=pd.to_datetime(trades["exit_date"]).dt.year)
    return t.pivot_table(index="year", columns="variant", values="r_multiple", aggfunc="mean").round(3)
