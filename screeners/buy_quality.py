"""Quality-filtered BUY signals: breakout quality + volume + trend + ATR stop + regime +
next-day entry + minimum 2:1 reward/risk.

Signal on the close of day t (all computed from data up to that close):

  breakout        close > prior N-day high (the breakout level L)
  close strength  close in the upper part of the day's range and above the open
  not extended    close no more than ``max_extension_atr`` ATR beyond L
  volume          volume >= ``volume_multiple`` x the prior ``volume_window``-day average
  liquidity       20-day average turnover >= ``min_turnover_cr`` crore
  trend           close > SMA50 > SMA200, SMA200 rising, and 3-month return above the Nifty's
  regime          Nifty above its 200-day average (and optional breadth floor)
  ATR stop        stop = L - buffer x ATR, at least ``min_stop_atr`` and at most ``max_stop_atr``
                  ATR from the entry and at most ``max_stop_pct`` of the price
  reward/risk     target = measured move (L + base height), capped at the prior 52-week high
                  when that resistance is overhead and at ``target_max_atr`` ATR from the close
                  (what a stock typically moves over the holding period);
                  (target - entry) / (entry - stop) >= 2

Entry on day t+1 (``plan_entry``): only at the open, only if the open is at or above L and at
most ``entry_max_gap_pct`` above the signal close; the stop distance and reward/risk are
recomputed from the actual open and must still pass.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from config.config import BuyQualityConfig
from screeners.screener import atr as atr_fn

GATES = ["breakout", "close_strength", "not_extended", "volume", "liquidity", "trend", "regime", "stop", "reward_risk"]
GATE_LABELS = {"breakout": "breakout", "close_strength": "strong close", "not_extended": "not extended",
               "volume": "volume", "liquidity": "liquidity", "trend": "trend", "regime": "market regime",
               "stop": "ATR stop", "reward_risk": "reward/risk >= 2", "model": "model score"}


def regime_state(benchmark: pd.DataFrame | None, breadth: pd.Series | None, cfg: BuyQualityConfig,
                 index: pd.DatetimeIndex) -> pd.Series:
    """'on' (Nifty above its 200-day), 'neutral' (within ``neutral_band_pct`` below it, or breadth up at least
    ``neutral_breadth_gain`` over 20 sessions) or 'off', per date, known at that close."""
    if benchmark is None:
        return pd.Series("on", index=index)
    bc = benchmark["Close"]
    sma = bc.rolling(200, min_periods=200).mean()
    on = bc > sma
    near = bc >= sma * (1 - cfg.neutral_band_pct / 100)
    recovering = (breadth - breadth.shift(20) >= cfg.neutral_breadth_gain).reindex(bc.index).fillna(False) \
        if breadth is not None else pd.Series(False, index=bc.index)
    state = pd.Series(np.where(on, "on", np.where(near | recovering, "neutral", "off")), index=bc.index)
    state[sma.isna()] = "off"
    return state.reindex(index).ffill().fillna("off")


def regime_series(benchmark: pd.DataFrame | None, breadth: pd.Series | None, cfg: BuyQualityConfig,
                  index: pd.DatetimeIndex) -> pd.Series:
    """True on dates where new BUYs are allowed by the market regime."""
    if not cfg.regime_required:
        return pd.Series(True, index=index)
    if benchmark is None:
        return pd.Series(True, index=index)
    state = regime_state(benchmark, breadth, cfg, index)
    ok = state.isin(["on", "neutral"]) if cfg.regime_mode == "three_state" else state.eq("on")
    if cfg.min_breadth > 0 and breadth is not None:
        ok &= (breadth.reindex(index).ffill() >= cfg.min_breadth).fillna(False)
    return ok.astype(bool)


def quality_frame(df: pd.DataFrame, cfg: BuyQualityConfig, benchmark: pd.DataFrame | None = None,
                  breadth: pd.Series | None = None) -> pd.DataFrame:
    """Per-bar gate results, levels and the combined ``buy`` flag for one stock."""
    c, h, lo, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    a = atr_fn(h, lo, c, cfg.atr_period)
    level = h.shift(1).rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max()
    base_low = lo.shift(1).rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).min()
    resistance = h.shift(1).rolling(cfg.resistance_lookback, min_periods=cfg.resistance_lookback // 2).max()
    sma50, sma200 = c.rolling(50, min_periods=50).mean(), c.rolling(200, min_periods=200).mean()
    vol_avg = v.shift(1).rolling(cfg.volume_window, min_periods=cfg.volume_window).mean()
    turnover = (c * v).rolling(20, min_periods=20).mean() / 1e7
    rng = h - lo
    cloc = ((c - lo) / rng).where(rng > 0, 0.5)
    if benchmark is not None and cfg.require_relative_strength:
        bc = benchmark["Close"].reindex(df.index).ffill()
        rs = (c / c.shift(63) - 1) - (bc / bc.shift(63) - 1)
        rs_ok = rs > 0
    else:
        rs, rs_ok = pd.Series(np.nan, index=df.index), pd.Series(True, index=df.index)

    dist = (c - (level - cfg.stop_buffer_atr * a)).clip(lower=cfg.min_stop_atr * a)
    stop = c - dist
    measured = level + (level - base_low)
    overhead = resistance > c * 1.001
    target = measured.where(~overhead, np.minimum(measured, resistance))
    target = np.minimum(target, c + cfg.target_max_atr * a)
    rr = (target - c) / dist

    out = pd.DataFrame({
        "close": c, "level": level, "atr": a, "stop": stop, "target": target, "rr": rr,
        "volume_ratio": v / vol_avg, "close_location": cloc, "extension_atr": (c - level) / a,
        "turnover_cr": turnover, "rs_63": rs, "resistance": resistance, "sma50": sma50,
        "breakout": c > level,
        "close_strength": (cloc >= cfg.min_close_location) & (c > o),
        "not_extended": (c - level) / a <= cfg.max_extension_atr,
        "volume": v / vol_avg >= cfg.volume_multiple,
        "liquidity": turnover >= cfg.min_turnover_cr,
        "trend": (c > sma50) & (sma50 > sma200) & (sma200 > sma200.shift(cfg.sma200_slope_days)) & rs_ok,
        "regime": regime_series(benchmark, breadth, cfg, df.index),
        "regime_state": regime_state(benchmark, breadth, cfg, df.index),
        "stop": stop,
    }, index=df.index)
    out["stop_ok"] = (dist <= cfg.max_stop_atr * a) & (dist / c <= cfg.max_stop_pct) & (a > 0)
    out["reward_risk"] = rr >= cfg.min_reward_risk
    gates = out[["breakout", "close_strength", "not_extended", "volume", "liquidity", "trend", "regime",
                 "stop_ok", "reward_risk"]].fillna(False).astype(bool)
    out[gates.columns] = gates
    out["buy"] = gates.all(axis=1)
    return out


READINESS_WEIGHTS = {"breakout": 20, "close_strength": 10, "not_extended": 10, "volume": 15, "liquidity": 5,
                     "trend": 20, "stop_ok": 10, "reward_risk": 10}


def readiness(r: pd.Series, cfg: BuyQualityConfig) -> float:
    """Setup readiness 0-100: how complete the BUY setup is, ignoring the market regime.

    Partial credit for being close to a breakout (full at the level, none 5% below) and for volume building
    towards the requirement. It measures completeness, NOT the probability of a profit.
    """
    score = 0.0
    for g, w in READINESS_WEIGHTS.items():
        if bool(r.get(g, False)):
            score += w
        elif g == "breakout" and r["close"] > 0 and np.isfinite(r["level"]):
            below = r["level"] / r["close"] - 1
            score += w * max(0.0, 1 - below / 0.05)
        elif g == "volume" and np.isfinite(r["volume_ratio"]):
            score += w * min(1.0, max(0.0, r["volume_ratio"] / cfg.volume_multiple))
        elif g == "reward_risk" and not bool(r.get("breakout", False)):
            score += w * 0.5  # cannot be judged until there is a breakout to measure from
    return round(score, 0)


def setup_state(r: pd.Series, rd: float) -> str:
    """BUY NOW / WATCH CLOSELY / WAIT / IGNORE from the gates and readiness."""
    if bool(r["buy"]):
        return "BUY NOW"
    if not bool(r.get("trend", False)):
        return "IGNORE"
    if rd >= 85:
        return "WATCH CLOSELY"
    if rd >= 60:
        return "WAIT"
    return "IGNORE"


def breakout_status(r: pd.Series, cfg: BuyQualityConfig) -> str:
    """Where price sits relative to its breakout level, with a don't-chase warning when extended."""
    if not (np.isfinite(r["level"]) and r["close"] > 0):
        return ""
    d = 100 * (r["close"] / r["level"] - 1)
    if d >= 0:
        if r["extension_atr"] > cfg.max_extension_atr:
            return f"⚠️ {d:+.1f}% above breakout: extended, don't chase"
        return f"✅ {d:+.1f}% above breakout"
    if d >= -1:
        return f"🟢 {d:.1f}%: very close"
    if d >= -3:
        return f"🟡 {d:.1f}%: approaching"
    return f"⚪ {d:.1f}%: early"


def _gate_cols() -> list[str]:
    return ["breakout", "close_strength", "not_extended", "volume", "liquidity", "trend", "regime", "stop_ok",
            "reward_risk"]


def evaluate_latest(histories: Mapping[str, pd.DataFrame], cfg: BuyQualityConfig,
                    benchmark: pd.DataFrame | None = None, breadth: pd.Series | None = None) -> pd.DataFrame:
    """One row per stock for its latest bar: every gate, the levels, and what failed."""
    rows = []
    for sym, df in histories.items():
        if len(df) < max(cfg.breakout_lookback, 200) + 25:
            continue
        r = quality_frame(df, cfg, benchmark, breadth).iloc[-1]
        failed = [GATE_LABELS["stop" if g == "stop_ok" else g] for g in _gate_cols() if not r[g]]
        rd = readiness(r, cfg)
        rows.append({"Ticker": sym, "Session": df.index[-1].date().isoformat(), "passes": bool(r["buy"]),
                     "Readiness": rd, "State": setup_state(r, rd), "Breakout status": breakout_status(r, cfg),
                     "To breakout %": round(100 * (r["level"] / r["close"] - 1), 2) if r["close"] > 0 else np.nan,
                     "Regime state": str(r["regime_state"]), "SMA50": round(float(r["sma50"]), 2),
                     "Close": round(float(r["close"]), 2), "Breakout level": round(float(r["level"]), 2),
                     "Stop": round(float(r["stop"]), 2), "Target": round(float(r["target"]), 2),
                     "R:R": round(float(r["rr"]), 2) if np.isfinite(r["rr"]) else np.nan,
                     "ATR": round(float(r["atr"]), 2), "Volume x avg": round(float(r["volume_ratio"]), 2),
                     "Close location": round(float(r["close_location"]), 2),
                     "Extension (ATR)": round(float(r["extension_atr"]), 2),
                     "RS vs Nifty 3M %": round(100 * float(r["rs_63"]), 1) if np.isfinite(r["rs_63"]) else np.nan,
                     "Target capped by 52w high": bool(r["resistance"] > r["close"] * 1.001),
                     "Target at ATR cap": bool(abs(r["target"] - (r["close"] + cfg.target_max_atr * r["atr"])) < 1e-6
                                               * max(r["close"], 1)),
                     **{f"g_{g}": bool(r[g]) for g in _gate_cols()},
                     "Failed": ", ".join(failed)})
    return pd.DataFrame(rows)


def plan_entry(level: float, signal_close: float, atr_value: float, target: float, open_price: float,
               cfg: BuyQualityConfig) -> dict:
    """Next-day entry decision at ``open_price``; recomputes stop and reward/risk from the actual open."""
    out = {"entry": open_price, "stop": np.nan, "target": target, "rr": np.nan}
    if not (np.isfinite(open_price) and open_price > 0 and atr_value > 0):
        return {**out, "status": "No price yet"}
    if open_price < level:
        return {**out, "status": f"SKIP: opened below the breakout level ₹{level:,.2f} (failed breakout)"}
    gap = 100 * (open_price / signal_close - 1)
    if gap > cfg.entry_max_gap_pct:
        return {**out, "status": f"SKIP: opened {gap:.1f}% above the signal close (chasing)"}
    dist = max(open_price - (level - cfg.stop_buffer_atr * atr_value), cfg.min_stop_atr * atr_value)
    if dist > cfg.max_stop_atr * atr_value or dist / open_price > cfg.max_stop_pct:
        return {**out, "status": "SKIP: stop would be too wide from this open"}
    rr = (target - open_price) / dist
    stop = open_price - dist
    if rr < cfg.min_reward_risk:
        return {**out, "stop": stop, "rr": rr, "status": f"SKIP: reward/risk {rr:.1f} below {cfg.min_reward_risk:.0f}:1"}
    return {"entry": open_price, "stop": stop, "target": target, "rr": rr, "status": "OK"}


def backtest_quality(histories: Mapping[str, pd.DataFrame], cfg: BuyQualityConfig, benchmark: pd.DataFrame | None,
                     cost_pct: float, breadth: pd.Series | None = None, delay: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trades (next-open entry via ``plan_entry``; stop, target or time exit) and the gate funnel."""
    trades, funnel = [], {g: 0 for g in ["raw breakouts", *[GATE_LABELS["stop" if g == "stop_ok" else g]
                                                            for g in _gate_cols()[1:]], "next-day entry OK"]}
    labels = list(funnel)
    for sym, df in histories.items():
        if len(df) < max(cfg.breakout_lookback, 200) + 30:
            continue
        f = quality_frame(df, cfg, benchmark, breadth)
        gates = f[_gate_cols()]
        alive = gates["breakout"]
        funnel["raw breakouts"] += int(alive.sum())
        for g, lab in zip(_gate_cols()[1:], labels[1:-1]):
            alive = alive & gates[g]
            funnel[lab] += int(alive.sum())
        trades += _simulate_mask(sym, df, f, f["buy"], cfg, cost_pct, funnel, delay)
    tr = pd.DataFrame(trades)
    closed = tr[tr["outcome"] != "open"].reset_index(drop=True) if not tr.empty else tr
    open_ = tr[tr["outcome"] == "open"].reset_index(drop=True) if not tr.empty else tr
    fun = pd.DataFrame({"Stage": list(funnel), "Signals remaining": list(funnel.values())})
    fun.attrs["open_trades"] = open_
    return closed, fun


def _simulate_mask(sym: str, df: pd.DataFrame, f: pd.DataFrame, mask: pd.Series, cfg: BuyQualityConfig,
                   cost_pct: float, funnel: dict | None = None, delay: int = 0) -> list[dict]:
    """Next-open entry via ``plan_entry`` (``delay`` extra sessions late, for robustness tests);
    stop / target / time exit. One open trade per stock."""
    trades = []
    o, h, lo, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    busy = -1
    for t in np.flatnonzero(mask.fillna(False).to_numpy()):
        e = t + 1 + delay
        if e >= len(df) or t <= busy:
            continue
        r = f.iloc[t]
        p = plan_entry(r["level"], r["close"], r["atr"], r["target"], o[e], cfg)
        if p["status"] != "OK" or not (np.isfinite(r["level"]) and np.isfinite(r["atr"])):
            continue
        if funnel is not None:
            funnel["next-day entry OK"] += 1
        entry, stop, target = p["entry"], p["stop"], p["target"]
        risk = entry - stop
        x = min(e + cfg.max_hold_days - 1, len(df) - 1)
        exit_px, outcome = None, "time"
        for i in range(e, x + 1):
            if i > e and o[i] <= stop:
                exit_px, outcome, x = o[i], "gap_stop", i
                break
            if lo[i] <= stop:
                exit_px, outcome, x = stop, "stop", i
                break
            if h[i] >= target:
                exit_px, outcome, x = (max(o[i], target) if i > e else target), "target", i
                break
        if exit_px is None:
            exit_px = c[x]
            if x < e + cfg.max_hold_days - 1:
                outcome = "open"
        trades.append({"Ticker": sym, "variant": "quality", "side": "long",
                       "signal_date": df.index[t].date(), "entry_date": df.index[e].date(),
                       "exit_date": df.index[x].date(), "entry": entry, "stop": stop, "target": target,
                       "planned_rr": p["rr"], "exit": exit_px, "outcome": outcome,
                       "r_multiple": (exit_px - entry - cost_pct * entry) / risk,
                       "return_pct": 100 * ((exit_px - entry) / entry - cost_pct), "hold_days": x - e + 1,
                       "regime_state": str(r["regime_state"])})
        busy = x
    return trades


def _stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"Trades": 0, "Win %": np.nan, "Avg R": np.nan, "Profit factor": np.nan, "Max DD (R)": np.nan}
    r = tr.sort_values("exit_date")["r_multiple"].to_numpy(float)
    cum = np.cumsum(r)
    dd = float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min())
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    return {"Trades": len(r), "Win %": round(100 * (r > 0).mean(), 1), "Avg R": round(r.mean(), 3),
            "Profit factor": round(wins / losses, 2) if losses > 0 else np.inf, "Max DD (R)": round(dd, 1)}


ABLATION_ORDER = [("Breakout only", "breakout"), ("+ strong close", "close_strength"), ("+ not extended", "not_extended"),
                  ("+ volume", "volume"), ("+ liquidity", "liquidity"), ("+ trend (50/200-day, RS)", "trend"),
                  ("+ market regime", "regime"), ("+ ATR stop limits", "stop_ok"), ("+ reward/risk >= 2", "reward_risk")]


def filter_ablation(histories: Mapping[str, pd.DataFrame], cfg: BuyQualityConfig, benchmark: pd.DataFrame | None,
                    cost_pct: float, breadth: pd.Series | None = None) -> pd.DataFrame:
    """Add the checks one at a time and backtest each stage: does each check improve results or only cut trades?"""
    per_stage: dict[str, list] = {name: [] for name, _ in ABLATION_ORDER}
    for sym, df in histories.items():
        if len(df) < max(cfg.breakout_lookback, 200) + 30:
            continue
        f = quality_frame(df, cfg, benchmark, breadth)
        mask = pd.Series(True, index=df.index)
        for name, gate in ABLATION_ORDER:
            mask = mask & f[gate]
            per_stage[name] += _simulate_mask(sym, df, f, mask, cfg, cost_pct)
    rows, prev = [], None
    for name, _ in ABLATION_ORDER:
        tr = pd.DataFrame(per_stage[name])
        tr = tr[tr["outcome"] != "open"] if not tr.empty else tr
        st_ = {"Stage": name, **_stats(tr)}
        if prev is None or not np.isfinite(st_["Avg R"]) or not np.isfinite(prev):
            st_["Effect"] = "baseline" if prev is None else "no trades left"
        else:
            d = st_["Avg R"] - prev
            st_["Effect"] = ("improves results" if d >= 0.03 else "hurts results" if d <= -0.03
                             else "mostly just cuts trades")
        prev = st_["Avg R"] if np.isfinite(st_["Avg R"]) else prev
        rows.append(st_)
    return pd.DataFrame(rows)


def regime_breakdown(trades_no_regime: pd.DataFrame) -> pd.DataFrame:
    """Results of the BUY rules (run WITHOUT the regime check) split by the market regime at the signal."""
    if trades_no_regime is None or trades_no_regime.empty or "regime_state" not in trades_no_regime:
        return pd.DataFrame()
    names = {"on": "Risk-on (Nifty above 200-day)", "neutral": "Neutral (near 200-day or breadth recovering)",
             "off": "Risk-off (well below 200-day)"}
    rows = [{"Regime at signal": names.get(k, k), **_stats(g)} for k, g in trades_no_regime.groupby("regime_state")]
    order = {v: i for i, v in enumerate(names.values())}
    return pd.DataFrame(rows).sort_values("Regime at signal", key=lambda s: s.map(order)).reset_index(drop=True)
