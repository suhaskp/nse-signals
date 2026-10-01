"""Market research: regime, breadth, sector strength and per-stock profiles.

All figures use completed daily bars only, so the same functions can build
historical features without lookahead.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from screeners.screener import rsi


def close_panel(histories: Mapping[str, pd.DataFrame], field: str = "Close") -> pd.DataFrame:
    """Dates x symbols matrix of one OHLCV field."""
    return pd.DataFrame({s: df[field] for s, df in histories.items()}).sort_index()


def regime_series(benchmark: pd.DataFrame) -> pd.Series:
    """True on dates when the index closed above its 200-day average (known at that close)."""
    c = benchmark["Close"]
    return (c > c.rolling(200, min_periods=200).mean()).rename("risk_on")


def market_regime(benchmark: pd.DataFrame) -> dict:
    """Snapshot of the index: trend state, returns, drawdown and volatility regime."""
    c = benchmark["Close"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    last, s50, s200 = c.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]
    if last > s200 and s50 > s200:
        state, label = "bull", "Uptrend: index above its 200-day average and the 50-day above the 200-day"
    elif last < s200 and s50 < s200:
        state, label = "bear", "Downtrend: index below its 200-day average and the 50-day below the 200-day"
    else:
        state, label = "neutral", "Mixed: the index and its averages disagree; trend is unclear"
    vol = np.log(c).diff().rolling(21).std() * np.sqrt(252)
    vol_hist = vol.dropna().tail(756)
    return {
        "close": float(last), "sma50": float(s50), "sma200": float(s200), "state": state, "label": label,
        "ret_1m": float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) > 22 else np.nan,
        "ret_3m": float(c.iloc[-1] / c.iloc[-64] - 1) if len(c) > 64 else np.nan,
        "ret_1y": float(c.iloc[-1] / c.iloc[-253] - 1) if len(c) > 253 else np.nan,
        "drawdown": float(last / c.tail(252).max() - 1),
        "vol_21d": float(vol.iloc[-1]),
        "vol_percentile": float((vol_hist < vol.iloc[-1]).mean()) if len(vol_hist) else np.nan,
        "risk_on": bool(last > s200),
    }


def breadth_series(histories: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-date share of stocks above their 50/200-day averages (a market-health gauge)."""
    closes = close_panel(histories)
    above50 = (closes > closes.rolling(50, min_periods=50).mean()).astype(float).where(closes.notna())
    above200 = (closes > closes.rolling(200, min_periods=200).mean()).astype(float).where(closes.notna())
    return pd.DataFrame({"pct_above_50": above50.mean(axis=1), "pct_above_200": above200.mean(axis=1)})


def market_breadth(histories: Mapping[str, pd.DataFrame]) -> dict:
    """Latest breadth: % above averages, new 52-week highs/lows, advancers vs decliners."""
    closes = close_panel(histories)
    highs, lows = close_panel(histories, "High"), close_panel(histories, "Low")
    last = closes.iloc[-1]
    bser = breadth_series(histories)
    bs = bser.iloc[-1]
    ago = lambda n, col: float(bser[col].iloc[-1 - n]) if len(bser) > n else np.nan  # noqa: E731
    chg = closes.iloc[-1] / closes.iloc[-2] - 1
    return {
        "n": int(last.notna().sum()),
        "pct_above_50": float(bs["pct_above_50"]), "pct_above_200": float(bs["pct_above_200"]),
        "pct_above_200_5d": ago(5, "pct_above_200"), "pct_above_200_20d": ago(20, "pct_above_200"),
        "pct_above_50_5d": ago(5, "pct_above_50"), "pct_above_50_20d": ago(20, "pct_above_50"),
        "new_highs": int((highs.iloc[-1] >= highs.tail(252).max()).sum()),
        "new_lows": int((lows.iloc[-1] <= lows.tail(252).min()).sum()),
        "advancers": int((chg > 0).sum()), "decliners": int((chg < 0).sum()),
    }


def sector_strength(histories: Mapping[str, pd.DataFrame], sectors: Mapping[str, str]) -> pd.DataFrame:
    """Sectors ranked by median 3-month return, with breadth and 1-month change."""
    closes = close_panel(histories)
    r1 = closes.iloc[-1] / closes.iloc[-22] - 1
    r3 = closes.iloc[-1] / closes.iloc[-64] - 1
    above200 = closes.iloc[-1] > closes.rolling(200).mean().iloc[-1]
    df = pd.DataFrame({"sector": [sectors.get(s, "Unknown") for s in closes.columns],
                       "r1": r1.values, "r3": r3.values, "above200": above200.values})
    out = df.groupby("sector").agg(Stocks=("r3", "size"), **{"3M median %": ("r3", "median"),
                                   "1M median %": ("r1", "median"), "% above 200DMA": ("above200", "mean")})
    out[["3M median %", "1M median %", "% above 200DMA"]] *= 100
    out = out.sort_values("3M median %", ascending=False).round(1)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out.reset_index().rename(columns={"sector": "Sector"})


def stock_profile(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict:
    """Research card figures for one stock."""
    c = df["Close"]

    def ret(n: int) -> float:
        return float(c.iloc[-1] / c.iloc[-n - 1] - 1) if len(c) > n else np.nan

    out = {"close": float(c.iloc[-1]), "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_1y": ret(252),
           "from_52w_high": float(c.iloc[-1] / df["High"].tail(252).max() - 1),
           "from_52w_low": float(c.iloc[-1] / df["Low"].tail(252).min() - 1),
           "vol_ann": float(np.log(c).diff().tail(63).std() * np.sqrt(252)),
           "max_dd_1y": float((c.tail(252) / c.tail(252).cummax() - 1).min()),
           "rsi": float(rsi(c, 14).iloc[-1]),
           "above_50dma": bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1]),
           "above_200dma": bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])}
    if benchmark is not None and len(benchmark) > 64:
        b = benchmark["Close"].reindex(c.index).ffill()
        out["vs_index_3m"] = out["ret_3m"] - float(b.iloc[-1] / b.iloc[-64] - 1)
    return out


def trend_leaders(histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None,
                  quality: pd.DataFrame | None = None, scores: pd.DataFrame | None = None,
                  sectors: Mapping[str, str] | None = None, min_turnover_cr: float = 10.0,
                  max_from_high: float = 0.05, n: int = 15) -> pd.DataFrame:
    """Stocks in their own strong uptrend: close > 50-day > rising 200-day, within ``max_from_high`` of the
    52-week high, beating the Nifty over 3 months, and liquid. Ranked by 3-month strength vs the Nifty.

    ``Still needs`` lists the BUY checks each one fails (market regime reported separately), so it is clear
    how close each is to a real BUY signal. This is a research list, not a signal.
    """
    bench_ret3 = np.nan
    if benchmark is not None and len(benchmark) > 64:
        bc = benchmark["Close"]
        bench_ret3 = float(bc.iloc[-1] / bc.iloc[-64] - 1)
    q = quality.set_index("Ticker") if quality is not None and not quality.empty else None
    rows = []
    for sym, df in histories.items():
        if len(df) < 260:
            continue
        c = df["Close"]
        last = float(c.iloc[-1])
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        hi = float(df["High"].tail(252).max())
        turnover = float((c * df["Volume"]).tail(20).mean() / 1e7)
        r1, r3 = last / float(c.iloc[-22]) - 1, last / float(c.iloc[-64]) - 1
        rs = r3 - bench_ret3 if np.isfinite(bench_ret3) else r3
        from_high = last / hi - 1
        if not (last > sma50.iloc[-1] > sma200.iloc[-1] and sma200.iloc[-1] > sma200.iloc[-21]
                and from_high >= -max_from_high and rs > 0 and turnover >= min_turnover_cr):
            continue
        level = float(df["High"].iloc[-56:-1].max())
        failed = ""
        if q is not None and sym in q.index:
            failed = ", ".join(f for f in str(q.at[sym, "Failed"]).split(", ") if f and f != "market regime")
        rows.append({"Ticker": sym, "Sector": (sectors or {}).get(sym, ""), "Price": round(last, 2),
                     "1M %": round(100 * r1, 1), "3M %": round(100 * r3, 1), "vs Nifty 3M %": round(100 * rs, 1),
                     "From 52w high %": round(100 * from_high, 1), "Breakout level": round(level, 2),
                     "To breakout %": round(100 * (level / last - 1), 1),
                     "Rank": round(100 * float(scores.at[sym, "score"]), 0)
                     if scores is not None and not scores.empty and sym in scores.index else np.nan,
                     "Still needs": failed or "nothing except the market regime"})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_regime_only"] = out["Still needs"] == "nothing except the market regime"
    out = out.sort_values(["_regime_only", "vs Nifty 3M %"], ascending=[False, False]).head(n)
    return out.reset_index(drop=True)


REGIME_EXPOSURE = {"Risk-on": "100% of normal size", "Risk-on, weakening": "50-75% of normal size",
                   "Transition": "25-50% of normal size", "Risk-off, improving": "0-25% of normal size",
                   "Risk-off": "0%: no new positions"}


def regime_detail(benchmark: pd.DataFrame, histories: Mapping[str, pd.DataFrame]) -> dict:
    """Five-state market regime with a confidence level and early-recovery detection.

    Eight components, each bullish or bearish: index vs 200-day, 50-day vs 200-day, 200-day slope, breadth level,
    breadth 20-day trend, share above the 50-day (10-day trend), new 52-week highs vs lows (5 days), and whether the
    index is making a higher low. Structure (the first two) sets the family; the improving components
    decide 'weakening' / 'improving'. Confidence = share of components agreeing with the state.
    Descriptive only: the trading rules use the regime check configured in buy_quality.
    """
    c = benchmark["Close"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    bser = breadth_series(histories)
    b200, b50 = bser["pct_above_200"], bser["pct_above_50"]
    highs, lows = close_panel(histories, "High"), close_panel(histories, "Low")
    nh = (highs >= highs.rolling(252, min_periods=126).max()).tail(5).sum().sum()
    nl = (lows <= lows.rolling(252, min_periods=126).min()).tail(5).sum().sum()
    comp = {
        "Nifty above its 200-day average": bool(c.iloc[-1] > sma200.iloc[-1]),
        "50-day average above the 200-day": bool(sma50.iloc[-1] > sma200.iloc[-1]),
        "200-day average rising": bool(sma200.iloc[-1] > sma200.iloc[-21]),
        "At least half of stocks above their 200-day": bool(b200.iloc[-1] >= 0.5),
        "Breadth rising over 20 sessions": bool(b200.iloc[-1] > b200.iloc[-21]) if len(b200) > 21 else False,
        "More stocks above their 50-day than 10 sessions ago": bool(b50.iloc[-1] > b50.iloc[-11]) if len(b50) > 11 else False,
        "More new 52-week highs than lows (5 days)": bool(nh > nl),
        "Nifty making a higher low (last 20 vs prior 20 sessions)": bool(c.iloc[-20:].min() > c.iloc[-40:-20].min()),
    }
    vals = list(comp.values())
    structure_bull = vals[0] and vals[1]
    structure_bear = (not vals[0]) and (not vals[1])
    improving = sum(vals[4:])            # breadth trend, 50-day participation, highs vs lows, higher low
    if structure_bull:
        state = "Risk-on" if improving >= 2 else "Risk-on, weakening"
        agree = sum(vals) / len(vals)
    elif structure_bear:
        state = "Risk-off, improving" if improving >= 3 else "Risk-off"
        agree = (sum(not v for v in vals[:4]) + (improving if state.endswith("improving") else 4 - improving)) / 8
    else:
        state, agree = "Transition", 0.5 + abs(sum(vals) / len(vals) - 0.5)
    icon = {"Risk-on": "🟢", "Risk-on, weakening": "🟢", "Transition": "🟡", "Risk-off, improving": "🟡",
            "Risk-off": "🔴"}[state]
    return {"state": state, "icon": icon, "confidence": round(100 * agree), "components": comp,
            "exposure": REGIME_EXPOSURE[state], "breadth_path": [round(100 * b200.iloc[-1 - n]) for n in (15, 10, 5, 0)
                                                                   if len(b200) > n]}
