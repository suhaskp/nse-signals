"""Does buying "today's strongest movers" make money? Two kinds of evidence.

1. Historical proxy (daily bars; intraday history is not available for free):
   a "mover day" = close >= ``min_gain`` above the previous close on >= ``min_volume`` x the prior 20-day
   average volume, with enough liquidity. Entry at the NEXT open (you only know after the close), exit at the
   close 1, 5 and 10 sessions later, round-trip costs deducted. Compared with the Nifty over the same window
   and with an average liquid stock over the same windows (the "typical stock" baseline).

2. Live follow-up: the dashboard's real intraday movers list is recorded at fixed checkpoints (12:00 and
   15:15 IST) with the price shown at that moment, then followed: did the gain hold to that day's close, and
   what happened 1, 5 and 10 sessions later, versus the Nifty.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10)
CHECKPOINTS = (("midday", dtime(12, 0)), ("late", dtime(15, 15)))
LOG_COLUMNS = ["date", "checkpoint", "recorded_at", "Ticker", "entry_price", "today_pct", "volume_pace"]


# --------------------------------------------------------------------------- historical proxy
def historical_movers(histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None,
                      min_gain: float = 0.03, min_volume: float = 1.5, min_turnover_cr: float = 10.0,
                      cost_pct: float = 0.0025) -> dict:
    """Mover-day follow-through vs the Nifty and vs a typical stock, from daily bars."""
    idx_fwd = {}
    if benchmark is not None:
        bo, bc = benchmark["Open"], benchmark["Close"]
        idx_fwd = {h: bc.shift(-h) / bo.shift(-1) - 1 for h in HORIZONS}
    movers, typical = [], []
    for sym, df in histories.items():
        if len(df) < 60:
            continue
        c, o, v = df["Close"], df["Open"], df["Volume"]
        chg = c / c.shift(1) - 1
        pace = v / v.shift(1).rolling(20, min_periods=20).mean()
        liquid = (c * v).rolling(20, min_periods=20).mean() / 1e7 >= min_turnover_cr
        fwd = pd.DataFrame({f"r{h}": c.shift(-h) / o.shift(-1) - 1 - cost_pct for h in HORIZONS}, index=df.index)
        for h in HORIZONS:
            fwd[f"n{h}"] = idx_fwd[h].reindex(df.index) if idx_fwd else np.nan
        fwd["year"] = df.index.year
        sig = (chg >= min_gain) & (pace >= min_volume) & liquid
        movers.append(fwd[sig].assign(Ticker=sym))
        typical.append(fwd[liquid].iloc[::5])  # every 5th day keeps the baseline manageable
    if not movers:
        return {"summary": pd.DataFrame(), "by_year": pd.DataFrame(), "n": 0}
    m = pd.concat(movers).dropna(subset=["r1"]).reset_index(drop=True)  # many stocks share dates: use row positions
    t = pd.concat(typical).dropna(subset=["r1"]).reset_index(drop=True)
    rows = []
    for h in HORIZONS:
        r = m[f"r{h}"].dropna()
        n_ = m.loc[r.index, f"n{h}"]
        base = t[f"r{h}"].dropna()
        rows.append({"Hold": f"{h} session{'s' if h > 1 else ''}", "Mover trades": len(r),
                     "Avg return %": round(100 * r.mean(), 2), "Median %": round(100 * r.median(), 2),
                     "Win %": round(100 * (r > 0).mean(), 1),
                     "Beat Nifty %": round(100 * (r > n_).mean(), 1) if n_.notna().any() else np.nan,
                     "Avg vs Nifty %": round(100 * (r - n_).mean(), 2) if n_.notna().any() else np.nan,
                     "Typical stock avg %": round(100 * base.mean(), 2),
                     "Movers minus typical %": round(100 * (r.mean() - base.mean()), 2)})
    by_year = (m.groupby("year")["r5"].agg(["size", "mean"]).rename(columns={"size": "Mover trades", "mean": "Avg 5-day %"}))
    by_year["Avg 5-day %"] = (100 * by_year["Avg 5-day %"]).round(2)
    return {"summary": pd.DataFrame(rows), "by_year": by_year.reset_index().rename(columns={"year": "Year"}),
            "n": int(len(m)), "rule": f"up >= {100 * min_gain:.0f}% on >= {min_volume:g}x volume, liquid; "
                                      f"buy next open; costs {100 * cost_pct:.2f}% deducted"}


def verdict(proxy: dict) -> tuple[str, str]:
    """Plain-language reading of the historical proxy."""
    s = proxy.get("summary")
    if s is None or s.empty:
        return "none", "Not enough history to test the movers idea."
    r5 = s[s["Hold"] == "5 sessions"].iloc[0]
    good = r5["Movers minus typical %"] > 0.1 and r5["Avg return %"] > 0
    text = (f"Historically ({proxy['n']:,} mover days on this universe), buying the next morning and holding 5 "
            f"sessions returned {r5['Avg return %']:+.2f}% on average after costs, won {r5['Win %']:.0f}% of the time, "
            f"and did {r5['Movers minus typical %']:+.2f}% versus simply holding a typical stock over the same days.")
    if good:
        return "ok", text + " Movers did better than a typical stock, but check the year-by-year row for consistency."
    return "bad", text + " In other words, chasing movers has not beaten just holding an average stock here."


# --------------------------------------------------------------------------- live log
def record_movers(movers: pd.DataFrame | None, now: datetime, path: Path) -> bool:
    """Append the movers list once per checkpoint per day. Returns True if something was recorded."""
    if movers is None or movers.empty:
        return False
    due = [name for name, t in CHECKPOINTS if now.time() >= t]
    if not due:
        return False
    name = due[-1]
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    today = now.date().isoformat()
    if ((log["date"] == today) & (log["checkpoint"] == name)).any():
        return False
    rows = pd.DataFrame({"date": today, "checkpoint": name, "recorded_at": now.strftime("%H:%M"),
                         "Ticker": movers["Ticker"], "entry_price": movers["Price"],
                         "today_pct": movers["Today %"], "volume_pace": movers["Volume pace"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    out = rows if log.empty else pd.concat([log, rows], ignore_index=True)
    out.to_csv(path, index=False)
    return True


def follow_up(path: Path, histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None) -> dict:
    """Outcome of every recorded mover: held to the close? and 1/5/10 sessions later vs the Nifty."""
    if not path.exists():
        return {"rows": pd.DataFrame(), "summary": pd.DataFrame(), "days": 0}
    log = pd.read_csv(path)
    out = []
    for _, r in log.iterrows():
        df = histories.get(r["Ticker"])
        d = pd.Timestamp(r["date"])
        if df is None or d not in df.index:
            continue
        i = df.index.get_loc(d)
        entry = float(r["entry_price"])
        row = {**r.to_dict(), "Held to close %": round(100 * (df["Close"].iloc[i] / entry - 1), 2)}
        for h in HORIZONS:
            if i + h < len(df):
                ret = df["Close"].iloc[i + h] / entry - 1
                row[f"{h}d %"] = round(100 * ret, 2)
                if benchmark is not None and d in benchmark.index:
                    bi = benchmark.index.get_loc(d)
                    if bi + h < len(benchmark):
                        # Nifty from the same day's close (the mover's entry was intraday that day)
                        nret = benchmark["Close"].iloc[bi + h] / benchmark["Close"].iloc[bi] - 1
                        row[f"{h}d vs Nifty %"] = round(100 * (ret - nret), 2)
            else:
                row[f"{h}d %"] = np.nan
        out.append(row)
    rows = pd.DataFrame(out)
    if rows.empty:
        return {"rows": rows, "summary": pd.DataFrame(), "days": int(log["date"].nunique())}
    summ = []
    for label, col in [("Same day (to the close)", "Held to close %")] + [(f"{h} session{'s' if h > 1 else ''}", f"{h}d %")
                                                                         for h in HORIZONS]:
        x = rows[col].dropna() if col in rows else pd.Series(dtype=float)
        vs = rows.get(col.replace(" %", " vs Nifty %")) if col != "Held to close %" else None
        summ.append({"Horizon": label, "Movers followed": len(x),
                     "Avg %": round(x.mean(), 2) if len(x) else np.nan,
                     "Win %": round(100 * (x > 0).mean(), 1) if len(x) else np.nan,
                     "Avg vs Nifty %": round(vs.dropna().mean(), 2) if vs is not None and vs.notna().any() else np.nan})
    return {"rows": rows, "summary": pd.DataFrame(summ), "days": int(log["date"].nunique())}
