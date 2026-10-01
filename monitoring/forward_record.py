"""True forward record of BUY signals: saved when issued, stamped with the rule version, then followed.

Unlike a backtest filtered by date, nothing here is recomputed when rules change: each signal keeps the levels
and rule version it was issued with. Outcomes follow the same execution rules as the backtest: entry only at the
next open inside the entry range (``plan_entry``), then stop, target or time exit.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import BuyQualityConfig
from screeners.buy_quality import plan_entry

COLUMNS = ["session", "Ticker", "rule_version", "close", "breakout_level", "atr", "stop", "target", "rr", "qty",
           "risk_amount"]


class ForwardRecord:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, dtype={"session": str}) if self.path.exists() else pd.DataFrame(columns=COLUMNS)

    def record(self, buys: pd.DataFrame, session: date, version: str) -> int:
        """Save BUY signals for ``session`` (once per session and stock). Returns how many were new."""
        if buys is None or buys.empty:
            return 0
        log = self.load()
        seen = set(zip(log["session"], log["Ticker"])) if not log.empty else set()
        rows = [{"session": session.isoformat(), "Ticker": r["Ticker"], "rule_version": version,
                 "close": r["Price"], "breakout_level": r["Breakout level"], "atr": r["ATR"],
                 "stop": r["Stop Loss"], "target": r["Target Price"], "rr": r["R:R"],
                 "qty": int(r.get("Qty", 0) or 0), "risk_amount": float(r.get("Risk (INR)", 0) or 0)}
                for _, r in buys.iterrows() if (session.isoformat(), r["Ticker"]) not in seen]
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows, columns=COLUMNS)
        (new if log.empty else pd.concat([log, new], ignore_index=True)).to_csv(self.path, index=False)
        return len(rows)

    def evaluate(self, histories: Mapping[str, pd.DataFrame], cfg: BuyQualityConfig, cost_pct: float) -> pd.DataFrame:
        """One row per recorded signal with what actually happened."""
        out = []
        for _, s in self.load().iterrows():
            df = histories.get(s["Ticker"])
            base = {**s.to_dict(), "entry_date": None, "entry": np.nan, "exit_date": None, "exit": np.nan,
                    "outcome": "waiting for the next open", "r_multiple": np.nan}
            d = pd.Timestamp(s["session"])
            if df is None or d not in df.index:
                out.append(base)
                continue
            i = df.index.get_loc(d)
            e = i + 1
            if e >= len(df):
                out.append(base)
                continue
            o, h, lo, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
            p = plan_entry(s["breakout_level"], s["close"], s["atr"], s["target"], o[e], cfg)
            if p["status"] != "OK":
                out.append({**base, "entry_date": str(df.index[e].date()), "outcome": "not entered: " + p["status"]})
                continue
            entry, stop, target = p["entry"], p["stop"], p["target"]
            risk = entry - stop
            x_last = min(e + cfg.max_hold_days - 1, len(df) - 1)
            exit_px, outcome, x = None, None, x_last
            for j in range(e, x_last + 1):
                if j > e and o[j] <= stop:
                    exit_px, outcome, x = o[j], "stop (gap)", j
                    break
                if lo[j] <= stop:
                    exit_px, outcome, x = stop, "stop", j
                    break
                if h[j] >= target:
                    exit_px, outcome, x = (max(o[j], target) if j > e else target), "target", j
                    break
            if exit_px is None:
                if x_last == e + cfg.max_hold_days - 1:
                    exit_px, outcome = c[x_last], "time exit"
                else:
                    out.append({**base, "entry_date": str(df.index[e].date()), "entry": entry, "outcome": "open",
                                "r_multiple": (c[-1] - entry) / risk})
                    continue
            out.append({**base, "entry_date": str(df.index[e].date()), "entry": entry, "exit_date": str(df.index[x].date()),
                        "exit": exit_px, "outcome": outcome, "r_multiple": (exit_px - entry - cost_pct * entry) / risk})
        return pd.DataFrame(out)


def held_positions(results: pd.DataFrame, default_risk: float) -> list[tuple[str, float, str]]:
    """(ticker, rupee risk, signal date) for signals still open or waiting for their entry."""
    if results is None or results.empty:
        return []
    live = results[results["outcome"].isin(["open", "waiting for the next open"])]
    out = []
    for _, r in live.iterrows():
        risk = r.get("risk_amount")
        out.append((r["Ticker"], float(risk) if risk == risk and risk else default_risk, str(r["session"])))
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per rule version: signals, entered, closed, win rate, average R."""
    if results.empty:
        return pd.DataFrame()
    rows = []
    for v, g in results.groupby("rule_version", sort=False):
        closed = g[~g["outcome"].isin(["open", "waiting for the next open"]) & ~g["outcome"].str.startswith("not entered")]
        rows.append({"Rule version": v, "First signal": g["session"].min(), "Signals": len(g),
                     "Entered": int(g["entry"].notna().sum()), "Not entered": int(g["outcome"].str.startswith("not entered").sum()),
                     "Closed": len(closed), "Open": int((g["outcome"] == "open").sum()),
                     "Win %": round(100 * (closed["r_multiple"] > 0).mean(), 1) if len(closed) else np.nan,
                     "Avg R (closed)": round(closed["r_multiple"].mean(), 3) if len(closed) else np.nan})
    return pd.DataFrame(rows)
