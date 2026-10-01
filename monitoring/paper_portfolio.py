"""Automatic paper portfolio: follows the chosen plan every session, with no human input.

Each completed session S:
  1. Orders decided at the previous rebalance are filled at the OPEN of the first session after
     the decision (the price you could realistically get), with half the round-trip cost per side.
  2. Holdings are valued at S's close and the equity curve is extended (with the Nifty for comparison).
  3. Every ``horizon`` sessions a new target portfolio is decided from S's model scores using the
     chosen settings (top N, hold-buffer, regime filter, weighting). Those orders fill at the next open.

State lives in ``outputs/paper_portfolio.json``. Delete that file to restart from fresh capital.
No broker is involved; this measures what the plan would have earned.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from models.portfolio import Settings, target_weights

logger = logging.getLogger(__name__)


class PaperPortfolio:
    def __init__(self, path: Path, capital: float, cost_pct: float) -> None:
        self.path = Path(path)
        self.cost_pct = cost_pct
        self.state: dict[str, Any] = self._load() or {
            "capital": capital, "cash": capital, "holdings": {}, "pending": None, "last_session": None,
            "sessions_since_rebalance": None, "history": [], "trades": [], "start": None, "settings": None}

    # ------------------------------------------------------------------ io
    def _load(self) -> dict | None:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.path)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _price(histories: Mapping[str, pd.DataFrame], t: str, d: pd.Timestamp, field: str) -> float | None:
        df = histories.get(t)
        if df is None or d not in df.index:
            return None
        v = float(df.at[d, field])
        return v if math.isfinite(v) and v > 0 else None

    def _last_close(self, histories: Mapping[str, pd.DataFrame], t: str, d: pd.Timestamp, fallback: float) -> float:
        df = histories.get(t)
        if df is None:
            return fallback
        s = df.loc[df.index <= d, "Close"]
        return float(s.iloc[-1]) if len(s) else fallback

    def equity(self, histories: Mapping[str, pd.DataFrame], d: pd.Timestamp) -> float:
        st = self.state
        return st["cash"] + sum(h["shares"] * self._last_close(histories, t, d, h["entry_price"])
                                for t, h in st["holdings"].items())

    # --------------------------------------------------------------- update
    def update(self, session: date, histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None,
               scores: pd.DataFrame, s: Settings, risk_on: bool) -> None:
        """Advance the portfolio to ``session`` (idempotent per session)."""
        st = self.state
        S = pd.Timestamp(session)
        if st["last_session"] and pd.Timestamp(st["last_session"]) >= S:
            return
        if st["start"] is None:
            st["start"] = str(session)
        st["settings"] = s.to_dict()

        # 1) fill pending orders at the first open after the decision date
        if st["pending"] and pd.Timestamp(st["pending"]["date"]) < S:
            self._fill(st["pending"]["targets"], histories, pd.Timestamp(st["pending"]["date"]), S)
            st["pending"] = None

        # 2) mark to market at this close
        eq = self.equity(histories, S)
        nifty = float(benchmark.at[S, "Close"]) if benchmark is not None and S in benchmark.index else None
        st["history"].append({"date": str(session), "equity": round(eq, 2), "nifty": nifty,
                              "invested": round(eq - st["cash"], 2)})

        # 3) decide the next rebalance
        since = st["sessions_since_rebalance"]
        if since is None or since + 1 >= s.horizon:
            if not scores.empty:
                day = scores.rename(columns={"score": "_s"}).assign(pred=scores["pred"])
                held = {t: 1.0 for t in st["holdings"]}
                targets = target_weights(day, held, s, risk_on)
                st["pending"] = {"date": str(session), "targets": targets, "risk_on": risk_on}
                st["sessions_since_rebalance"] = 0
        else:
            st["sessions_since_rebalance"] = since + 1
        st["last_session"] = str(session)
        self.save()

    def _fill(self, targets: dict[str, float], histories: Mapping[str, pd.DataFrame], decided: pd.Timestamp,
              upto: pd.Timestamp) -> None:
        st = self.state
        side_cost = self.cost_pct / 2

        def fill_day(t: str) -> tuple[pd.Timestamp, float] | None:
            df = histories.get(t)
            if df is None:
                return None
            after = df.index[(df.index > decided) & (df.index <= upto)]
            if not len(after):
                return None
            return after[0], float(df.at[after[0], "Open"])

        for t in [t for t in list(st["holdings"]) if t not in targets]:  # sells first
            h = st["holdings"][t]
            f = fill_day(t)
            px = f[1] if f else self._last_close(histories, t, upto, h["entry_price"])
            when = f[0] if f else upto
            value = h["shares"] * px
            st["cash"] += value * (1 - side_cost)
            st["trades"].append({"date": str(when.date()), "side": "SELL", "ticker": t, "shares": h["shares"],
                                 "price": round(px, 2), "pnl_pct": round(100 * (px / h["entry_price"] - 1), 2)})
            del st["holdings"][t]
        new = [t for t in targets if t not in st["holdings"]]
        if not new:
            return
        first = [fill_day(t) for t in new]
        ref = next((f[0] for f in first if f), upto)
        eq = self.equity(histories, ref)
        for t, f in zip(new, first):
            if f is None:
                continue
            when, px = f
            budget = min(eq * targets[t], st["cash"] / (1 + side_cost))
            shares = int(budget // (px * (1 + side_cost)))
            if shares <= 0:
                continue
            st["cash"] -= shares * px * (1 + side_cost)
            st["holdings"][t] = {"shares": shares, "entry_price": round(px, 2), "entry_date": str(when.date())}
            st["trades"].append({"date": str(when.date()), "side": "BUY", "ticker": t, "shares": shares,
                                 "price": round(px, 2), "pnl_pct": None})

    # ------------------------------------------------------------- reporting
    def summary(self, histories: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
        st = self.state
        hist = pd.DataFrame(st["history"])
        out: dict[str, Any] = {"start": st["start"], "capital": st["capital"], "history": hist,
                               "trades": pd.DataFrame(st["trades"]), "pending": st["pending"], "holdings": pd.DataFrame()}
        if hist.empty:
            return out
        last = pd.Timestamp(hist["date"].iloc[-1])
        out["equity"] = float(hist["equity"].iloc[-1])
        out["return_pct"] = 100 * (out["equity"] / st["capital"] - 1)
        n = hist["nifty"].dropna()
        out["nifty_return_pct"] = 100 * (n.iloc[-1] / n.iloc[0] - 1) if len(n) > 1 else None
        rows = []
        for t, h in st["holdings"].items():
            px = self._last_close(histories, t, last, h["entry_price"])
            rows.append({"Ticker": t, "Shares": h["shares"], "Entry date": h["entry_date"], "Entry": h["entry_price"],
                         "Last close": round(px, 2), "Value": round(h["shares"] * px, 2),
                         "P&L %": round(100 * (px / h["entry_price"] - 1), 2)})
        out["holdings"] = pd.DataFrame(rows)
        out["cash"] = st["cash"]
        return out
