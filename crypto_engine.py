"""Crypto decision engine (Binance spot, paper only): the NSE rules and safeguards, adapted to crypto.

Same quality BUY checks (breakout, strong close, not extended, volume, liquidity, trend, ATR stop, market regime,
reward/risk >= 2, next-open entry), the same cross-sectional ranker with walk-forward validation, the same
forward record, alerts and stress test. Differences: BTC is the market (regime = BTC above its 200-day average;
relative strength is vs BTC), the market never closes (daily candles complete at 00:00 UTC), quantities are
fractional (Binance lot sizes), costs are higher and coins are far more correlated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from alerts import AlertCenter
from config.config import AppConfig, BuyQualityConfig, RankerConfig
from data_ingestion.cache import DailyCache
from data_ingestion.data_fetcher import IST, DataFetchError, MarketDataFetcher
from models.ranker import CrossSectionalRanker, build_panel
from monitoring.forward_record import ForwardRecord, held_positions, summarize as summarize_forward
from monitoring.strategy_lab import lab_summary
from research.market import breadth_series
from research.movers import historical_movers
from screeners.buy_quality import backtest_quality, evaluate_latest, regime_breakdown

logger = logging.getLogger(__name__)
CRYPTO_VERSION = 1


def crypto_quality_config(c) -> BuyQualityConfig:
    return BuyQualityConfig(breakout_lookback=c.breakout_lookback, volume_multiple=c.volume_multiple,
                            min_turnover_cr=c.min_turnover, max_stop_pct=c.max_stop_pct,
                            entry_max_gap_pct=c.entry_max_gap_pct, max_hold_days=30)


def crypto_rule_version(cfg: AppConfig) -> str:
    return "c" + hashlib.sha1(json.dumps(cfg.crypto.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:7]


def size_position(entry: float, stop: float, equity: float, risk_pct: float, max_pos_pct: float,
                  step: float) -> tuple[float, float]:
    """Fractional quantity (rounded down to the lot step) risking ``risk_pct`` of equity, capped by position size."""
    risk_unit = entry - stop
    if not (risk_unit > 0 and entry > 0):
        return 0.0, 0.0
    qty = min(equity * risk_pct / risk_unit, equity * max_pos_pct / entry)
    if step and step > 0:
        qty = math.floor(qty / step + 1e-9) * step  # tolerance: 1999.9999999 must not become 1999
    qty = float(f"{qty:.8f}")
    if qty * entry < 10:          # Binance minimum order value is about 10 USDT
        return 0.0, 0.0
    return qty, round(qty * risk_unit, 2)


def crypto_regime(bench: pd.DataFrame, done: dict, brd: pd.Series, benchmark: str) -> dict:
    """Crypto market score from seven BTC/breadth components, plus a separate altcoin regime (descriptive)."""
    c = bench["Close"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    vol = np.log(c).diff().rolling(21).std()
    vol_pct = float((vol.dropna().tail(730) < vol.iloc[-1]).mean())
    comp = {
        "BTC above its 200-day average": bool(c.iloc[-1] > sma200.iloc[-1]),
        "BTC 50-day average above the 200-day": bool(sma50.iloc[-1] > sma200.iloc[-1]),
        "BTC 200-day average rising": bool(sma200.iloc[-1] > sma200.iloc[-21]),
        "At least half of coins above their 200-day": bool(brd.iloc[-1] >= 0.5),
        "Breadth rising over 20 days": bool(brd.iloc[-1] > brd.iloc[-21]) if len(brd) > 21 else False,
        "BTC positive over 30 days": bool(c.iloc[-1] > c.iloc[-31]),
        "Volatility not extreme (BTC below its 80th percentile)": vol_pct < 0.8,
    }
    score = round(100 * sum(comp.values()) / len(comp))
    b30 = c.iloc[-1] / c.iloc[-31] - 1
    rel = [df["Close"].iloc[-1] / df["Close"].iloc[-31] - 1 - b30 for s, df in done.items()
           if s != benchmark and len(df) > 31]
    share = float(np.mean([r > 0 for r in rel])) if rel else float("nan")
    med = float(np.median(rel)) if rel else float("nan")
    alt_state = ("Altcoins leading BTC" if share >= 0.55 and med > 0 else
                 "Altcoins lagging BTC" if share <= 0.45 else "Altcoins mixed vs BTC")
    return {"score": score, "components": comp, "vol_percentile": vol_pct, "alt_state": alt_state,
            "alt_share_beating_btc_30d": share, "alt_median_vs_btc_30d": med}


def strength_vs_btc(done: dict, bench: pd.DataFrame) -> pd.DataFrame:
    """Each coin's return minus BTC's over 7, 30 and 90 days."""
    bc = bench["Close"]
    rows = []
    for s, df in done.items():
        c = df["Close"]
        row = {"Ticker": s}
        for n in (7, 30, 90):
            if len(c) > n and len(bc) > n:
                row[f"vs BTC {n}d %"] = round(100 * ((c.iloc[-1] / c.iloc[-1 - n]) - (bc.iloc[-1] / bc.iloc[-1 - n])), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("vs BTC 30d %", ascending=False).reset_index(drop=True)


@dataclass
class CryptoReport:
    as_of: datetime
    session: date
    symbols: int
    regime: dict
    breadth: float
    quality: pd.DataFrame
    signals: pd.DataFrame
    lab: pd.DataFrame
    trades: pd.DataFrame
    regime_table: pd.DataFrame
    validation: dict | None
    movers_proxy: dict
    movers_now: pd.DataFrame
    forward: dict
    histories: dict = field(default_factory=dict)
    error: str | None = None
    coverage: dict = field(default_factory=dict)
    vs_btc: pd.DataFrame | None = None


class CryptoEngine:
    def __init__(self, cfg: AppConfig, provider=None) -> None:
        from data_ingestion.binance import BinanceProvider
        self.cfg, self.c = cfg, cfg.crypto
        self.provider = provider or BinanceProvider()
        self.fetcher = MarketDataFetcher(replace(cfg.data, requests_per_minute=300, max_retries=3), provider=self.provider)
        self.cache = DailyCache(Path(cfg.cache_dir) / "crypto")
        self.out = Path(cfg.output_dir) / "crypto"
        self.q = crypto_quality_config(self.c)
        self._lock = threading.Lock()
        self._last: CryptoReport | None = None
        self.status, self.status_since, self.last_error = "Waiting to start", datetime.now(IST), None

    def set_status(self, msg: str) -> None:
        if msg != self.status:
            self.status, self.status_since = msg, datetime.now(IST)

    def peek(self) -> CryptoReport | None:
        return self._last

    # ---------------------------------------------------------------- refresh
    def refresh(self, now: datetime | None = None, force: bool = False) -> CryptoReport:
        now = (now or datetime.now(IST)).astimezone(IST)
        with self._lock:
            last = self._last
            if not force and last is not None and (now - last.as_of).total_seconds() < 60 * self.c.refresh_minutes:
                return last
            try:
                self._last = self._compute(now)
                self.last_error = None
                self.set_status("Ready")
            except Exception as exc:
                self.last_error = str(exc)
                self.set_status(f"Crypto refresh failed ({exc}); retrying automatically")
                raise
            return self._last

    def _universe(self) -> list[str]:
        path = Path(self.cfg.cache_dir) / "crypto" / "universe.json"
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            saved = json.loads(path.read_text())
            if saved["date"] == today:
                getattr(self.provider, "lot_steps", {}).update(saved.get("steps", {}))
                return saved["symbols"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        syms = self.provider.top_symbols(self.c.universe_size, self.c.min_quote_volume_usdt)
        if self.c.benchmark not in syms:
            syms = [self.c.benchmark] + syms
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"date": today, "symbols": syms, "steps": getattr(self.provider, "lot_steps", {})}))
        return syms

    def _compute(self, now: datetime) -> CryptoReport:
        self.set_status("Loading the Binance universe")
        symbols = self._universe()
        self.set_status(f"Updating daily candles for {len(symbols)} coins")
        frames = self.cache.sync(self.fetcher, symbols, now, self.c.history_days, status=self.set_status)
        today_utc = pd.Timestamp(now.astimezone(timezone.utc).date())
        done = {s: df[df.index < today_utc] for s, df in frames.items()}   # only completed UTC days
        short = sorted(s for s, df in done.items() if len(df) <= 260)
        done = {s: df for s, df in done.items() if len(df) > 260}
        coverage = {"requested": len(symbols), "used": len(done), "short_history": short,
                    "no_data": sorted(set(symbols) - set(frames))}
        bench = done.get(self.c.benchmark)
        if bench is None or not done:
            raise DataFetchError("Bitcoin history is unavailable, so the crypto market regime cannot be judged")
        session = max(df.index[-1] for df in done.values()).date()
        brd = breadth_series(done)["pct_above_200"]
        heavy = self._heavy(done, bench, brd, session)
        self.set_status("Applying the BUY checks")
        quality = evaluate_latest(done, self.q, bench, brd)
        bc = bench["Close"]
        sma200 = bc.rolling(200).mean()
        regime = {"risk_on": bool(bc.iloc[-1] > sma200.iloc[-1]), "btc": float(bc.iloc[-1]),
                  "btc_vs_200d": float(bc.iloc[-1] / sma200.iloc[-1] - 1),
                  "btc_3m": float(bc.iloc[-1] / bc.iloc[-91] - 1),
                  "state": "Risk-on" if bc.iloc[-1] > sma200.iloc[-1] else "Risk-off",
                  **crypto_regime(bench, done, brd, self.c.benchmark)}
        vs_btc = strength_vs_btc(done, bench)
        fwd = ForwardRecord(self.out / "forward_signals.csv")
        prior = fwd.evaluate(done, self.q, self.c.cost_pct)
        prior = prior[prior["session"] < session.isoformat()] if not prior.empty else prior
        held = held_positions(prior, self.c.account_equity_usdt * self.c.risk_per_trade_pct)
        signals = self._signals(quality, heavy, done, held)
        issued = signals[(signals["Action"] == "BUY") & (signals["Qty"] > 0)]
        fwd.record(issued, session, crypto_rule_version(self.cfg))
        results = fwd.evaluate(done, self.q, self.c.cost_pct)
        closed = results[results["exit"].notna()] if not results.empty else results
        forward = {"results": results, "summary": summarize_forward(results), "closed": closed,
                   "win_rate": float((closed["r_multiple"] > 0).mean()) if len(closed) else None,
                   "avg_r": float(closed["r_multiple"].mean()) if len(closed) else None}
        try:
            movers_now = self.provider.movers_24h(list(done)) if hasattr(self.provider, "movers_24h") else pd.DataFrame()
        except Exception:  # noqa: BLE001 - informational only
            movers_now = pd.DataFrame()
        self._alerts(issued, session, now, regime)
        self._open_alerts(results, now)
        from research.robustness import health
        forward["health"] = health(results, heavy["trades"][heavy["trades"]["variant"] == "quality"]
                                   if heavy["trades"] is not None and not heavy["trades"].empty else pd.DataFrame())
        return CryptoReport(now, session, len(done), regime, float(brd.iloc[-1]), quality, signals, heavy["lab"],
                            heavy["trades"], heavy["regime_table"], heavy["validation"], heavy["movers"], movers_now,
                            forward, done, None, coverage, vs_btc)

    def _heavy(self, done, bench, brd, session: date) -> dict:
        """Backtests, regime study, ranker validation and movers test: once per UTC day."""
        key = hashlib.sha1(repr((CRYPTO_VERSION, session, sorted(done), self.c)).encode()).hexdigest()[:12]
        path = Path(self.cfg.cache_dir) / "crypto" / f"heavy_{session}_{key}.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001
                pass
        self.set_status("Backtesting the BUY rules on crypto history")
        trades, _ = backtest_quality(done, self.q, bench, self.c.cost_pct, brd)
        nr, _ = backtest_quality(done, replace(self.q, regime_required=False), bench, self.c.cost_pct, brd)
        parts = [t for t in (trades, nr.assign(variant="quality_no_regime") if not nr.empty else nr) if not t.empty]
        alltr = pd.concat(parts, ignore_index=True) if parts else trades
        out = {"trades": alltr, "lab": lab_summary(alltr), "regime_table": regime_breakdown(nr), "validation": None,
               "scores": pd.DataFrame()}
        self.set_status("Validating the ranking model walk-forward on crypto")
        try:
            rk = RankerConfig(horizon_days=10, n_splits=4, min_train_dates=400, min_turnover_cr=self.c.min_turnover,
                              cost_pct=self.c.cost_pct, optimize=False, holdout_days=180)
            panel = build_panel(done, bench, {}, 10)
            ranker = CrossSectionalRanker(rk)
            val = ranker.walk_forward(panel)
            val.pop("oos", None)
            out["validation"] = val
            ranker.fit(panel)
            latest = panel[panel["date"] == pd.Timestamp(session)].dropna(subset=["r_ret_63"]).set_index("ticker")
            if not latest.empty:
                latest["score"] = pd.Series(ranker.predict(latest), index=latest.index).rank(pct=True)
                out["scores"] = latest[["score"]]
        except ValueError as exc:
            logger.warning("Crypto ranker unavailable: %s", exc)
        self.set_status("Testing whether buying crypto movers has paid off")
        out["movers"] = historical_movers(done, bench, min_gain=self.c.mover_min_gain,
                                          min_turnover_cr=self.c.min_turnover, cost_pct=self.c.cost_pct)
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in sorted(path.parent.glob("heavy_*.joblib"))[:-2]:
            old.unlink(missing_ok=True)
        joblib.dump(out, path)
        return out

    def _signals(self, quality: pd.DataFrame, heavy: dict, done: dict, held: list) -> pd.DataFrame:
        cols = ["Action", "Ticker", "State", "Readiness", "Rank", "Price", "Breakout level", "Stop Loss", "Target Price",
                "R:R", "ATR", "Qty", "Risk (INR)", "Allocation", "Checks"]
        if quality is None or quality.empty:
            return pd.DataFrame(columns=cols)
        val = heavy.get("validation") or {}
        validated = bool(val.get("validated"))
        scores = heavy.get("scores", pd.DataFrame())
        held_names = {h[0]: h[2] for h in held}
        rows = []
        for _, r in quality.iterrows():
            score = float(scores.at[r["Ticker"], "score"]) if not scores.empty and r["Ticker"] in scores.index else np.nan
            buy = bool(r["passes"]) and (not validated or not (score < self.q.min_model_score))
            if not buy and r["State"] not in ("WATCH CLOSELY", "WAIT"):
                continue
            rows.append({"Action": "BUY" if buy else "WATCH", "Ticker": r["Ticker"], "State": r["State"],
                         "Readiness": r["Readiness"], "Rank": round(100 * score) if score == score else np.nan,
                         "Price": r["Close"], "Breakout level": r["Breakout level"], "Stop Loss": r["Stop"],
                         "Target Price": r["Target"], "R:R": r["R:R"] if r["g_breakout"] else np.nan, "ATR": r["ATR"],
                         "Qty": 0.0, "Risk (INR)": 0.0, "Allocation": "Watch only" if not buy else "",
                         "Checks": "all passed" if buy else f"failed: {r['Failed']}"})
        sig = pd.DataFrame(rows, columns=cols)
        if sig.empty:
            return sig
        sig = sig.sort_values(["Action", "Rank", "Readiness"], ascending=[True, False, False]).reset_index(drop=True)
        steps = getattr(self.provider, "lot_steps", {})
        c = self.c
        used = sum(h[1] for h in held)
        taken = list(held_names)
        for i in sig.index[sig["Action"] == "BUY"]:
            tk = sig.at[i, "Ticker"]
            if tk in held_names:
                sig.at[i, "Allocation"] = f"Already held (signal of {held_names[tk]}); one position per coin"
                continue
            if len(taken) >= c.max_open_positions:
                sig.at[i, "Allocation"] = "Watch only: position limit"
                continue
            rets = done[tk]["Close"].pct_change().tail(60)
            twin = next((t for t in taken if t in done and rets.corr(done[t]["Close"].pct_change().tail(60))
                         > c.max_correlation), None)
            if twin:
                sig.at[i, "Allocation"] = f"Skipped: moves with {twin}"
                continue
            qty, risk = size_position(sig.at[i, "Price"], sig.at[i, "Stop Loss"], c.account_equity_usdt,
                                      c.risk_per_trade_pct, c.max_position_pct, steps.get(tk, 0.0))
            room = c.account_equity_usdt * c.max_portfolio_heat_pct - used
            if risk > room and risk > 0:
                qty, risk = (qty * room / risk, room) if room > 0 else (0.0, 0.0)
                qty = math.floor(qty / steps[tk] + 1e-9) * steps[tk] if steps.get(tk) else qty
            if qty <= 0:
                sig.at[i, "Allocation"] = "Watch only: risk budget used"
                continue
            sig.at[i, "Qty"], sig.at[i, "Risk (INR)"], sig.at[i, "Allocation"] = qty, risk, "Sized"
            used += risk
            taken.append(tk)
        return sig

    def _open_alerts(self, results: pd.DataFrame, now: datetime) -> None:
        """At the daily open (05:30 IST), say whether each pending signal was entered or skipped (chase protection)."""
        if results is None or results.empty:
            return
        center = AlertCenter(Path(self.cfg.output_dir) / "alerts.json")
        for _, r in results[results["entry_date"].notna()].iterrows():
            key = f"crypto:open:{r['session']}:{r['Ticker']}"
            if str(r["outcome"]).startswith("not entered"):
                center.add(key, "skip", f"[Crypto] At the open: skip {r['Ticker']}; {r['outcome'][20:]}.", now)
            else:
                center.add(key, "ok", f"[Crypto] At the open: {r['Ticker']} entered at {r['entry']:,.6g} USDT "
                                      f"(stop {r['stop']:,.6g}).", now)

    def _alerts(self, issued: pd.DataFrame, session: date, now: datetime, regime: dict) -> None:
        center = AlertCenter(Path(self.cfg.output_dir) / "alerts.json")
        for _, r in issued.iterrows():
            center.add(f"crypto:buy:{session}:{r['Ticker']}", "buy",
                       f"[Crypto] BUY signal: {r['Ticker']} at the next daily open (05:30 IST). Entry from "
                       f"{r['Breakout level']:,.6g} USDT, stop {r['Stop Loss']:,.6g}, target {r['Target Price']:,.6g}, "
                       f"qty {r['Qty']:g}.", now)
        path = self.out / "regime_state.txt"
        prev = path.read_text().strip() if path.exists() else None
        if prev != regime["state"]:
            if prev is not None:
                center.add(f"crypto:regime:{session}:{regime['state']}", "watch",
                           f"[Crypto] Market regime changed: {prev} → {regime['state']} (BTC vs its 200-day average).", now)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(regime["state"])
