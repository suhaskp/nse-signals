"""Market-intelligence engine: research + predictive ranking + validation + suggestions.

Daily pipeline (heavy parts cached per session, so reopening the dashboard is fast):

    universe (Nifty 500, auto) + Superstar drop-in + Nifty index
      -> cached daily history (incremental)
      -> research: regime, breadth, sector strength
      -> cross-sectional ranker: walk-forward validation, calibration, final fit, today's scores
      -> strategy lab: Chartink rules vs improved variants
      -> suggestions: top-ranked BUYs, bottom-ranked SELLs, each with reasons, confirmations,
         historical calibration, ATR stop/target, position size and a review date

If the ranker fails its out-of-sample validation gate, every suggestion is labelled
"Unvalidated": the ranking is still shown, but it has not earned trust on this data.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.config import AppConfig
from data_ingestion.data_fetcher import IST, DataFetchError, MarketDataFetcher
from data_ingestion.universe import load_sectors
from live_engine import LiveSignalEngine, market_phase
from dataclasses import replace as dc_replace

from models.portfolio import Settings, relabel
from models.ranker import CrossSectionalRanker, build_panel, explain
from alerts import AlertCenter
from monitoring.forward_record import ForwardRecord, summarize as summarize_forward
from monitoring.paper_portfolio import PaperPortfolio
from storage import freeze_status, pin_freeze_settings, rule_version
import decisions
from research.robustness import (count_trials, deflated_sharpe, execution_stress, health, sensitivity,
                                 sensitivity_verdict, stress_verdict)
from research.optimizer import optimize
from monitoring.strategy_lab import lab_by_year, lab_summary, run_lab
from research.market import market_breadth, market_regime, sector_strength
from risk_engine.risk_manager import RiskError, RiskManager, Side
from research.market import breadth_series, regime_detail, trend_leaders
from research.movers import follow_up, historical_movers
from screeners.buy_quality import (backtest_quality, evaluate_latest, filter_ablation, plan_entry,
                                   regime_breakdown)
from screeners.screener import atr

logger = logging.getLogger(__name__)

REC_COLUMNS = ["Action", "Ticker", "Sector", "Conviction", "Model score", "Hist. excess %", "Hist. beat %",
               "Price", "Stop Loss", "Target Price", "R:R", "Qty", "Risk (INR)", "Allocation", "Entry rule",
               "Review by", "Why", "Checks", "Breakout level", "ATR", "Superstar"]
ALLOC_NOTE = {"risk_per_trade": "Sized", "volatility_target": "Sized", "max_position_pct": "Sized (position cap)",
              "portfolio_heat": "Reduced: total-risk limit", "max_open_positions": "Watch only: position limit",
              "insufficient_equity": "Watch only: too expensive", "neutral_half_size": "Half size (neutral market)"}
LEVELS = ["Low", "Medium", "High"]
CACHE_VERSION = 9  # bump when cached results from older code must not be reused


@dataclass
class IntelligenceReport:
    as_of: datetime
    phase: str
    phase_note: str
    session: date
    universe_label: str
    superstar_label: str
    skipped: list[str]
    regime: dict | None
    breadth: dict
    sectors: pd.DataFrame
    validation: dict | None
    recommendations: pd.DataFrame
    scores: pd.DataFrame
    contributions: pd.DataFrame | None
    lab_summary: pd.DataFrame
    lab_by_year: pd.DataFrame
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark: pd.DataFrame | None = None
    error: str | None = None
    settings: Settings | None = None
    optimizer: dict | None = None
    paper: dict | None = None
    quality: pd.DataFrame | None = None
    quality_funnel: pd.DataFrame | None = None
    tracking: dict | None = None
    lab_trades: pd.DataFrame | None = None
    data_warning: str | None = None
    leaders: pd.DataFrame | None = None
    movers_proxy: dict | None = None
    movers_followup: dict | None = None
    regime_breakdown: pd.DataFrame | None = None
    ablation: pd.DataFrame | None = None
    regime_detail: dict | None = None
    changes: pd.DataFrame | None = None
    journal: pd.DataFrame | None = None
    research: dict | None = None

    @property
    def validated(self) -> bool:
        return bool(self.validation and self.validation.get("validated"))


def apply_portfolio_limits(recs: pd.DataFrame, plans: dict, risk: RiskManager,
                           histories: dict[str, pd.DataFrame] | None = None,
                           held: list[tuple[str, float, str]] | None = None) -> pd.DataFrame:
    """Size BUY ideas together, best first, within max positions and total open risk, while limiting
    concentration: at most ``max_per_sector`` per sector, and no BUY whose last-60-session daily returns move
    almost in lockstep (correlation above ``max_correlation``) with one already chosen.

    SELL ideas are exits/avoids for stocks you may hold, so they are not allocated capital.
    """
    recs = recs.copy()
    recs["Allocation"] = np.where(recs["Action"] == "SELL", "Exit / avoid if held",
                                  np.where(recs["Action"] == "WATCH", "Watch only", ""))
    buy_idx = recs.index[recs["Action"] == "BUY"]
    held = held or []
    held_names = {h[0]: h[2] for h in held}
    chosen: list[str] = [h[0] for h in held]  # open positions count for sector and correlation limits too
    skipped: dict[str, str] = {tk: f"Already held (signal of {d}); one position per stock"
                               for tk, d in held_names.items() if (recs["Ticker"] == tk).any()}
    rets = {}
    for i in buy_idx:
        tk, sector = recs.at[i, "Ticker"], recs.at[i, "Sector"] if "Sector" in recs else ""
        if tk in skipped:
            continue
        same = [c for c in chosen if sector and (recs.loc[recs["Ticker"] == c, "Sector"] == sector).any()]
        if sector and len(same) >= risk.cfg.max_per_sector:
            skipped[tk] = f"Skipped: sector limit ({risk.cfg.max_per_sector} {sector} already chosen)"
            continue
        if histories is not None and tk in histories:
            rets[tk] = histories[tk]["Close"].pct_change().tail(60)
            twin = next(((c, rets[tk].corr(rets[c])) for c in chosen if c in rets
                         and rets[tk].corr(rets[c]) > risk.cfg.max_correlation), None)
            if twin:
                skipped[tk] = f"Skipped: moves with {twin[0]} (correlation {twin[1]:.2f})"
                continue
        chosen.append(tk)
    for tk, why in skipped.items():
        recs.loc[recs["Ticker"] == tk, ["Qty", "Risk (INR)", "Allocation"]] = [0, 0.0, why]
    buy_idx = [i for i in buy_idx if recs.at[i, "Ticker"] not in skipped]
    ordered = [plans[("BUY", recs.at[i, "Ticker"])] for i in buy_idx if ("BUY", recs.at[i, "Ticker"]) in plans]
    allocated = {p.ticker: p for p in risk.allocate(ordered, used_heat=sum(h[1] for h in held), taken=len(held))}
    for i in buy_idx:
        p = allocated.get(recs.at[i, "Ticker"])
        if p is None:
            recs.at[i, "Allocation"] = "No plan"
            continue
        recs.at[i, "Qty"], recs.at[i, "Risk (INR)"] = p.units, p.risk_amount
        recs.at[i, "Allocation"] = ALLOC_NOTE.get(p.binding_constraint, p.binding_constraint)
    return recs


def freshness_warning(failed: list[str], n_symbols: int, session: date, now: datetime, phase: str) -> str | None:
    """Plain warning when today's price update failed, so stale data is never shown silently."""
    expected = now.date() if phase == "after_close" else None
    if len(failed) > max(0.2 * n_symbols, 25):
        return (f"Today's price update failed for {len(failed)} of {n_symbols} stocks (the free Yahoo feed is probably "
                f"rate-limiting). Showing data up to the close of {session:%d %b %Y}. Retrying automatically; make sure "
                "only one dashboard is running.")
    if expected is not None and session < expected:
        return (f"Today's closing prices are not in yet; showing data up to {session:%d %b %Y}. "
                "Retrying automatically.")
    return None


class IntelligenceEngine:
    """Thread-safe daily intelligence builder with memory and disk caching."""

    def __init__(self, cfg: AppConfig, fetcher: MarketDataFetcher | None = None, auto_universe: bool = True,
                 live: LiveSignalEngine | None = None) -> None:
        self.cfg = cfg
        self.live = live or LiveSignalEngine(cfg, fetcher, auto_universe)
        self.fetcher = self.live.fetcher
        self.risk = RiskManager(cfg.risk)
        self.store = Path(cfg.cache_dir) / "intelligence"
        self._lock = threading.Lock()
        self._last: IntelligenceReport | None = None
        self._check: tuple[datetime, date, pd.DataFrame | None, str] | None = None
        self.status = "Waiting for the breakout scan to finish downloading prices"
        self.status_since = datetime.now(IST)
        self.last_error: str | None = None

    def set_status(self, msg: str) -> None:
        if msg != self.status:
            self.status, self.status_since = msg, datetime.now(IST)

    def peek(self) -> "IntelligenceReport | None":
        """Latest report without waiting (None until the first run completes)."""
        return self._last

    # ------------------------------------------------------------------ io
    def _sectors(self, symbols: list[str]) -> dict[str, str]:
        src = Path(self.cfg.universe_file) if self.cfg.universe_file else \
            Path(self.cfg.cache_dir) / "universe" / "ind_nifty500list.csv"
        sectors = load_sectors(src) if src.exists() else {}
        provider = self.fetcher.provider
        if hasattr(provider, "sector"):  # synthetic demo market
            sectors.update({s: provider.sector(s) for s in symbols if s not in sectors})
        return sectors

    def _key(self, session: date, symbols: list[str]) -> str:
        blob = repr((CACHE_VERSION, session, sorted(symbols), self.cfg.ranker, self.cfg.breakout,
                     self.cfg.risk.atr_stop_multiplier, self.cfg.risk.reward_risk_ratio))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    # ------------------------------------------------------------- refresh
    def refresh(self, now: datetime | None = None, force: bool = False) -> IntelligenceReport:
        now = (now or datetime.now(IST)).astimezone(IST)
        phase, note = market_phase(self.cfg, now)
        with self._lock:
            last = self._last
            if not force and last is not None and last.phase == phase and (now - last.as_of).total_seconds() < 1800:
                return last
            try:
                self._last = self._compute(now, phase, note)
            except Exception as exc:
                self.last_error = str(exc)
                self.set_status(f"Last run failed ({exc}); retrying automatically")
                raise
            self.last_error = None
            self.set_status("Ready")
            return self._last

    def _compute(self, now: datetime, phase: str, note: str) -> IntelligenceReport:
        cfg = self.cfg
        symbols, uni_label, skipped = self.live.resolve_universe()
        superstar, ss_label, ss_skipped = self.live.resolve_superstar()
        scan = list(dict.fromkeys(symbols + sorted(superstar)))
        self.set_status("Updating prices")
        frames = self.live.cache.sync(self.fetcher, scan + [cfg.benchmark], now, cfg.data.history_lookback_days,
                                      status=self.set_status)
        today = pd.Timestamp(now.date())
        cut = (lambda df: df[df.index <= today]) if phase == "after_close" else (lambda df: df[df.index < today])
        done = {s: cut(df) for s, df in frames.items()}
        done = {s: df for s, df in done.items() if not df.empty}
        bench = done.pop(cfg.benchmark, None)
        if not done:
            raise DataFetchError("No market data could be loaded. Check your internet connection.")
        if bench is None:
            logger.warning("Benchmark %s unavailable; index-relative features and regime are disabled", cfg.benchmark)
        session = max(df.index[-1] for df in done.values()).date()
        sectors = self._sectors(list(done))

        heavy = self._heavy(done, bench, sectors, session)
        regime = market_regime(bench) if bench is not None and len(bench) > 200 else None
        breadth = market_breadth(done)
        sect = sector_strength(done, sectors)
        self.set_status("Applying the BUY quality checks")
        q = cfg.buy_quality
        brd = breadth_series(done)["pct_above_200"]
        quality = evaluate_latest(done, q, bench, brd)
        quality = self._watch_history(quality, session)
        fwd = ForwardRecord(Path(cfg.output_dir) / "forward_signals.csv")
        prior = fwd.evaluate(done, q, cfg.breakout.est_round_trip_cost_pct)
        prior = prior[prior["session"] < session.isoformat()] if not prior.empty else prior
        from monitoring.forward_record import held_positions
        self._held = held_positions(prior, cfg.risk.account_equity * cfg.risk.risk_per_trade_pct)
        recs = self._recommend(heavy, done, regime, sect, superstar, session, quality)

        leaders = trend_leaders(done, bench, quality, heavy["scores"], sectors, q.min_turnover_cr)
        movers_followup = follow_up(Path(cfg.output_dir) / "movers_log.csv", done, bench)
        paper = None
        if heavy.get("settings") is not None and not heavy["scores"].empty:
            pp = PaperPortfolio(Path(cfg.output_dir) / "paper_portfolio.json", cfg.risk.account_equity,
                                cfg.ranker.cost_pct)
            pp.update(session, done, bench, heavy["scores"], heavy["settings"], regime["risk_on"] if regime else True)
            paper = pp.summary(done)
        no_data = sorted(set(self.live.cache.unavailable()) & set(scan))
        data_warning = freshness_warning(self.live.cache.last_failed, len(scan), session, now, phase)
        if data_warning and not recs.empty:
            stale_buy = recs["Action"] == "BUY"
            recs.loc[stale_buy, ["Qty", "Risk (INR)"]] = [0, 0.0]
            recs.loc[stale_buy, "Allocation"] = "Waiting for fresh data before any BUY"
        regime_d = regime_detail(bench, done) if bench is not None and len(bench) > 220 else None
        tracking = self._tracking(recs, done, session)
        self._alerts(recs, session, now, data_warning, regime_d, tracking["freeze"])
        changes = self._changes(session)
        journal = self._journal(quality, recs, session, data_warning)
        research = {**(heavy.get("robustness") or {}), **self._trials_and_dsr(heavy),
                    "health": health(tracking["results"], heavy.get("quality_trades"))}
        return IntelligenceReport(now, phase, note, session, uni_label, ss_label, skipped + ss_skipped + no_data, regime,
                                  breadth, sect, heavy["validation"], recs, heavy["scores"], heavy["contrib"],
                                  heavy["lab_summary"], heavy["lab_by_year"], done, bench, heavy.get("error"),
                                  heavy.get("settings"), heavy.get("optimizer"), paper, quality,
                                  heavy.get("quality_funnel"), tracking, heavy.get("lab_trades"), data_warning,
                                  leaders, heavy.get("movers_proxy"), movers_followup, heavy.get("regime_breakdown"),
                                  heavy.get("ablation"), regime_d, changes, journal, research)

    def _heavy(self, done: dict[str, pd.DataFrame], bench: pd.DataFrame | None, sectors: dict[str, str],
               session: date) -> dict[str, Any]:
        """Walk-forward validation, final model, today's scores, strategy lab (cached per session)."""
        key = self._key(session, list(done))
        path = self.store / f"{session.isoformat()}_{key}.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001
                logger.warning("Cached intelligence unreadable; recomputing")
        out: dict[str, Any] = {"validation": None, "scores": pd.DataFrame(), "contrib": None, "error": None}
        try:
            rk = self.cfg.ranker
            logger.info("Building ranker panel for %d stocks (session %s)", len(done), session)
            self.set_status(f"Computing ~20 factors for {len(done)} stocks")
            panel = build_panel(done, bench, sectors, rk.horizon_days)
            chosen = Settings(rk.horizon_days, rk.top_n)
            if rk.optimize:
                opt = self._optimizer(panel, done, bench, session)
                out["optimizer"] = {k: v for k, v in opt.items() if k != "reports"}
                chosen = opt["chosen"]
                fr = freeze_status(self.cfg, Path(self.cfg.data_home))
                if fr.get("active"):
                    if fr.get("settings"):
                        chosen = Settings(**fr["settings"])  # the freeze keeps the plan fixed
                        out["optimizer"]["reason"] = ("Rules are frozen, so the plan chosen at the start of the freeze is "
                                                      "kept. This week's tuning: " + opt["reason"])
                    else:
                        pin_freeze_settings(Path(self.cfg.data_home), chosen.to_dict())
                out["validation"] = opt["reports"].get(chosen.horizon, opt["reports"][rk.horizon_days])
            else:
                self.set_status("Validating the prediction model walk-forward (the longest step, a few minutes)")
                out["validation"] = CrossSectionalRanker(rk).walk_forward(panel)
                out["validation"].pop("oos", None)
            out["settings"] = chosen
            if chosen.horizon != rk.horizon_days:
                panel = relabel(panel, done, bench, chosen.horizon)
            self.set_status(f"Training the final model ({chosen.horizon}-day horizon) and scoring today's stocks")
            ranker = CrossSectionalRanker(dc_replace(rk, horizon_days=chosen.horizon, top_n=chosen.top_n))
            ranker.fit(panel)
            latest = panel[(panel["date"] == pd.Timestamp(session))
                           & (panel["turnover_cr"] >= self.cfg.ranker.min_turnover_cr)].dropna(subset=["r_ret_63"])
            latest = latest.set_index("ticker")
            latest["pred"] = ranker.predict(latest)
            latest["score"] = latest["pred"].rank(pct=True)
            out["scores"] = latest.sort_values("score", ascending=False)
            out["contrib"] = ranker.contributions(latest)
        except ValueError as exc:
            logger.error("Ranker unavailable: %s", exc)
            out["error"] = str(exc)
        self.set_status("Backtesting the strategy-lab variants")
        trades = run_lab(done, bench, self.cfg.breakout, self.cfg.risk.atr_stop_multiplier,
                         self.cfg.risk.reward_risk_ratio)
        self.set_status("Backtesting the quality-filtered BUY rules")
        q = self.cfg.buy_quality
        brd = breadth_series(done)["pct_above_200"]
        q_trades, funnel = backtest_quality(done, q, bench, self.cfg.breakout.est_round_trip_cost_pct, brd)
        self.set_status("Backtesting the BUY rules without the market-regime check (for comparison)")
        nr_trades, _ = backtest_quality(done, dc_replace(q, regime_required=False), bench,
                                        self.cfg.breakout.est_round_trip_cost_pct, brd)
        out["regime_breakdown"] = regime_breakdown(nr_trades)
        self.set_status("Backtesting the three-state market regime (half size in a neutral market)")
        tr3, _ = backtest_quality(done, dc_replace(q, regime_required=True, regime_mode="three_state"), bench,
                                  self.cfg.breakout.est_round_trip_cost_pct, brd)
        if not tr3.empty:
            half = tr3["regime_state"] == "neutral"
            tr3.loc[half, "r_multiple"] *= 0.5          # half-size position in a neutral market
            tr3.loc[half, "return_pct"] *= 0.5
            tr3["variant"] = "quality_3regime"
        import os
        out["robustness"] = (self._robustness(done, bench, brd, q_trades, session)
                             if os.getenv("PIPELINE_ROBUSTNESS", "1") != "0" else {})
        self.set_status("Testing each BUY check one at a time")
        out["ablation"] = filter_ablation(done, q, bench, self.cfg.breakout.est_round_trip_cost_pct, brd)
        parts = [t for t in (q_trades, nr_trades.assign(variant="quality_no_regime") if not nr_trades.empty else nr_trades,
                             tr3) if not t.empty]
        q_trades_all = pd.concat(parts, ignore_index=True) if parts else q_trades
        out["quality_trades"], out["quality_funnel"] = q_trades, funnel
        out["quality_open"] = funnel.attrs.get("open_trades", pd.DataFrame())
        if not q_trades_all.empty:
            trades = pd.concat([trades, q_trades_all[[c for c in trades.columns if c in q_trades_all.columns]]],
                               ignore_index=True)
        out["lab_summary"], out["lab_by_year"] = lab_summary(trades), lab_by_year(trades)
        out["lab_trades"] = trades
        self.set_status("Testing whether buying 'today's movers' has paid off historically")
        out["movers_proxy"] = historical_movers(done, bench, min_turnover_cr=q.min_turnover_cr,
                                                cost_pct=self.cfg.breakout.est_round_trip_cost_pct)
        self.store.mkdir(parents=True, exist_ok=True)
        for old in sorted(p for p in self.store.glob("*.joblib") if not p.name.startswith("optimizer_"))[:-4]:
            old.unlink(missing_ok=True)
        if out["error"] is None:  # a failed run is retried on the next refresh, not cached for the day
            joblib.dump(out, path)
        return out

    LIVE_STATES = ("BUY NOW", "WATCH CLOSELY", "WAIT")

    def _watch_history(self, quality: pd.DataFrame, session: date) -> pd.DataFrame:
        """Log every stock's state and failed checks per session; add days on watch and readiness trend;
        expire setups that have not broken out after ``watch_expiry_sessions`` on watch."""
        if quality is None or quality.empty:
            return quality
        path = Path(self.cfg.output_dir) / "watch_history.csv"
        cols = ["session", "Ticker", "Readiness", "State", "Failed"]
        hist = pd.read_csv(path, dtype={"session": str}) if path.exists() else pd.DataFrame(columns=cols)
        if "Failed" not in hist:
            hist["Failed"] = ""
        base = quality if "Failed" in quality else quality.assign(Failed="")
        today = base.assign(session=session.isoformat())[cols]
        hist = hist[hist["session"] != session.isoformat()]
        hist = pd.concat([hist, today], ignore_index=True) if not hist.empty else today
        path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(path, index=False)
        self._hist_cache = hist
        live = hist[hist["State"].isin(self.LIVE_STATES + ("EXPIRED",))]
        sessions = sorted(hist["session"].unique(), reverse=True)
        present = {s_: set(g["Ticker"]) for s_, g in live.groupby("session")}
        days, trend = [], []
        for _, r in quality.iterrows():
            n = 0
            for s_ in sessions:
                if r["Ticker"] in present.get(s_, set()):
                    n += 1
                else:
                    break
            days.append(n)
            old = live[(live["Ticker"] == r["Ticker"]) & (live["session"] == sessions[min(3, len(sessions) - 1)])]
            trend.append(float(r["Readiness"] - old["Readiness"].iloc[0]) if n > 3 and len(old) else np.nan)
        quality = quality.copy()
        quality["Days on watch"], quality["Readiness trend"] = days, trend
        stale = (quality["Days on watch"] > self.cfg.buy_quality.watch_expiry_sessions) & ~quality["g_breakout"] \
            & quality["State"].isin(["WATCH CLOSELY", "WAIT"])
        quality.loc[stale, "State"] = "EXPIRED"
        return quality

    def _changes(self, session: date) -> pd.DataFrame:
        """What changed since the previous session: state moves and big readiness moves, with the checks involved."""
        hist = getattr(self, "_hist_cache", None)
        if hist is None or hist.empty:
            return pd.DataFrame()
        prev_sessions = sorted(s_ for s_ in hist["session"].unique() if s_ < session.isoformat())
        if not prev_sessions:
            return pd.DataFrame()
        prev = hist[hist["session"] == prev_sessions[-1]].set_index("Ticker")
        cur = hist[hist["session"] == session.isoformat()].set_index("Ticker")
        rank = {"BUY NOW": 4, "WATCH CLOSELY": 3, "WAIT": 2, "EXPIRED": 1, "IGNORE": 0}
        rows = []
        for tk in cur.index.intersection(prev.index):
            a, b = prev.loc[tk], cur.loc[tk]
            if a["State"] == "IGNORE" and b["State"] == "IGNORE":
                continue
            dr = float(b["Readiness"] - a["Readiness"])
            if a["State"] == b["State"] and abs(dr) < 15:
                continue
            fa = set(filter(None, str(a["Failed"] if isinstance(a["Failed"], str) else "").split(", ")))
            fb = set(filter(None, str(b["Failed"] if isinstance(b["Failed"], str) else "").split(", ")))
            why = [f"now passes {x}" for x in sorted(fa - fb)] + [f"now fails {x}" for x in sorted(fb - fa)]
            move = rank.get(b["State"], 0) - rank.get(a["State"], 0)
            rows.append({"Ticker": tk, "Change": "↑" if move > 0 or (move == 0 and dr > 0) else "↓",
                         "Was": a["State"], "Now": b["State"], "Readiness": f"{a['Readiness']:.0f} → {b['Readiness']:.0f}",
                         "Why": "; ".join(why) if why else "readiness moved (closer to / further from the breakout)",
                         "_order": (-abs(move), -abs(dr))})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        out.attrs["previous"] = prev_sessions[-1]
        return out

    def _journal(self, quality: pd.DataFrame, recs: pd.DataFrame, session: date, data_warning: str | None) -> pd.DataFrame:
        """Append-only decision journal: every candidate's decision for the session, written once, never rewritten."""
        path = Path(self.cfg.output_dir) / "decision_journal.csv"
        log = pd.read_csv(path, dtype={"session": str}) if path.exists() else pd.DataFrame()
        done_keys = set(zip(log["session"], log["Ticker"])) if not log.empty else set()
        issued = set(recs.loc[(recs["Action"] == "BUY") & (recs["Qty"] > 0), "Ticker"]) if not recs.empty else set()
        q = self.cfg.buy_quality
        cands = quality[quality["State"].isin(self.LIVE_STATES)] if quality is not None and not quality.empty else quality
        rows = []
        for _, r in (cands.iterrows() if cands is not None else []):
            if (session.isoformat(), r["Ticker"]) in done_keys:
                continue
            stale = bool(data_warning)
            rows.append({"session": session.isoformat(), "recorded_at": datetime.now(IST).isoformat(timespec="seconds"),
                         "Ticker": r["Ticker"], "Decision": "BUY (issued)" if r["Ticker"] in issued else r["State"],
                         "Readiness": r["Readiness"], "Rule version": rule_version(self.cfg), "Price": r["Close"],
                         "Breakout level": r["Breakout level"], "Stop": r["Stop"], "Target": r["Target"],
                         "R:R": r["R:R"] if r["g_breakout"] else np.nan,
                         "Blockers": "; ".join(decisions.short_blocker(b) for b in decisions.blockers(r, q, stale))
                                     or "none",
                         "Becomes BUY if": "; ".join(decisions.becomes_buy(r, q, stale)),
                         "Becomes AVOID if": "; ".join(decisions.becomes_avoid(r))})
        if rows:
            new = pd.DataFrame(rows)
            log = new if log.empty else pd.concat([log, new], ignore_index=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            log.to_csv(path, index=False)
        return log

    def _tracking(self, recs: pd.DataFrame, done: dict[str, pd.DataFrame], session: date) -> dict:
        """Record today's BUY signals with the rule version, then follow every recorded signal."""
        fwd = ForwardRecord(Path(self.cfg.output_dir) / "forward_signals.csv")
        issued = recs[(recs["Action"] == "BUY") & (recs["Qty"] > 0)] if not recs.empty else recs
        fwd.record(issued, session, rule_version(self.cfg))
        results = fwd.evaluate(done, self.cfg.buy_quality, self.cfg.breakout.est_round_trip_cost_pct)
        fr = freeze_status(self.cfg, Path(self.cfg.data_home))
        closed = results[results["exit"].notna()] if not results.empty else results
        return {"start": fr.get("start") or (results["session"].min() if not results.empty else session.isoformat()),
                "results": results, "summary": summarize_forward(results), "closed": closed,
                "open": results[results["outcome"] == "open"] if not results.empty else results,
                "win_rate": float((closed["r_multiple"] > 0).mean()) if len(closed) else None,
                "avg_r": float(closed["r_multiple"].mean()) if len(closed) else None, "freeze": fr}

    def _alerts(self, recs: pd.DataFrame, session: date, now: datetime, data_warning: str | None,
                regime_d: dict | None, freeze: dict) -> None:
        center = AlertCenter(Path(self.cfg.output_dir) / "alerts.json")
        q = self.cfg.buy_quality
        if not recs.empty:
            for _, r in recs[(recs["Action"] == "BUY") & (recs["Qty"] > 0)].iterrows():
                center.add(f"buy:{session}:{r['Ticker']}", "buy",
                           f"BUY signal: {r['Ticker']} for the next open. Entry ₹{r['Breakout level']:,.2f}–"
                           f"₹{r['Price'] * (1 + q.entry_max_gap_pct / 100):,.2f}, stop ₹{r['Stop Loss']:,.2f}, "
                           f"target ₹{r['Target Price']:,.2f}, {int(r['Qty'])} shares.", now)
        if data_warning:
            center.add(f"stale:{now:%Y-%m-%d-%H}", "warn", data_warning, now)
        if regime_d:
            path = Path(self.cfg.output_dir) / "regime_state.txt"
            prev = path.read_text().strip() if path.exists() else None
            if prev != regime_d["state"]:
                if prev is not None:
                    center.add(f"regime:{session}:{regime_d['state']}", "watch",
                               f"Market regime changed: {prev} → {regime_d['state']} "
                               f"({regime_d['confidence']}% confidence).", now)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(regime_d["state"])
        if freeze.get("active") and freeze.get("drifted"):
            center.add(f"freeze_drift:{freeze['current_version']}", "warn",
                       f"Rules changed during the freeze (version {freeze['version']} → {freeze['current_version']}). "
                       "The forward record now reports the new version separately.", now)

    def _robustness(self, done, bench, brd, base_trades: pd.DataFrame, session: date) -> dict:
        """Parameter sensitivity and execution stress (weekly; several extra backtests)."""
        y, w, _ = session.isocalendar()
        key = hashlib.sha1(repr((CACHE_VERSION, sorted(done), self.cfg.buy_quality)).encode()).hexdigest()[:12]
        path = self.store / f"robustness_{y}-W{w:02d}_{key}.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001
                pass
        cost = self.cfg.breakout.est_round_trip_cost_pct
        q = self.cfg.buy_quality
        self.set_status("Robustness: re-running the BUY rules with nearby parameter values")
        sens = sensitivity(done, q, bench, cost, brd)
        self.set_status("Robustness: entering one session late")
        delayed, _ = backtest_quality(done, q, bench, cost, brd, delay=1)
        stress = execution_stress(base_trades, delayed, cost)
        out = {"sensitivity": sens, "sensitivity_verdict": sensitivity_verdict(sens), "stress": stress,
               "stress_verdict": stress_verdict(stress)}
        self.store.mkdir(parents=True, exist_ok=True)
        for old in sorted(self.store.glob("robustness_*.joblib"))[:-2]:
            old.unlink(missing_ok=True)
        joblib.dump(out, path)
        return out

    def _trials_and_dsr(self, heavy: dict) -> dict:
        """Count every distinct experiment and deflate the tuner's chosen setup for them."""
        exps = []
        opt = heavy.get("optimizer") or {}
        table = opt.get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            exps += [f"tune:{x}" for x in table["Setup"]]
        lab = heavy.get("lab_summary")
        if isinstance(lab, pd.DataFrame) and not lab.empty:
            exps += [f"lab:{v}" for v in lab["Variant"]]
        ab = heavy.get("ablation")
        if isinstance(ab, pd.DataFrame) and not ab.empty:
            exps += [f"ablation:{x}" for x in ab["Stage"]]
        rob = heavy.get("robustness") or {}
        sens = rob.get("sensitivity")
        if isinstance(sens, pd.DataFrame) and not sens.empty:
            exps += [f"sens:{p}={v}" for p, v in zip(sens["Parameter"], sens["Value"])]
        n = count_trials(Path(self.cfg.data_home) / "research_trials.json", exps)
        out = {"n_trials": n, "dsr": None}
        curves = opt.get("curves") or {}
        chosen = curves.get("Chosen setup")
        if isinstance(table, pd.DataFrame) and not table.empty and isinstance(chosen, pd.DataFrame) and not chosen.empty:
            hz = table["Setup"].str.extract(r"(\d+)-day")[0].astype(float)
            per_period = (table["sel_sharpe"] / np.sqrt(252 / hz)).dropna().tolist()
            out["dsr"] = deflated_sharpe(chosen["ret"], per_period, n)
        return out

    def _optimizer(self, panel: pd.DataFrame, done: dict[str, pd.DataFrame], bench: pd.DataFrame | None,
                   session: date) -> dict[str, Any]:
        """Weekly tuning (the costly part), cached per ISO week and universe."""
        y, w, _ = session.isocalendar()
        key = hashlib.sha1(repr((CACHE_VERSION, sorted(done), self.cfg.ranker)).encode()).hexdigest()[:12]
        path = self.store / f"optimizer_{y}-W{w:02d}_{key}.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001
                logger.warning("Cached tuning unreadable; re-running")
        opt = optimize(panel, done, bench, self.cfg.ranker, status=self.set_status)
        self.store.mkdir(parents=True, exist_ok=True)
        for old in sorted(self.store.glob("optimizer_*.joblib"))[:-2]:
            old.unlink(missing_ok=True)
        joblib.dump(opt, path)
        return opt

    def _recommend(self, heavy: dict, done: dict[str, pd.DataFrame], regime: dict | None, sect: pd.DataFrame,
                   superstar: set[str], session: date, quality: pd.DataFrame | None = None) -> pd.DataFrame:
        """BUY = stocks passing every quality check (and the model, when validated); WATCH = top-ranked
        stocks that failed a check; SELL = bottom-ranked stocks."""
        scores: pd.DataFrame = heavy["scores"]
        q = self.cfg.buy_quality
        rk = self.cfg.ranker
        strategy: Settings = heavy.get("settings") or Settings(rk.horizon_days, rk.top_n)
        rk = dc_replace(rk, horizon_days=strategy.horizon, top_n=strategy.top_n)
        val = heavy.get("validation") or {}
        validated = bool(val.get("validated"))
        cal = val.get("calibration")
        cal = cal.set_index("decile") if isinstance(cal, pd.DataFrame) and not cal.empty else None
        contrib = heavy.get("contrib")
        bottom_sectors = set(sect["Sector"].tail(3)) if len(sect) > 3 else set()
        quality = quality if quality is not None else pd.DataFrame()
        qidx = quality.set_index("Ticker") if not quality.empty else pd.DataFrame()

        def model_cols(tk: str, direction: int) -> dict:
            if scores.empty or tk not in scores.index:
                return {"Model score": np.nan, "Hist. excess %": np.nan, "Hist. beat %": np.nan, "Why": ""}
            r = scores.loc[tk]
            decile = min(int(r["score"] * 10) + 1, 10)
            ok = cal is not None and decile in cal.index
            why = explain(r, contrib.loc[tk] if contrib is not None and tk in contrib.index else None, direction)
            return {"Model score": round(100 * r["score"], 1),
                    "Hist. excess %": float(cal.at[decile, "avg_excess_pct"]) if ok else np.nan,
                    "Hist. beat %": float(cal.at[decile, "beat_median_pct"]) if ok else np.nan, "Why": "; ".join(why)}

        rows, plans = [], {}
        # ------------------------------------------------------------ BUY: every gate must pass
        if not qidx.empty:
            cand = qidx[qidx["passes"]].copy()
            if validated and not scores.empty:
                sc = scores["score"].reindex(cand.index)
                cand = cand[sc >= q.min_model_score]
                cand["_score"] = sc
            else:
                cand["_score"] = scores["score"].reindex(cand.index) if not scores.empty else np.nan
            cand = cand.sort_values(["_score", "R:R"], ascending=False, na_position="last")
            review = (pd.Timestamp(session) + pd.offsets.BDay(q.max_hold_days)).date()
            for tk, r in cand.iterrows():
                try:
                    plan = self.risk.plan_from_levels(tk, r["Close"], r["Stop"], r["Target"], r["ATR"])
                    plans[("BUY", tk)] = plan
                    qty, risk_amt = plan.units, plan.risk_amount
                except RiskError:
                    qty, risk_amt = 0, np.nan
                mc = model_cols(tk, +1)
                sc = r["_score"]
                if not validated:
                    conviction = "Unvalidated"
                else:
                    conviction = LEVELS[2 if sc >= 0.9 and r["R:R"] >= 3 else 1 if sc >= 0.7 else 0]
                rows.append({"Action": "BUY", "Ticker": tk,
                             "Sector": scores.at[tk, "sector"] if not scores.empty and tk in scores.index else "",
                             "Conviction": conviction, **{k: mc[k] for k in ("Model score", "Hist. excess %", "Hist. beat %")},
                             "Price": r["Close"], "Stop Loss": r["Stop"], "Target Price": r["Target"], "R:R": r["R:R"],
                             "Qty": qty, "Risk (INR)": risk_amt, "Allocation": "",
                             "Entry rule": (f"Next open only, between ₹{r['Breakout level']:,.2f} and "
                                            f"₹{r['Close'] * (1 + q.entry_max_gap_pct / 100):,.2f}"),
                             "Review by": review.isoformat(), "Why": mc["Why"],
                             "Checks": "all passed" + (" (target capped by 52-week high)" if r["Target capped by 52w high"]
                                                       else f" (target at the {q.target_max_atr:g}-ATR cap)"
                                                       if r.get("Target at ATR cap") else ""),
                             "Breakout level": r["Breakout level"], "ATR": r["ATR"],
                             "Superstar": "Yes" if tk in superstar else ""})
        # ------------------------------------------------------------ WATCH: top-ranked but failed a check
        if not scores.empty:
            bought = {r["Ticker"] for r in rows}
            top = scores[scores["score"] >= 0.9].head(rk.top_n * 2)
            for tk, r in top.iterrows():
                if tk in bought:
                    continue
                failed = qidx.at[tk, "Failed"] if not qidx.empty and tk in qidx.index else "not enough history"
                if validated and not qidx.empty and tk in qidx.index and qidx.at[tk, "passes"]:
                    failed = "model score"
                mc = model_cols(tk, +1)
                rows.append({"Action": "WATCH", "Ticker": tk, "Sector": r["sector"],
                             "Conviction": "Watch", **{k: mc[k] for k in ("Model score", "Hist. excess %", "Hist. beat %")},
                             "Price": round(float(r["close"]), 2), "Stop Loss": np.nan, "Target Price": np.nan,
                             # reward/risk is measured from the breakout level, so it only means something after a breakout
                             "R:R": (qidx.at[tk, "R:R"] if not qidx.empty and tk in qidx.index
                                     and qidx.at[tk, "g_breakout"] else np.nan),
                             "Qty": 0, "Risk (INR)": 0.0, "Allocation": "Watch only", "Entry rule": "",
                             "Review by": "", "Why": mc["Why"], "Checks": f"failed: {failed}",
                             "Breakout level": np.nan, "ATR": np.nan, "Superstar": "Yes" if tk in superstar else ""})
            # -------------------------------------------------------- SELL: bottom-ranked
            sells = scores[scores["score"] <= 0.1].sort_values("score").head(rk.top_n)
            review_s = (pd.Timestamp(session) + pd.offsets.BDay(rk.horizon_days)).date()
            for tk, r in sells.iterrows():
                df = done[tk]
                a = float(atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
                try:
                    plan = self.risk.build_trade_plan(tk, float(r["close"]), a, Side.SHORT)
                    stop, target, qty, risk_amt, rr = (plan.stop_loss, plan.take_profit, plan.units,
                                                       plan.risk_amount, plan.reward_risk_ratio)
                except RiskError:
                    stop = target = risk_amt = rr = np.nan
                    qty = 0
                conf = [c for c, ok in (("below 50-day avg", r["c_sma50"] < 0), ("below 200-day avg", r["c_sma200"] < 0),
                                        ("bottom-3 sector", r["sector"] in bottom_sectors)) if ok]
                level = 2 if (1 - r["score"]) >= 0.97 and len(conf) >= 2 else 1 if (1 - r["score"]) >= 0.93 else 0
                mc = model_cols(tk, -1)
                rows.append({"Action": "SELL", "Ticker": tk, "Sector": r["sector"],
                             "Conviction": LEVELS[level] if validated else "Unvalidated",
                             **{k: mc[k] for k in ("Model score", "Hist. excess %", "Hist. beat %")},
                             "Price": round(float(r["close"]), 2), "Stop Loss": stop, "Target Price": target, "R:R": rr,
                             "Qty": qty, "Risk (INR)": risk_amt, "Allocation": "", "Entry rule": "",
                             "Review by": review_s.isoformat(), "Why": mc["Why"], "Checks": ", ".join(conf),
                             "Breakout level": np.nan, "ATR": a, "Superstar": "Yes" if tk in superstar else ""})
        recs = pd.DataFrame(rows, columns=REC_COLUMNS)
        if q.regime_mode == "three_state" and not qidx.empty:
            for key in list(plans):
                tk = key[1]
                if tk in qidx.index and qidx.at[tk, "Regime state"] == "neutral":
                    p0 = plans[key]
                    half_units = p0.units // 2
                    plans[key] = type(p0)(**{**p0.__dict__, "units": half_units,
                                             "notional": round(half_units * p0.entry_price, 2),
                                             "risk_amount": round(half_units * p0.risk_per_share, 2),
                                             "binding_constraint": "neutral_half_size"})
        recs = apply_portfolio_limits(recs, plans, self.risk, done, getattr(self, "_held", []))
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        recs.to_csv(out_dir / f"intelligence_{session.isoformat()}.csv", index=False)
        return recs


MORNING_COLUMNS = ["Action", "Ticker", "Conviction", "Yesterday close", "Today price", "Gap %", "Status",
                   "Entry", "Stop Loss", "Target Price", "Qty", "Risk (INR)"]


def _morning_status(action: str, gap: float, price: float, old_stop: float, max_gap: float) -> str:
    if action == "BUY":
        if np.isfinite(old_stop) and price <= old_stop:
            return "SKIP: opened below the planned stop"
        if gap > max_gap:
            return f"SKIP: gapped up {gap:.1f}% (chasing adds risk)"
        if gap < -max_gap:
            return f"CAUTION: gapped down {gap:.1f}%"
        return "OK"
    if np.isfinite(old_stop) and price >= old_stop:
        return "SKIP: opened above the planned stop"
    if gap < -max_gap:
        return f"SKIP: gapped down {gap:.1f}% (move already happened)"
    if gap > max_gap:
        return f"CAUTION: gapped up {gap:.1f}%"
    return "OK"


def morning_check(engine: "IntelligenceEngine", rep: IntelligenceReport, now: datetime | None = None,
                  min_interval: int = 240, fetch: bool = True) -> tuple[pd.DataFrame | None, str]:
    """Re-check the session's ideas against today's pre-open (09:00-09:15) or live price.

    Returns ``(table, note)``. The table recalculates entry, stop, target and quantity from today's
    price and marks ideas to skip when the gap makes the planned trade invalid or too risky.
    Throttled to one fetch per ``min_interval`` seconds.
    """
    now = (now or datetime.now(IST)).astimezone(IST)
    phase, _ = market_phase(engine.cfg, now)
    if phase not in ("pre_market", "open", "settling") or rep.recommendations.empty:
        return None, ""
    if rep.session >= now.date():
        return None, ""  # ideas already include today's close
    t = now.time()
    if t < dtime(9, 0):
        return None, "Pre-open starts at 09:00 IST; the ideas will be re-checked against it automatically."
    cached = engine._check
    if cached and cached[1] == now.date() and (not fetch or (now - cached[0]).total_seconds() < min_interval):
        return cached[2], cached[3]
    if not fetch:
        return None, "Checking today's prices; this appears within a minute."

    ideas = rep.recommendations[rep.recommendations["Action"].isin(["BUY", "SELL"])]
    if ideas.empty:
        return None, ""
    tickers = ideas["Ticker"].tolist()
    prices: dict[str, float] = {}
    try:
        if t < dtime(9, 15):
            quotes = engine.fetcher.fetch_preopen(tickers)
            prices = {s: q.iep for s, q in quotes.items()}
            src = "pre-open IEP (final)" if t >= dtime(9, 8) else "pre-open IEP (indicative until 09:08)"
        else:
            live = engine.fetcher.fetch_many(tickers, lookback_days=7, now=now)
            # the rule is "entry only at the open", so after 09:15 the check uses today's opening price
            prices = {s: float(df["Open"].iloc[-1]) for s, df in live.items()
                      if df.index[-1] == pd.Timestamp(now.date())}
            src = "today's opening price"
    except Exception as exc:  # noqa: BLE001 - the check is advisory; never break the page
        logger.warning("Morning check failed: %s", exc)
        return None, f"Today's prices are unavailable right now ({exc}); retrying automatically."

    rows = []
    max_gap = engine.cfg.ranker.max_entry_gap_pct
    for _, r in ideas.iterrows():
        tk, px = r["Ticker"], prices.get(r["Ticker"])
        if px is None or not np.isfinite(px):
            rows.append({"Action": r["Action"], "Ticker": tk, "Conviction": r["Conviction"],
                         "Yesterday close": r["Price"], "Today price": np.nan, "Gap %": np.nan,
                         "Status": "No price yet", "Entry": np.nan, "Stop Loss": np.nan, "Target Price": np.nan,
                         "Qty": 0, "Risk (INR)": np.nan})
            continue
        gap = 100 * (px / r["Price"] - 1)
        df = rep.histories[tk]
        a = float(atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
        if r["Action"] == "BUY":  # quality next-day entry rule, levels recomputed from today's price
            pe = plan_entry(r["Breakout level"], r["Price"], r["ATR"], r["Target Price"], px, engine.cfg.buy_quality)
            status = pe["status"]
            try:
                plan = engine.risk.plan_from_levels(tk, px, pe["stop"], pe["target"], r["ATR"])
                stop, target, qty, risk_amt = plan.stop_loss, plan.take_profit, plan.units, plan.risk_amount
            except (RiskError, ValueError):
                stop, target, qty, risk_amt = pe["stop"], pe["target"], 0, np.nan
        else:
            status = _morning_status(r["Action"], gap, px, r["Stop Loss"], max_gap)
            try:
                plan = engine.risk.build_trade_plan(tk, px, a, Side.SHORT)
                stop, target, qty, risk_amt = plan.stop_loss, plan.take_profit, plan.units, plan.risk_amount
            except RiskError:
                stop = target = risk_amt = np.nan
                qty = 0
        if status.startswith("SKIP"):
            qty, risk_amt = 0, 0.0
        rows.append({"Action": r["Action"], "Ticker": tk, "Conviction": r["Conviction"],
                     "Yesterday close": r["Price"], "Today price": round(px, 2), "Gap %": round(gap, 2),
                     "Status": status, "Entry": round(px, 2), "Stop Loss": stop, "Target Price": target,
                     "Qty": qty, "Risk (INR)": risk_amt})
    table = pd.DataFrame(rows, columns=MORNING_COLUMNS)
    tradable = [i for i in table.index if table.at[i, "Action"] == "BUY" and table.at[i, "Qty"] > 0]
    plans = []
    for i in tradable:
        tk = table.at[i, "Ticker"]
        df = rep.histories[tk]
        a = float(atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
        try:
            plans.append(engine.risk.plan_from_levels(tk, float(table.at[i, "Entry"]), float(table.at[i, "Stop Loss"]),
                                                      float(table.at[i, "Target Price"]), a))
        except (RiskError, ValueError):
            pass
    allocated = {p.ticker: p for p in engine.risk.allocate(plans)}
    for i in tradable:
        p = allocated.get(table.at[i, "Ticker"])
        if p is not None:
            table.at[i, "Qty"], table.at[i, "Risk (INR)"] = p.units, p.risk_amount
            if p.units == 0:
                table.at[i, "Status"] = "WATCH: portfolio limit reached"
    note = f"Checked against today's {src} at {now:%H:%M} IST."
    engine._check = (now, now.date(), table, note)
    center = AlertCenter(Path(engine.cfg.output_dir) / "alerts.json")
    stage = "preopen" if t < dtime(9, 15) else "open"
    lead = "Pre-open indication" if stage == "preopen" else "At the open"
    for _, r in table[table["Action"] == "BUY"].iterrows():
        key = f"{stage}:{now.date()}:{r['Ticker']}"   # one alert per stock per stage per day
        if str(r["Status"]).startswith("SKIP"):
            center.add(key, "skip", f"{lead}: skip {r['Ticker']} today; {r['Status'][6:]}.", now)
        elif r["Status"] == "OK" and r["Qty"] > 0:
            center.add(key, "ok", f"{lead}: {r['Ticker']} is inside its entry range at ₹{r['Today price']:,.2f}; "
                                  f"stop ₹{r['Stop Loss']:,.2f}, {int(r['Qty'])} shares.", now)
    out = Path(engine.cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"morning_check_{now.date().isoformat()}.csv", index=False)
    return table, note
