"""Daily NSE pre-open pipeline orchestrator.

Flow: kill switch -> session-timing checks -> daily history -> pre-open
quotes (IEP) -> store pre-open quantities -> technical screen -> load or
retrain model -> score candidates -> size trades -> portfolio limits ->
summary -> watch-list log -> (optional) paper ledger.

Best run between 09:08 (end of price discovery) and 09:15 IST.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime
from typing import Any

import numpy as np
import pandas as pd

from config.config import AppConfig
from data_ingestion.data_fetcher import IST, DataFetchError, MarketDataFetcher, completed_sessions
from data_ingestion.preopen import PreOpenHistory, PreOpenQuote
from models.ml_engine import MomentumClassifier, build_inference_row, train_and_evaluate
from monitoring.performance_log import PerformanceTracker
from risk_engine.guardrails import ExecutionGuard, PaperLedger
from risk_engine.risk_manager import RiskError, RiskManager, TradePlan
from screeners.screener import scan_universe

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = ["Ticker", "Signal", "Current Price", "Target Price", "Stop Loss", "ATR",
                   "Model Probability Score", "Max Position Units",
                   "Gap %", "RVOL", "RVOL Source", "RSI", "Notional (INR)", "Risk (INR)",
                   "Est. Cost (INR)", "R:R", "Constraint"]


def _hhmm(value: str) -> dtime:
    h, m = (int(x) for x in value.split(":"))
    return dtime(h, m)


def session_status(cfg: AppConfig, now: datetime) -> tuple[bool, str]:
    """Whether ``now`` is an NSE trading day, and a note about the timing.

    Returns:
        ``(is_trading_day, message)``. The message explains weekends,
        configured holidays, or a run outside the 09:08-09:15 window.
    """
    mk = cfg.market
    today = now.date()
    if today.weekday() >= 5:
        return False, f"{today:%A} - NSE is closed on weekends."
    if today.isoformat() in set(mk.holidays):
        return False, f"{today} is listed in market.holidays - NSE is closed."
    t = now.time()
    if t < _hhmm(mk.preopen_start):
        return True, "Pre-open has not started (09:00 IST); quotes will be yesterday's or empty."
    if t < _hhmm(mk.price_discovery_end):
        return True, "Price discovery is still running; the IEP is not final until ~09:08 IST."
    if t >= _hhmm(mk.session_open):
        return True, "The market is already open; entries at the IEP may no longer be available."
    return True, "Inside the 09:08-09:15 IST window: the IEP is final."


@dataclass
class PipelineResult:
    """Everything a run produced."""

    run_time: datetime
    summary: pd.DataFrame
    scan: pd.DataFrame
    cv_report: dict[str, Any] | None = None
    orders: list[dict[str, Any]] = field(default_factory=list)
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    timing_note: str = ""


class DailyPipeline:
    """Runs the full NSE pre-open workflow.

    Args:
        config: Validated application config.
        fetcher: Inject a fetcher (tests/demo); built from config otherwise.
        now: Evaluation time (defaults to now in IST).
    """

    def __init__(self, config: AppConfig, fetcher: MarketDataFetcher | None = None,
                 now: datetime | None = None) -> None:
        self.cfg = config
        self.now = (now or datetime.now(IST)).astimezone(IST)
        self.fetcher = fetcher or MarketDataFetcher(config.data)
        self.risk = RiskManager(config.risk)
        self.guard = ExecutionGuard(config.execution, config.risk, self.risk.equity)
        self.tracker = PerformanceTracker(config.output_dir, config.risk.est_round_trip_cost_pct)
        self.preopen_history = PreOpenHistory(config.data.preopen_history_path)

    @property
    def session_date(self) -> date:
        return self.now.date()

    # ----------------------------------------------------------------- steps
    def load_histories(self) -> dict[str, pd.DataFrame]:
        raw = self.fetcher.fetch_many(self.cfg.tickers, now=self.now)
        return {t: h for t, df in raw.items() if not (h := completed_sessions(df, self.session_date)).empty}

    def load_quotes(self, symbols: list[str]) -> dict[str, PreOpenQuote]:
        """Fetch pre-open quotes, store today's quantities, attach their recent averages."""
        quotes = self.fetcher.fetch_preopen(symbols)
        ind = self.cfg.indicators
        enriched = {s: replace(q, avg_preopen_quantity=self.preopen_history.average_quantity(
                        s, self.session_date, ind.rvol_lookback, ind.min_preopen_history_days))
                    for s, q in quotes.items()}
        if quotes:
            self.preopen_history.record(quotes, self.session_date)
        n_with = sum(np.isfinite(q.avg_preopen_quantity) for q in enriched.values())
        if n_with < len(enriched):
            logger.info("Pre-open RVOL available for %d/%d symbols; others use prior-session RVOL "
                        "until %d days of pre-open history are stored", n_with, len(enriched),
                        ind.min_preopen_history_days)
        return enriched

    def get_model(self, histories: dict[str, pd.DataFrame], retrain: bool
                  ) -> tuple[MomentumClassifier | None, dict | None]:
        mcfg = self.cfg.model
        if not retrain and mcfg.model_path.exists():
            try:
                model = MomentumClassifier.load(mcfg)
                if not model.is_stale(mcfg.max_model_age_days):
                    logger.info("Loaded model trained %s", model.metadata_.get("trained_at"))
                    return model, None
                logger.info("Model is stale; retraining")
            except Exception:  # noqa: BLE001
                logger.exception("Could not load model; retraining")
        try:
            model, report = train_and_evaluate(histories, self.cfg.indicators, mcfg)
            model.save()
            return model, report
        except ValueError as exc:
            logger.error("Model unavailable (%s); signals will be WATCH only", exc)
            return None, None

    def score(self, candidates: pd.DataFrame, histories: dict[str, pd.DataFrame],
              model: MomentumClassifier | None) -> pd.Series:
        probs: dict[str, float] = {}
        for _, row in candidates.iterrows():
            if model is None:
                probs[row["ticker"]] = np.nan
                continue
            feats = build_inference_row(histories[row["ticker"]], row["price"], self.session_date,
                                        self.cfg.indicators)
            probs[row["ticker"]] = float(model.predict_proba(feats)[0])
        return pd.Series(probs, dtype=float)

    def build_summary(self, candidates: pd.DataFrame, probs: pd.Series
                      ) -> tuple[pd.DataFrame, list[TradePlan]]:
        thr = self.cfg.model.probability_threshold
        plans: dict[str, TradePlan] = {}
        for _, row in candidates.iterrows():
            try:
                plans[row["ticker"]] = self.risk.build_trade_plan(row["ticker"], row["price"], row["atr"])
            except RiskError as exc:
                logger.warning("%s: no trade plan (%s)", row["ticker"], exc)

        longs = sorted((t for t in plans if probs.get(t, np.nan) >= thr), key=lambda t: -probs[t])
        allocated = {p.ticker: p for p in self.risk.allocate([plans[t] for t in longs])}

        rows = []
        for _, row in candidates.iterrows():
            t = row["ticker"]
            plan = allocated.get(t, plans.get(t))
            if plan is None:
                continue
            p = probs.get(t, np.nan)
            signal = ("LONG" if plan.units > 0 else "SKIP_RISK_LIMIT") if t in allocated else "WATCH"
            rows.append({
                "Ticker": t, "Signal": signal, "Current Price": plan.entry_price,
                "Target Price": plan.take_profit, "Stop Loss": plan.stop_loss, "ATR": plan.atr,
                "Model Probability Score": round(p, 4) if np.isfinite(p) else np.nan,
                "Max Position Units": plan.units, "Gap %": round(row["gap_pct"], 2),
                "RVOL": round(row["rvol"], 2), "RVOL Source": row["rvol_source"], "RSI": round(row["rsi"], 1),
                "Notional (INR)": plan.notional, "Risk (INR)": plan.risk_amount,
                "Est. Cost (INR)": plan.est_cost, "R:R": plan.reward_risk_ratio,
                "Constraint": plan.binding_constraint,
            })
        summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        order = {"LONG": 0, "SKIP_RISK_LIMIT": 1, "WATCH": 2}
        summary = summary.sort_values(["Signal", "Model Probability Score"],
                                      key=lambda s: s.map(order) if s.name == "Signal" else -s.fillna(-1))
        return summary.reset_index(drop=True), [p for p in allocated.values() if p.units > 0]

    # ------------------------------------------------------------------- run
    def run(self, retrain: bool = False, record_paper_orders: bool = False) -> PipelineResult:
        """Execute the full workflow and return its outputs."""
        self.guard.check_kill_switch()
        _, note = session_status(self.cfg, self.now)
        logger.info("=== NSE pre-open run %s IST | %d symbols | history=%s pre-open=%s ===",
                    self.now.isoformat(timespec="minutes"), len(self.cfg.tickers),
                    self.fetcher.provider.name, self.fetcher.preopen.name)
        logger.info(note)
        histories = self.load_histories()
        if not histories:
            raise DataFetchError("No usable history for any symbol")
        quotes = self.load_quotes(list(histories))
        scan = scan_universe(histories, quotes, self.cfg.indicators, self.cfg.screener)
        candidates = scan[scan["passed"]] if not scan.empty else scan
        logger.info("Screen: %d/%d passed", len(candidates), len(scan))

        model, cv_report = self.get_model(histories, retrain)
        probs = self.score(candidates, histories, model)
        summary, long_plans = self.build_summary(candidates, probs)
        self.tracker.log_watchlist(summary, self.session_date)
        self.tracker.log_scan(scan, self.session_date)

        orders: list[dict[str, Any]] = []
        if record_paper_orders and long_plans:
            orders = PaperLedger(self.cfg.output_dir, self.guard).record(long_plans, self.session_date)
        return PipelineResult(self.now, summary, scan, cv_report, orders, histories, note)
