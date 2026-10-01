"""Hands-free signal engine behind the live dashboard.

Every refresh it:
  1. resolves the universe (Nifty 500 list, re-downloaded weekly; fallback: config tickers)
  2. picks up the newest Superstar-investor CSV dropped into ``data/superstar``
  3. syncs daily history through the on-disk cache (incremental after the first run)
  4. computes CONFIRMED signals on the last completed session, plus the backtest of the rules
  5. while the market is open, computes PROVISIONAL signals on today's forming bar

What is shown depends on the time of day (IST):
  market open 09:15-16:00  -> live provisional signals (they can disappear before the close)
  after 16:00              -> today's confirmed signals, for the next session
  before 09:15 / holidays  -> the last session's confirmed signals, for the coming session
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import AppConfig
from data_ingestion.cache import DailyCache
from data_ingestion.data_fetcher import IST, DataFetchError, MarketDataFetcher
from data_ingestion.universe import latest_csv, load_symbols, refresh_nifty500
from monitoring.breakout_backtest import backtest_breakouts
from pipeline import session_status
from pipeline_eod import EodBreakoutPipeline
from monitoring.breakout_backtest import summarize as summarize_rules
from risk_engine.risk_manager import RiskError, RiskManager
from screeners.breakout import scan_breakouts
from screeners.buy_quality import backtest_quality, evaluate_latest

logger = logging.getLogger(__name__)
MIN_TRADES = 30


def market_phase(cfg: AppConfig, now: datetime) -> tuple[str, str]:
    """``(phase, description)`` where phase is closed | pre_market | open | settling | after_close."""
    trading, note = session_status(cfg, now)
    if not trading:
        return "closed", note
    t = now.time()
    hh, mm = (int(x) for x in cfg.breakout.eod_ready_time.split(":"))
    if t < dtime(9, 15):
        return "pre_market", "Before the open. Showing the last session's confirmed signals."
    if t < dtime(15, 30):
        return "open", "Market open. Signals are live and provisional until the close."
    if t < dtime(hh, mm):
        return "settling", "Market closed; waiting for final daily data. Signals are still provisional."
    return "after_close", "Market closed. Today's signals are confirmed for the next session."


@dataclass
class LiveSnapshot:
    as_of: datetime
    phase: str
    phase_note: str
    mode: str                     # "live" (provisional) or "confirmed"
    session: date                 # session the displayed signals were computed on
    signals: pd.DataFrame         # displayed BUY/SELL table
    confirmed: pd.DataFrame       # last completed session's signals
    confirmed_session: date
    rule_stats: pd.DataFrame
    trades: pd.DataFrame
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    universe_label: str = ""
    superstar_label: str = ""
    skipped: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    movers: pd.DataFrame | None = None

    def rule_health(self, rule: str) -> tuple[str, str]:
        """``(status, message)`` for 'BUY' or 'SELL': proven | losing | unproven | none."""
        row = self.rule_stats[self.rule_stats["Rule"].str.startswith(rule)] if not self.rule_stats.empty else None
        if row is None or row.empty:
            return "none", f"The {rule} rule has no historical trades on this universe."
        r = row.iloc[0]
        if r["Trades"] < MIN_TRADES:
            return "unproven", (f"The {rule} rule has only {r['Trades']} historical trades here: too few to "
                                "know whether it works. Treat its signals as a watch list.")
        if r["Avg R"] <= 0:
            return "losing", (f"The {rule} rule has LOST money historically on this universe "
                              f"({r['Avg R']:+.2f}R per trade over {r['Trades']} trades, after costs). "
                              "Treat its signals as a watch list, not trades.")
        pf = r.get("Profit factor", np.nan)
        if r["Avg R"] < 0.1 or not (pf > 1.3):
            return "unproven", (f"The {rule} rule has been roughly break-even historically on this universe "
                                f"({r['Avg R']:+.2f}R per trade, profit factor {pf:.2f}, {r['Trades']} trades, after "
                                "costs). Not enough edge to rely on; treat its signals as a watch list.")
        return "proven", (f"The {rule} rule made {r['Avg R']:+.2f}R per trade on average over {r['Trades']} "
                          f"historical trades ({r['Win rate %']:.0f}% winners, after costs). Past results can fail.")


def todays_movers(frames: dict[str, pd.DataFrame], now: datetime, min_turnover_cr: float = 10.0,
                  n: int = 15) -> pd.DataFrame:
    """Today's top gainers trading at least their normal volume pace (intraday, provisional)."""
    today = pd.Timestamp(now.date())
    elapsed = min(max((now.hour * 60 + now.minute - 555) / 375, 0.05), 1.0)  # share of the 09:15-15:30 session
    rows = []
    for sym, df in frames.items():
        if len(df) < 25 or df.index[-1] != today:
            continue
        prev, bar = df.iloc[-2], df.iloc[-1]
        avg_vol = float(df["Volume"].iloc[-21:-1].mean())
        turnover = float((df["Close"] * df["Volume"]).iloc[-21:-1].mean() / 1e7)
        if avg_vol <= 0 or turnover < min_turnover_cr:
            continue
        pace = float(bar["Volume"]) / (avg_vol * elapsed)
        chg = float(bar["Close"] / prev["Close"] - 1)
        if pace >= 1.0 and chg > 0:
            rng = bar["High"] - bar["Low"]
            rows.append({"Ticker": sym, "Price": round(float(bar["Close"]), 2), "Today %": round(100 * chg, 2),
                         "Volume pace": round(pace, 2),
                         "Near day high": round(float((bar["Close"] - bar["Low"]) / rng), 2) if rng > 0 else 1.0})
    out = pd.DataFrame(rows)
    return out.sort_values("Today %", ascending=False).head(n).reset_index(drop=True) if not out.empty else out


class LiveSignalEngine:
    """Thread-safe engine with its own refresh throttle and backtest cache."""

    def __init__(self, cfg: AppConfig, fetcher: MarketDataFetcher | None = None, auto_universe: bool = True) -> None:
        self.cfg = cfg
        self.fetcher = fetcher or MarketDataFetcher(cfg.data)
        self.cache = DailyCache(Path(cfg.cache_dir))
        self.auto_universe = auto_universe and cfg.auto_universe == "nifty500" and not cfg.universe_file
        self._lock = threading.Lock()
        self._last: LiveSnapshot | None = None
        self.status = "Starting"
        self.status_since = datetime.now(IST)
        self.last_error: str | None = None
        self._bt: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._quality: pd.DataFrame | None = None

    # -------------------------------------------------------------- inputs
    def resolve_universe(self) -> tuple[list[str], str, list[str]]:
        if self.cfg.universe_file:
            syms, skipped = load_symbols(Path(self.cfg.universe_file))
            return syms, f"{Path(self.cfg.universe_file).name} ({len(syms)} stocks)", skipped
        if self.auto_universe:
            path = Path(self.cfg.cache_dir) / "universe" / "ind_nifty500list.csv"
            if refresh_nifty500(path):
                syms, skipped = load_symbols(path)
                updated = datetime.fromtimestamp(path.stat().st_mtime, IST)
                return syms, f"Nifty 500 ({len(syms)} stocks, list updated {updated:%d %b})", skipped
            logger.warning("Nifty 500 list unavailable; falling back to configured tickers")
        return list(self.cfg.tickers), f"{len(self.cfg.tickers)} configured large caps", []

    def resolve_superstar(self) -> tuple[set[str], str, list[str]]:
        path = latest_csv(Path(self.cfg.superstar_dir)) or (
            Path(self.cfg.superstar_file) if self.cfg.superstar_file else None)
        if path is None or not path.exists():
            return set(), "No Superstar file (drop a Trendlyne export into data/superstar)", []
        syms, skipped = load_symbols(path)
        return set(syms), f"{path.name} ({len(syms)} stocks)", skipped

    # ------------------------------------------------------------- refresh
    def _min_interval(self, phase: str) -> int:
        return self.cfg.live_refresh_seconds if phase in ("open", "settling") else 1800

    def set_status(self, msg: str) -> None:
        if msg != self.status:
            self.status, self.status_since = msg, datetime.now(IST)

    def peek(self) -> LiveSnapshot | None:
        """Latest snapshot without waiting (None until the first refresh completes)."""
        return self._last

    def refresh(self, now: datetime | None = None, force: bool = False) -> LiveSnapshot:
        """Return a fresh snapshot, or the cached one if refreshed recently."""
        now = (now or datetime.now(IST)).astimezone(IST)
        phase, note = market_phase(self.cfg, now)
        with self._lock:
            last = self._last
            if (not force and last is not None and last.phase == phase
                    and (now - last.as_of).total_seconds() < self._min_interval(phase)):
                return last
            try:
                snap = self._compute(now, phase, note)
            except Exception as exc:
                self.last_error = str(exc)
                self.set_status(f"Last refresh failed ({exc}); retrying automatically")
                raise
            self._last, self.last_error = snap, None
            self.set_status("Ready")
            return snap

    def _compute(self, now: datetime, phase: str, note: str) -> LiveSnapshot:
        self.set_status("Loading the stock universe")
        symbols, uni_label, skipped = self.resolve_universe()
        superstar, ss_label, ss_skipped = self.resolve_superstar()
        scan_symbols = list(dict.fromkeys(symbols + sorted(superstar)))
        frames = self.cache.sync(self.fetcher, scan_symbols + [self.cfg.benchmark], now,
                                 self.cfg.data.history_lookback_days, status=self.set_status)
        bench_full = frames.pop(self.cfg.benchmark, None)
        if not frames:
            raise DataFetchError("No market data could be loaded. Check your internet connection.")

        today = pd.Timestamp(now.date())
        final_today = phase == "after_close"
        completed = {s: df[df.index <= today] if final_today else df[df.index < today] for s, df in frames.items()}
        completed = {s: df for s, df in completed.items() if not df.empty}
        conf_session = max(df.index[-1] for df in completed.values()).date()
        bench = None
        if bench_full is not None:
            bench = bench_full[bench_full.index <= today] if final_today else bench_full[bench_full.index < today]
        stale = [s for s, df in completed.items() if df.index[-1].date() < conf_session]

        self.set_status(f"Scanning {len(completed)} stocks and backtesting the breakout rules")
        trades, stats = self._backtest(completed, conf_session, bench)
        builder = EodBreakoutPipeline(self.cfg, self.fetcher, now)
        conf_scan = self._quality_buys(scan_breakouts(completed, self.cfg.breakout), completed, bench)
        confirmed = self._apply_levels(builder._build_signals(conf_scan, trades, stats, superstar, conf_session, stale))

        mode, session, shown = "confirmed", conf_session, confirmed
        if phase in ("open", "settling"):
            live = {s: df for s, df in frames.items() if df.index[-1] == today}
            if live:
                live_bench = bench_full if bench_full is not None else None
                live_scan = self._quality_buys(scan_breakouts(live, self.cfg.breakout), live, live_bench)
                shown = self._apply_levels(builder._build_signals(live_scan, trades, stats, superstar, now.date(), []))
                mode, session = "live", now.date()
            else:
                note += " No intraday data yet from the feed; showing the last confirmed list."

        shown = self._add_rs(shown, frames, bench_full, now, mode == "live")
        confirmed = self._add_rs(confirmed, frames, bench_full, now, False)
        movers = todays_movers(frames, now, self.cfg.buy_quality.min_turnover_cr) if phase in ("open", "settling") else None
        if mode == "live" and not shown.empty:
            from alerts import AlertCenter
            center = AlertCenter(Path(self.cfg.output_dir) / "alerts.json")
            for t in shown.loc[shown["Signal"] == "BUY", "Ticker"]:
                center.add(f"live:{now.date()}:{t}", "watch",
                           f"{t} is breaking out now with every check passing (provisional until the close).", now)
        if phase == "open":
            from research.movers import record_movers
            record_movers(movers, now, Path(self.cfg.output_dir) / "movers_log.csv")
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        confirmed.to_csv(out / f"eod_signals_{conf_session.isoformat()}.csv", index=False)
        return LiveSnapshot(now, phase, note, mode, session, shown, confirmed, conf_session, stats, trades,
                            frames, uni_label, ss_label, skipped + ss_skipped, stale, movers)

    def _quality_buys(self, scan: pd.DataFrame, histories: dict[str, pd.DataFrame],
                      bench: pd.DataFrame | None) -> pd.DataFrame:
        """BUY = every quality check passes (replaces the plain 125-day rule); SELL rows are kept."""
        self._quality = evaluate_latest(histories, self.cfg.buy_quality, bench)
        if scan.empty:
            return scan
        scan = scan.copy()
        scan.loc[scan["Signal"] == "BUY", "Signal"] = ""
        if not self._quality.empty:
            ok = set(self._quality.loc[self._quality["passes"], "Ticker"])
            scan.loc[scan["Ticker"].isin(ok), "Signal"] = "BUY"
        return scan

    def _add_rs(self, signals: pd.DataFrame, frames: dict[str, pd.DataFrame], bench: pd.DataFrame | None,
                now: datetime, intraday: bool) -> pd.DataFrame:
        """Relative strength vs the Nifty: over 3 months, and today (intraday only)."""
        if signals.empty:
            return signals
        out = signals.copy()
        q = self._quality.set_index("Ticker") if self._quality is not None and not self._quality.empty else None
        out["RS vs Nifty 3M %"] = [q.at[t, "RS vs Nifty 3M %"] if q is not None and t in q.index else np.nan
                                   for t in out["Ticker"]]
        if intraday and bench is not None and len(bench) > 1 and bench.index[-1] == pd.Timestamp(now.date()):
            n_today = bench["Close"].iloc[-1] / bench["Close"].iloc[-2] - 1
            vals = []
            for t in out["Ticker"]:
                df = frames.get(t)
                ok = df is not None and len(df) > 1 and df.index[-1] == pd.Timestamp(now.date())
                vals.append(round(100 * (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1 - n_today), 2) if ok else np.nan)
            out["RS today %"] = vals
        return out

    def _apply_levels(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Use the quality rule's structural stop/target (and size from them) for BUY rows."""
        if signals.empty or self._quality is None or self._quality.empty:
            return signals
        q = self._quality.set_index("Ticker")
        risk = RiskManager(self.cfg.risk)
        out = signals.copy()
        for i in out.index[out["Signal"] == "BUY"]:
            tk = out.at[i, "Ticker"]
            if tk not in q.index:
                continue
            r = q.loc[tk]
            out.at[i, "Stop Loss"], out.at[i, "Target Price"] = r["Stop"], r["Target"]
            try:
                plan = risk.plan_from_levels(tk, r["Close"], r["Stop"], r["Target"], r["ATR"])
                out.at[i, "Max Position Units"], out.at[i, "Risk (INR)"] = plan.units, plan.risk_amount
            except RiskError:
                out.at[i, "Max Position Units"], out.at[i, "Risk (INR)"] = 0, float("nan")
            out.at[i, "Note"] = (f"R:R {r['R:R']:.1f}. Buy at the next open only if it is between "
                                 f"₹{r['Breakout level']:,.2f} and "
                                 f"₹{r['Close'] * (1 + self.cfg.buy_quality.entry_max_gap_pct / 100):,.2f}.")
        return out

    def _backtest(self, histories: dict[str, pd.DataFrame], session: date,
                  bench: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        b, r = self.cfg.breakout, self.cfg.risk
        key = hashlib.sha1(repr((session, b, self.cfg.buy_quality, r.atr_stop_multiplier, r.reward_risk_ratio,
                                 sorted(histories))).encode()).hexdigest()
        if key not in self._bt:
            logger.info("Backtesting breakout rules on %d symbols for session %s", len(histories), session)
            chartink = backtest_breakouts(histories, b, r.atr_stop_multiplier, r.reward_risk_ratio)
            quality, _ = backtest_quality(histories, self.cfg.buy_quality, bench, b.est_round_trip_cost_pct)
            sells = chartink[chartink["side"] == "short"] if not chartink.empty else chartink
            trades = pd.concat([quality, sells], ignore_index=True) if not quality.empty else sells
            stats = summarize_rules(trades)
            if not stats.empty:
                stats["Rule"] = stats["Rule"].replace({"BUY (125D breakout)": "BUY (quality breakout)"})
            self._bt = {key: (trades, stats)}
        return self._bt[key]
