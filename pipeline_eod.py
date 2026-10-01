"""End-of-day breakout pipeline: tomorrow's BUY / SELL candidates.

Run after the NSE close (16:00 IST onward, once Yahoo has today's bar):

    fetch daily history -> 125-day breakout (BUY) / breakdown (SELL) scan ->
    backtest the same rules on the same history -> size each signal ->
    tag Superstar-investor holdings -> save outputs/eod_signals_YYYY-MM-DD.csv

The backtest is what tells you whether the list is worth acting on: a rule
with negative average R after costs loses money on average, however good
today's chart looks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import pandas as pd

from config.config import AppConfig
from data_ingestion.data_fetcher import IST, DataFetchError, MarketDataFetcher, completed_sessions
from data_ingestion.universe import load_symbols
from monitoring.breakout_backtest import backtest_breakouts, summarize
from risk_engine.risk_manager import RiskError, RiskManager, Side
from screeners.breakout import scan_breakouts

logger = logging.getLogger(__name__)

SIGNAL_COLUMNS = ["Ticker", "Signal", "Session", "Close", "Beyond Range %", "Volume x Avg", "RSI", "ATR",
                  "Stop Loss", "Target Price", "Max Position Units", "Risk (INR)", "Superstar Buy",
                  "Rule Hist. Trades", "Rule Hist. Win %", "Rule Hist. Avg R", "Ticker Hist. Trades",
                  "Ticker Hist. Avg R", "Note"]

SELL_NOTE = ("Exit or avoid if held. Shorting: cash-segment shorts must be squared off the same day; "
             "overnight shorts need F&O.")
BUY_NOTE = "Entry next session near the open; re-check the pre-open IEP (gap-ups change the risk)."


@dataclass
class EodResult:
    session: date
    signals: pd.DataFrame
    scan: pd.DataFrame
    rule_stats: pd.DataFrame
    trades: pd.DataFrame
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def resolve_universe(cfg: AppConfig) -> tuple[list[str], set[str], list[str]]:
    """Symbols to scan, the Superstar-buy set, and names skipped for lacking an NSE code."""
    skipped: list[str] = []
    symbols = list(cfg.tickers)
    if cfg.universe_file:
        symbols, sk = load_symbols(Path(cfg.universe_file))
        skipped += sk
    superstar: set[str] = set()
    if cfg.superstar_file:
        ss, sk = load_symbols(Path(cfg.superstar_file))
        superstar = set(ss)
        skipped += sk
    return symbols, superstar, skipped


class EodBreakoutPipeline:
    """Builds the next-session BUY/SELL list from the Chartink 125-day rules."""

    def __init__(self, cfg: AppConfig, fetcher: MarketDataFetcher | None = None,
                 now: datetime | None = None) -> None:
        self.cfg = cfg
        self.now = (now or datetime.now(IST)).astimezone(IST)
        self.fetcher = fetcher or MarketDataFetcher(cfg.data)
        self.risk = RiskManager(cfg.risk)

    def _cutoff(self) -> date:
        """Sessions strictly before this date count as complete."""
        h, m = (int(x) for x in self.cfg.breakout.eod_ready_time.split(":"))
        today = self.now.date()
        return today + timedelta(days=1) if self.now.time() >= dtime(h, m) else today

    def run(self) -> EodResult:
        symbols, superstar, skipped = resolve_universe(self.cfg)
        scan_symbols = list(dict.fromkeys(symbols + sorted(superstar)))
        logger.info("=== EOD breakout scan %s IST | %d symbols (%d Superstar buys) ===",
                    self.now.isoformat(timespec="minutes"), len(scan_symbols), len(superstar))
        raw = self.fetcher.fetch_many(scan_symbols, now=self.now)
        cutoff = self._cutoff()
        histories = {s: h for s, df in raw.items() if not (h := completed_sessions(df, cutoff)).empty}
        if not histories:
            raise DataFetchError("No usable history for any symbol")
        session = max(df.index[-1] for df in histories.values()).date()
        stale = [s for s, df in histories.items() if df.index[-1].date() < session]
        if stale:
            logger.warning("%d symbols have no bar for %s yet (data lag or suspended): %s",
                           len(stale), session, ", ".join(stale))

        bcfg = self.cfg.breakout
        scan = scan_breakouts(histories, bcfg)
        trades = backtest_breakouts(histories, bcfg, self.cfg.risk.atr_stop_multiplier,
                                    self.cfg.risk.reward_risk_ratio)
        rule_stats = summarize(trades)
        signals = self._build_signals(scan, trades, rule_stats, superstar, session, stale)

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        signals.to_csv(out_dir / f"eod_signals_{session.isoformat()}.csv", index=False)
        scan.to_csv(out_dir / f"eod_scan_{session.isoformat()}.csv", index=False)
        if not trades.empty:
            trades.to_csv(out_dir / "breakout_backtest_trades.csv", index=False)
        logger.info("EOD: %d BUY, %d SELL for the session after %s",
                    int((signals["Signal"] == "BUY").sum()), int((signals["Signal"] == "SELL").sum()), session)
        return EodResult(session, signals, scan, rule_stats, trades, histories, skipped)

    def _build_signals(self, scan: pd.DataFrame, trades: pd.DataFrame, rule_stats: pd.DataFrame,
                       superstar: set[str], session: date, stale: list[str]) -> pd.DataFrame:
        if scan.empty:
            return pd.DataFrame(columns=SIGNAL_COLUMNS)
        live = scan[(scan["Signal"] != "") & (scan["Session"] == session.isoformat())
                    & ~scan["Ticker"].isin(stale)]
        by_rule = {("long" if r.startswith("BUY") else "short"): row
                   for r, row in rule_stats.set_index("Rule").iterrows()} if not rule_stats.empty else {}
        rows = []
        for _, r in live.iterrows():
            side = Side.LONG if r["Signal"] == "BUY" else Side.SHORT
            key = "long" if side is Side.LONG else "short"
            try:
                plan = self.risk.build_trade_plan(r["Ticker"], r["Close"], r["ATR"], side)
                stop, target, units, risk_amt = plan.stop_loss, plan.take_profit, plan.units, plan.risk_amount
            except RiskError as exc:
                logger.warning("%s: no plan (%s)", r["Ticker"], exc)
                stop = target = risk_amt = float("nan")
                units = 0
            rs = by_rule.get(key)
            tk = trades[(trades["Ticker"] == r["Ticker"]) & (trades["side"] == key)] if not trades.empty else trades
            rows.append({
                **{k: r[k] for k in ("Ticker", "Signal", "Session", "Close", "Beyond Range %",
                                     "Volume x Avg", "RSI", "ATR")},
                "Stop Loss": stop, "Target Price": target, "Max Position Units": units, "Risk (INR)": risk_amt,
                "Superstar Buy": "Yes" if r["Ticker"] in superstar else "",
                "Rule Hist. Trades": int(rs["Trades"]) if rs is not None else 0,
                "Rule Hist. Win %": rs["Win rate %"] if rs is not None else float("nan"),
                "Rule Hist. Avg R": rs["Avg R"] if rs is not None else float("nan"),
                "Ticker Hist. Trades": len(tk),
                "Ticker Hist. Avg R": round(tk["r_multiple"].mean(), 3) if len(tk) else float("nan"),
                "Note": BUY_NOTE if side is Side.LONG else SELL_NOTE,
            })
        return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
