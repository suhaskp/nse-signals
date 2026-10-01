"""Automatic NSE pre-open scan (no manual runs).

Every trading day the scan runs by itself from 09:08 IST (when price discovery fixes the IEP) and repeats
every ``rerun_seconds`` until the 09:15 open, so the final pre-open prices are used. If the dashboard only
starts after 09:15, it runs once for the day and says the market was already open. The latest result is
saved to disk, so restarts and page reloads show it immediately.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, time as dtime
from pathlib import Path

import joblib

from config.config import AppConfig
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from pipeline import DailyPipeline, PipelineResult, session_status

logger = logging.getLogger(__name__)
DISCOVERY_END, OPEN = dtime(9, 8), dtime(9, 15)


class PreOpenEngine:
    def __init__(self, cfg: AppConfig, fetcher: MarketDataFetcher | None = None, rerun_seconds: int = 120) -> None:
        self.cfg = cfg
        self.fetcher = fetcher or MarketDataFetcher(cfg.data)
        self.rerun_seconds = rerun_seconds
        self.path = Path(cfg.output_dir) / "preopen_latest.joblib"
        self._lock = threading.Lock()
        self.status = "Waiting for the next pre-open session (09:08 IST on trading days)"
        self.status_since = datetime.now(IST)
        self.last_error: str | None = None
        self._last: PipelineResult | None = self._load()

    def _load(self) -> PipelineResult | None:
        try:
            return joblib.load(self.path) if self.path.exists() else None
        except Exception:  # noqa: BLE001
            return None

    def set_status(self, msg: str) -> None:
        if msg != self.status:
            self.status, self.status_since = msg, datetime.now(IST)

    def peek(self) -> PipelineResult | None:
        return self._last

    def due(self, now: datetime) -> bool:
        """Whether a scan should run now."""
        trading, _ = session_status(self.cfg, now)
        if not trading or now.time() < DISCOVERY_END:
            return False
        last = self._last
        if last is None or last.run_time.date() != now.date():
            return True  # first run today (on time, or late if the dashboard started after 09:15)
        if now.time() >= OPEN:
            return False  # today's result is final once the market opens
        return (now - last.run_time).total_seconds() >= self.rerun_seconds

    def refresh(self, now: datetime | None = None) -> PipelineResult | None:
        now = (now or datetime.now(IST)).astimezone(IST)
        trading, note = session_status(self.cfg, now)
        if not trading:
            self.set_status(f"{note} The next scan runs automatically at 09:08 IST on the next trading day.")
            return self._last
        if now.time() < DISCOVERY_END:
            self.set_status("Waiting for pre-open price discovery; the scan starts automatically at 09:08 IST.")
            return self._last
        if not self.due(now):
            return self._last
        with self._lock:
            self.set_status("Running the pre-open scan")
            try:
                result = DailyPipeline(self.cfg, self.fetcher, now=now).run(
                    retrain=False, record_paper_orders=self.cfg.execution.record_paper_orders)
            except Exception as exc:
                self.last_error = str(exc)
                self.set_status(f"Pre-open scan failed ({exc}); retrying automatically")
                raise
            self._last, self.last_error = result, None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, self.path)
            final = now.time() >= OPEN
            self.set_status("Final for today (the market is open)" if final else
                            f"Updated {now:%H:%M}; re-checks every {self.rerun_seconds // 60} min until 09:15")
            return result
