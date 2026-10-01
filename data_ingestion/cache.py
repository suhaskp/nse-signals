"""On-disk cache of daily history so the dashboard opens fast.

First sync downloads full history (~5 years) for every symbol. Later syncs
download only the last few weeks and splice them onto the cached series.
A full re-download happens weekly per symbol, so dividend and split
adjustments from Yahoo flow through.

Only completed sessions (before today) are written to disk; today's bar,
which may still be forming, is returned to the caller but never cached.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timedelta
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from data_ingestion.data_fetcher import IST, MarketDataFetcher

logger = logging.getLogger(__name__)


class DailyCache:
    """Pickle-per-symbol cache with incremental updates (syncs are serialized process-wide)."""

    _sync_lock = threading.Lock()

    def __init__(self, root: Path, full_refresh_days: int = 7, recent_days: int = 20,
                 unavailable_retry_days: int = 7) -> None:
        self.root = Path(root) / "daily"
        self.meta_path = self.root / "_meta.json"
        self.unavailable_path = self.root / "_unavailable.json"
        self.unavailable_retry_days = unavailable_retry_days
        self.last_failed: list[str] = []  # cached symbols whose latest update failed in the last sync
        self.full_refresh_days, self.recent_days = full_refresh_days, recent_days

    def _path(self, symbol: str) -> Path:
        return self.root / f"{symbol.replace('/', '_')}.pkl"

    def _meta(self) -> dict[str, str]:
        try:
            return json.loads(self.meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def unavailable(self) -> dict[str, str]:
        """Symbols the data source had no history for, with the date they were last tried."""
        try:
            return json.loads(self.unavailable_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        try:
            return pd.read_pickle(p) if p.exists() else None
        except Exception:  # noqa: BLE001 - a corrupt file is simply re-downloaded
            logger.warning("%s: cache file unreadable; will re-download", symbol)
            return None

    def sync(self, fetcher: MarketDataFetcher, symbols: list[str], now: datetime, lookback_days: int,
             status: Callable[[str], None] | None = None) -> dict[str, pd.DataFrame]:
        """Bring the cache up to date and return full frames (including today's bar if any)."""
        if status and DailyCache._sync_lock.locked():
            status("Waiting for another refresh to finish downloading prices")
        with DailyCache._sync_lock:
            return self._sync(fetcher, symbols, now, lookback_days, status or (lambda _msg: None))

    def _sync(self, fetcher: MarketDataFetcher, symbols: list[str], now: datetime,
              lookback_days: int, status: Callable[[str], None]) -> dict[str, pd.DataFrame]:
        now = now.astimezone(IST)
        today = now.date()
        meta = self._meta()
        bad = self.unavailable()
        retry_before = (today - timedelta(days=self.unavailable_retry_days)).isoformat()
        skip = {s for s, d in bad.items() if d >= retry_before}
        symbols = [s for s in symbols if s not in skip]
        stale_before = (today - timedelta(days=self.full_refresh_days)).isoformat()
        cached = {s: df for s in symbols if (df := self.load(s)) is not None and s in meta
                  and meta[s] >= stale_before}
        need_full = [s for s in symbols if s not in cached]
        out: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        if need_full:
            logger.info("Downloading full history for %d symbols (first run or weekly refresh)", len(need_full))
            first = not cached
            label = ("First run: downloading ~5 years of history" if first else "Weekly full refresh")
            status(f"{label} for {len(need_full)} stocks")
            prog = lambda i, n: status(f"{label}: {i}/{n} stocks done")  # noqa: E731
            for s, df in fetcher.fetch_many(need_full, lookback_days, now, progress=prog).items():
                out[s] = df
                meta[s] = today.isoformat()
        if cached:
            status(f"Updating recent prices for {len(cached)} stocks")
            recent = fetcher.fetch_many(list(cached), self.recent_days, now)
            for s, old in cached.items():
                new = recent.get(s)
                if new is None or new.empty:
                    out[s] = old  # keep serving the cache if the refresh failed
                    failed.append(s)
                    continue
                out[s] = pd.concat([old[old.index < new.index.min()], new]).sort_index()

        self.root.mkdir(parents=True, exist_ok=True)
        for s, df in out.items():
            done = df[df.index < pd.Timestamp(today)]
            if not done.empty:
                done.to_pickle(self._path(s))
        self.meta_path.write_text(json.dumps(meta))
        self.last_failed = failed
        if failed:
            logger.warning("Price update failed for %d of %d cached symbols; serving their cached history until the "
                           "next refresh succeeds", len(failed), len(cached))
        missing = [s for s in need_full if s not in out]
        if missing:
            logger.warning("No price data available for %d symbol(s), skipped for %d days: %s",
                           len(missing), self.unavailable_retry_days, ", ".join(missing))
        bad = {s: d for s, d in bad.items() if s not in out}
        bad.update({s: today.isoformat() for s in missing})
        self.unavailable_path.write_text(json.dumps(bad))
        return out

    def last_session(self) -> date | None:
        """Most recent completed session in the cache (for display)."""
        dates = [df.index.max() for p in self.root.glob("*.pkl") if (df := pd.read_pickle(p)) is not None]
        return max(dates).date() if dates else None
