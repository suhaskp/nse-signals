"""Process-wide engine singletons (no Streamlit dependency) and the background refresher.

The web pages and the background thread import this module, so they share the same
engines and caches. ``serve.py`` starts the refresher before the web server, so data
stays fresh from boot even if no browser is ever opened.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from datetime import datetime

from background import BackgroundRefresher
from config import load_config, setup_logging
from config.config import DEFAULT_TICKERS
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket, demo_now
from intelligence_engine import IntelligenceEngine, morning_check
from data_ingestion.preopen import PreOpenHistory
from live_engine import LiveSignalEngine
from preopen_engine import PreOpenEngine

DEMO = os.getenv("PIPELINE_DEMO", "0") == "1"
DEMO_TIME = "09:10"
_lock = threading.Lock()
_engines: dict[bool, tuple] = {}
_refresher: BackgroundRefresher | None = None
_preopen_refresher: BackgroundRefresher | None = None
_crypto_refresher: BackgroundRefresher | None = None
_watchdog: BackgroundRefresher | None = None
_crypto: dict[bool, object] = {}


def now_ist() -> datetime:
    return demo_now(DEMO_TIME) if DEMO else datetime.now(IST)


def get_engines(demo: bool = DEMO) -> tuple[IntelligenceEngine, LiveSignalEngine, object]:
    return _get_all(demo)[:3]


def get_preopen(demo: bool = DEMO) -> PreOpenEngine:
    return _get_all(demo)[3]


def _get_all(demo: bool) -> tuple:
    with _lock:
        if demo not in _engines:
            if demo:
                cfg = load_config(overrides={
                    "tickers": list(DEFAULT_TICKERS) + [f"DEMO{i:02d}" for i in range(40)],
                    "data": {"provider": "synthetic", "preopen_source": "synthetic",
                             "preopen_history_path": "data/demo/preopen_history.csv"},
                    "cache_dir": "data/demo/cache", "output_dir": "outputs/demo",
                    "superstar_dir": "data/demo/superstar",
                    "model": {"model_dir": "artifacts/demo"}})
                market = SyntheticMarket(now=demo_now(DEMO_TIME))
                fetcher = MarketDataFetcher(cfg.data, provider=market, preopen=market)
                market.seed_history(PreOpenHistory(cfg.data.preopen_history_path), list(cfg.tickers))
                live = LiveSignalEngine(cfg, fetcher, auto_universe=False)
                pre = PreOpenEngine(cfg, fetcher)
            else:
                from storage import migrate_legacy
                cfg = load_config()
                migrate_legacy(Path(__file__).resolve().parent, Path(cfg.data_home))
                live = LiveSignalEngine(cfg)
                pre = PreOpenEngine(cfg)
            setup_logging(cfg.log_dir, cfg.log_level)
            _engines[demo] = (IntelligenceEngine(cfg, live=live), live, cfg, pre)
        return _engines[demo]


def background_running() -> bool:
    """True when the background refresher is doing the work (pages then never wait)."""
    return _refresher is not None


def get_crypto(demo: bool = DEMO):
    """The crypto engine (Binance public data; offline synthetic coins in demo mode)."""
    from crypto_engine import CryptoEngine
    cfg = _get_all(demo)[2]
    with _lock:
        if demo not in _crypto:
            if demo:
                from data_ingestion.synthetic import SyntheticCrypto
                _crypto[demo] = CryptoEngine(cfg, SyntheticCrypto(now=demo_now(DEMO_TIME)))
            else:
                _crypto[demo] = CryptoEngine(cfg)
        return _crypto[demo]


def start_background(demo: bool = DEMO) -> BackgroundRefresher | None:
    """Start the refresher once per process (no-op when PIPELINE_BACKGROUND=0)."""
    global _refresher
    if os.getenv("PIPELINE_BACKGROUND", "1") == "0":
        return None
    with _lock:
        if _refresher is not None:
            return _refresher
    intel, live, cfg = get_engines(demo)
    pre = get_preopen(demo)
    crypto = get_crypto(demo) if cfg.crypto.enabled else None  # resolved BEFORE taking the lock (no re-entry)

    def backup_job(now: datetime) -> None:
        from storage import backup_records
        backup_records(Path(cfg.data_home), today=now.date())

    def intelligence_job(now: datetime) -> None:
        morning_check(intel, intel.refresh(now), now)

    with _lock:
        if _refresher is None:
            _refresher = BackgroundRefresher([("breakouts", live.refresh), ("intelligence", intelligence_job),
                                              ("backup", backup_job)], now_ist).start()
            if crypto is not None:
                global _crypto_refresher
                # crypto trades 24/7 and has its own heavy daily run: its own thread so NSE jobs are never delayed
                _crypto_refresher = BackgroundRefresher([("crypto", crypto.refresh)], now_ist, interval_seconds=60).start()
            global _watchdog

            def watchdog_job(now: datetime) -> None:
                from alerts import AlertCenter
                from pipeline import session_status
                from status import heartbeat
                beats = heartbeat(intel.peek(), live.peek(), crypto.peek() if crypto else None, now,
                                  session_status(cfg, now)[0])
                for b in beats:
                    if b["overdue"]:
                        AlertCenter(Path(cfg.output_dir) / "alerts.json").add(
                            f"heartbeat:{b['engine']}:{now:%Y-%m-%d-%H}", "warn",
                            f"{b['engine']} has not refreshed since {b['last']:%H:%M}. Check the dashboard window and "
                            "your internet connection; it retries automatically.", now)
            # its own thread, so a stalled job elsewhere cannot silence the warning
            _watchdog = BackgroundRefresher([("watchdog", watchdog_job)], now_ist, interval_seconds=300).start()
            if cfg.preopen_scan_enabled:
                global _preopen_refresher
                # the pre-open window is only 7 minutes: its own thread, so long research runs never delay it
                _preopen_refresher = BackgroundRefresher([("preopen", pre.refresh)], now_ist, interval_seconds=30).start()
        return _refresher
