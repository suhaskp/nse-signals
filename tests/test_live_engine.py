import os
import time
from datetime import datetime

import pandas as pd
import pytest

from config.config import load_config
from data_ingestion.cache import DailyCache
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from data_ingestion.universe import latest_csv, refresh_nifty500
from live_engine import LiveSignalEngine, market_phase

TICKERS = ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "LT", "KOTAKBANK", "BAJFINANCE", "NTPC"]


def _engine(tmp_path, now, **extra):
    cfg = load_config(overrides={"tickers": TICKERS, "data": {"provider": "synthetic", "preopen_source": "synthetic"},
                                 "cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out"),
                                 "superstar_dir": str(tmp_path / "ss"), **extra})
    m = SyntheticMarket(now=now)
    return LiveSignalEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                            auto_universe=False), cfg


@pytest.mark.parametrize("hhmm, phase", [("08:30", "pre_market"), ("11:00", "open"),
                                         ("15:45", "settling"), ("16:30", "after_close")])
def test_market_phase(hhmm, phase):
    h, m = map(int, hhmm.split(":"))
    assert market_phase(load_config(), datetime(2026, 9, 28, h, m, tzinfo=IST))[0] == phase
    assert market_phase(load_config(), datetime(2026, 9, 27, h, m, tzinfo=IST))[0] == "closed"  # Sunday


@pytest.mark.parametrize("hhmm, mode, same_day", [("08:30", "confirmed", False), ("11:00", "live", True),
                                                  ("16:30", "confirmed", True)])
def test_engine_modes(tmp_path, hhmm, mode, same_day):
    h, m = map(int, hhmm.split(":"))
    now = datetime(2026, 9, 28, h, m, tzinfo=IST)
    snap = _engine(tmp_path, now)[0].refresh(now)
    assert snap.mode == mode and (snap.session == now.date()) is same_day
    assert set(snap.signals["Signal"]) <= {"BUY", "SELL"}
    assert (tmp_path / "out" / f"eod_signals_{snap.confirmed_session}.csv").exists()


def test_refresh_is_throttled_and_cache_excludes_today(tmp_path):
    now = datetime(2026, 9, 28, 11, 0, tzinfo=IST)
    engine, _ = _engine(tmp_path, now)
    a = engine.refresh(now)
    b = engine.refresh(now.replace(minute=2))
    assert a is b  # within live_refresh_seconds -> cached snapshot
    c = engine.refresh(now.replace(minute=6))
    assert c is not a
    cached = pd.read_pickle(tmp_path / "cache" / "daily" / "TCS.pkl")
    assert cached.index.max() < pd.Timestamp(now.date())


def test_superstar_dropin_folder_picks_newest(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    engine, cfg = _engine(tmp_path, now)
    ss = tmp_path / "ss"
    ss.mkdir()
    pd.DataFrame({"Stock": ["Old"], "NSE Code": ["ITC"]}).to_csv(ss / "old.csv", index=False)
    time.sleep(0.05)
    pd.DataFrame({"Stock": ["Arvind", "X"], "NSE Code": ["ARVINDFASN", None]}).to_csv(ss / "new.csv", index=False)
    assert latest_csv(ss).name == "new.csv"
    snap = engine.refresh(now)
    assert "new.csv" in snap.superstar_label and "ARVINDFASN" in snap.histories and snap.skipped == ["X"]


def test_nifty500_download_and_fallback(tmp_path):
    class R:
        def __init__(self, code, body): self.status_code, self.content = code, body
    dest = tmp_path / "n500.csv"
    assert not refresh_nifty500(dest, getter=lambda url: R(403, b""))  # no copy, download blocked
    body = b"Company Name,Industry,Symbol,Series,ISIN Code\nReliance,Energy,RELIANCE,EQ,INE002A01018\n"
    assert refresh_nifty500(dest, getter=lambda url: R(200, body)) and dest.exists()
    os.utime(dest, (time.time() - 30 * 86400,) * 2)  # stale copy + failing network -> keep old copy
    assert refresh_nifty500(dest, getter=lambda url: (_ for _ in ()).throw(ConnectionError("offline")))


def test_cache_incremental_sync(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    m = SyntheticMarket(now=now)
    f = MarketDataFetcher(load_config().data, provider=m, preopen=m, sleeper=lambda s: None)
    calls = []
    orig = f.fetch_many
    f.fetch_many = lambda syms, days=None, now=None, **k: (calls.append((len(syms), days)), orig(syms, days, now))[1]
    cache = DailyCache(tmp_path)
    cache.sync(f, ["TCS", "ITC"], now, 1900)
    cache.sync(f, ["TCS", "ITC"], now, 1900)
    assert calls == [(2, 1900), (2, 20)]  # full first, then only recent days
