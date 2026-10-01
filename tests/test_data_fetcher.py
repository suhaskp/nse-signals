import numpy as np
import pandas as pd
import pytest

from config.config import DataConfig
from data_ingestion.data_fetcher import (DataFetchError, MarketDataFetcher, NonRetryableDataError,
                                         RateLimiter, call_with_backoff, completed_sessions,
                                         sanitize_ohlcv, yahoo_symbol)
from data_ingestion.preopen import NSEPreOpenSource, PreOpenHistory, PreOpenQuote, _num
from data_ingestion.synthetic import SyntheticMarket


def test_yahoo_symbol():
    assert yahoo_symbol("RELIANCE") == "RELIANCE.NS" and yahoo_symbol("M&M") == "M&M.NS"


def test_sanitize_handles_nulls_bad_prices_and_ist_index(ohlcv):
    raw = ohlcv.rename(columns=str.lower).copy()
    raw.index = raw.index.tz_localize("Asia/Kolkata")
    raw.iloc[5, raw.columns.get_loc("close")] = np.nan
    raw.iloc[6, raw.columns.get_loc("open")] = -1
    raw.iloc[7, raw.columns.get_loc("volume")] = np.nan
    raw = pd.concat([raw, raw.iloc[[10]]])
    clean = sanitize_ohlcv(raw)
    assert clean.index.tz is None and clean.index.is_unique and clean.index.is_monotonic_increasing
    assert not clean.isna().any().any()
    assert (clean["High"] >= clean[["Open", "Close"]].max(axis=1)).all()
    assert len(clean) == len(ohlcv) - 2


def test_sanitize_missing_columns():
    with pytest.raises(NonRetryableDataError):
        sanitize_ohlcv(pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-01-02"])))


def test_backoff_retries_then_succeeds():
    calls, sleeps = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert call_with_backoff(flaky, max_retries=5, base_delay=1, max_delay=10, sleeper=sleeps.append) == "ok"
    assert calls["n"] == 3 and len(sleeps) == 2


def test_backoff_gives_up_and_skips_nonretryable():
    with pytest.raises(DataFetchError):
        call_with_backoff(lambda: 1 / 0, max_retries=2, base_delay=0, max_delay=0, sleeper=lambda s: None)
    calls = {"n": 0}

    def fatal():
        calls["n"] += 1
        raise NonRetryableDataError("bad symbol")

    with pytest.raises(NonRetryableDataError):
        call_with_backoff(fatal, max_retries=5, base_delay=0, max_delay=0, sleeper=lambda s: None)
    assert calls["n"] == 1


def test_rate_limiter_spaces_calls():
    t, waits = {"now": 0.0}, []
    limiter = RateLimiter(60, 60.0, clock=lambda: t["now"], sleeper=waits.append)
    limiter.acquire(); limiter.acquire()
    assert waits == [pytest.approx(1.0)]


def test_completed_sessions_drops_today(ohlcv):
    last = ohlcv.index[-1].date()
    assert completed_sessions(ohlcv, last).index.max() < pd.Timestamp(last)


SAMPLE_NSE = {"data": [
    {"metadata": {"symbol": "RELIANCE", "iep": "1,352.40", "previousClose": 1336.8, "finalQuantity": "184,221"}},
    {"metadata": {"symbol": "TCS", "iep": 0, "lastPrice": 3050.1, "previousClose": "3,041.00", "finalQuantity": 9000}},
    {"metadata": {"symbol": "INFY", "iep": "-", "previousClose": 1500}},  # unusable: no price
    {"metadata": {"symbol": "OTHER", "iep": 10, "previousClose": 9}},     # not requested
]}


def test_nse_parser(now):
    quotes = NSEPreOpenSource.parse(SAMPLE_NSE, ["RELIANCE", "TCS", "INFY"], now)
    assert set(quotes) == {"RELIANCE", "TCS"}
    assert quotes["RELIANCE"].iep == pytest.approx(1352.40) and quotes["RELIANCE"].preopen_quantity == 184221
    assert quotes["TCS"].iep == pytest.approx(3050.1) and quotes["TCS"].prev_close == 3041.0
    with pytest.raises(DataFetchError):
        NSEPreOpenSource.parse({"unexpected": 1}, ["TCS"], now)
    assert np.isnan(_num("-")) and _num("1,23,456.5") == 123456.5


def test_nse_source_reprimes_cookies_on_403(now):
    class Resp:
        def __init__(self, code, payload=None):
            self.status_code, self._p = code, payload
        def raise_for_status(self): pass
        def json(self): return self._p

    class Session:
        def __init__(self): self.api_calls, self.page_calls = 0, 0
        def get(self, url, **kw):
            if "api" not in url:
                self.page_calls += 1
                return Resp(200)
            self.api_calls += 1
            return Resp(403) if self.api_calls == 1 else Resp(200, SAMPLE_NSE)

    session = Session()
    src = NSEPreOpenSource(session=session)
    with pytest.raises(DataFetchError):
        src.fetch(["RELIANCE"])
    assert "RELIANCE" in src.fetch(["RELIANCE"]) and session.page_calls == 2


def test_preopen_history_average(tmp_path, now):
    store = PreOpenHistory(tmp_path / "po.csv")
    for i, d in enumerate(pd.bdate_range(end="2026-09-25", periods=12)):
        store.record({"X": PreOpenQuote("X", 100, 100, 1000 + i, now, "t")}, d.date())
    store.record({"X": PreOpenQuote("X", 100, 100, 1000, now, "t")}, pd.Timestamp("2026-09-25").date())  # replace
    avg = store.average_quantity("X", now.date(), lookback=5, min_days=5)
    assert avg == pytest.approx(np.mean([1007, 1008, 1009, 1010, 1000]))
    assert np.isnan(store.average_quantity("X", now.date(), lookback=20, min_days=20))
    assert np.isnan(store.average_quantity("NONE", now.date(), 5, 1))


def test_fetcher_with_synthetic_market(now):
    market = SyntheticMarket(now=now)
    f = MarketDataFetcher(DataConfig(provider="synthetic", preopen_source="synthetic"),
                          provider=market, preopen=market, sleeper=lambda s: None)
    daily = f.fetch_history("RELIANCE", now=now)
    quotes = f.fetch_preopen(["RELIANCE", "TCS"])
    assert daily.index.max() < pd.Timestamp(now.date())
    assert quotes["RELIANCE"].prev_close == pytest.approx(daily["Close"].iloc[-1])


def test_missing_symbol_fails_fast_and_is_remembered(tmp_path, now):
    from data_ingestion.cache import DailyCache

    class Provider:
        name = "fake"
        calls: list = []

        def get_daily(self, symbol, start, end):
            Provider.calls.append(symbol)
            return pd.DataFrame() if symbol == "GONE" else SyntheticMarket(now=now).get_daily(symbol, start, end)

    fetcher = MarketDataFetcher(DataConfig(max_retries=4), provider=Provider(),
                                preopen=SyntheticMarket(now=now), sleeper=lambda s: None)
    cache = DailyCache(tmp_path)
    out = cache.sync(fetcher, ["TCS", "GONE"], now, 400)
    assert "TCS" in out and "GONE" not in out
    assert Provider.calls.count("GONE") == 1          # no backoff retries for a symbol that doesn't exist
    assert "GONE" in cache.unavailable()
    Provider.calls.clear()
    cache.sync(fetcher, ["TCS", "GONE"], now, 400)
    assert "GONE" not in Provider.calls               # not asked again on later refreshes


def test_nse_placeholder_symbols_are_ignored(tmp_path):
    from data_ingestion.universe import load_symbols
    p = tmp_path / "n500.csv"
    pd.DataFrame({"Company Name": ["Reliance", "Dummy"], "Symbol": ["RELIANCE", "DUMMYHEG"]}).to_csv(p, index=False)
    assert load_symbols(p) == (["RELIANCE"], [])


def test_rate_limited_batch_is_retried_then_split(now):
    from config.config import DataConfig as DC

    class Throttled:
        name = "fake"
        calls = []

        def __init__(self, fail_first):
            self.fail_first = fail_first

        def get_daily_batch(self, symbols, start, end):
            Throttled.calls.append(len(symbols))
            if self.fail_first > 0 or len(symbols) > 20 and self.fail_first < 0:
                self.fail_first -= 1 if self.fail_first > 0 else 0
                return {}                                  # yfinance-style silent failure
            m = SyntheticMarket(now=now)
            return {s: m.get_daily(s, start, end) for s in symbols}

        def get_daily(self, symbol, start, end):
            return SyntheticMarket(now=now).get_daily(symbol, start, end)

    syms = [f"T{i:02d}" for i in range(40)]
    f = MarketDataFetcher(DC(max_retries=2), provider=Throttled(fail_first=1), preopen=SyntheticMarket(now=now),
                          sleeper=lambda s: None)
    assert len(f.fetch_many(syms, 400, now)) == 40      # recovered by retrying the batch
    Throttled.calls.clear()
    f2 = MarketDataFetcher(DC(max_retries=1), provider=Throttled(fail_first=-1), preopen=SyntheticMarket(now=now),
                           sleeper=lambda s: None)
    assert len(f2.fetch_many(syms, 400, now)) == 40     # big batches keep failing -> small pieces succeed
    assert 20 in Throttled.calls


def test_failed_update_is_flagged_not_silent(tmp_path, now):
    from data_ingestion.cache import DailyCache
    from intelligence_engine import freshness_warning
    m = SyntheticMarket(now=now)
    good = MarketDataFetcher(DataConfig(provider="synthetic", preopen_source="synthetic"), provider=m, preopen=m,
                             sleeper=lambda s: None)
    cache = DailyCache(tmp_path)
    syms = [f"T{i:02d}" for i in range(30)]
    cache.sync(good, syms, now, 400)

    class Down:
        name = "down"
        def get_daily_batch(self, symbols, start, end): return {}
        def get_daily(self, symbol, start, end): return pd.DataFrame()

    bad = MarketDataFetcher(DataConfig(max_retries=0), provider=Down(), preopen=m, sleeper=lambda s: None)
    frames = cache.sync(bad, syms, now, 400)
    assert len(frames) == 30 and len(cache.last_failed) == 30   # still served from cache...
    w = freshness_warning(cache.last_failed, 30, now.date(), now, "open")
    assert w and "failed for 30 of 30" in w                     # ...but loudly flagged
    assert freshness_warning([], 30, now.date(), now, "open") is None
