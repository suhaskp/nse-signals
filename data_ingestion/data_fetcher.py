"""Daily NSE market data with retries, rate limiting and sanitization.

The :class:`MarketDataFetcher` is provider-agnostic: providers return raw
daily OHLCV and the fetcher handles throttling, retries, cleaning and
anti-lookahead trimming. Pre-open (09:00-09:15 IST) data lives in
:mod:`data_ingestion.preopen`.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, TypeVar
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.config import DataConfig

if TYPE_CHECKING:
    from data_ingestion.preopen import PreOpenQuote, PreOpenSource

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLS = ["Open", "High", "Low", "Close"]
T = TypeVar("T")


def yahoo_symbol(symbol: str) -> str:
    """NSE symbol -> Yahoo Finance ticker (``RELIANCE`` -> ``RELIANCE.NS``; indices like ``^NSEI`` unchanged)."""
    return symbol if symbol.startswith("^") else f"{symbol}.NS"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class DataFetchError(RuntimeError):
    """A request failed (possibly transiently) or returned no usable data."""


class NonRetryableDataError(DataFetchError):
    """A failure that retrying cannot fix (bad schema, invalid symbol, auth)."""


# --------------------------------------------------------------------------- #
# Rate limiting & retry
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Thread-safe limiter that spaces calls evenly (``max_calls`` per ``period``)."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self._interval = period_seconds / max_calls
        self._clock, self._sleep = clock, sleeper
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a call is permitted. Returns the seconds waited."""
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next_allowed - now)
            if wait:
                self._sleep(wait)
            self._next_allowed = max(now, self._next_allowed) + self._interval
            return wait


def _is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("ratelimit", "too many requests", "429"))


def call_with_backoff(fn: Callable[[], T], *, max_retries: int, base_delay: float,
                      max_delay: float, description: str = "request",
                      sleeper: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn`` with exponential backoff and equal jitter.

    Delay for attempt *n* is ``min(max_delay, base * 2**(n-1))``, half fixed and
    half random. Rate-limit (HTTP 429) errors get double the delay.

    Raises:
        NonRetryableDataError: Immediately, without retrying.
        DataFetchError: After ``max_retries`` failed retries.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except NonRetryableDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - providers raise many types
            attempt += 1
            if attempt > max_retries:
                raise DataFetchError(f"{description} failed after {max_retries} retries: {exc}") from exc
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            if _is_rate_limit(exc):
                delay = min(max_delay, delay * 2)
            delay = delay / 2 + random.uniform(0, delay / 2)
            logger.warning("%s failed (attempt %d/%d): %s - retrying in %.1fs",
                           description, attempt, max_retries, exc, delay)
            sleeper(delay)


# --------------------------------------------------------------------------- #
# Sanitization
# --------------------------------------------------------------------------- #
def sanitize_ohlcv(df: pd.DataFrame, *, ticker: str = "") -> pd.DataFrame:
    """Normalize and clean a raw daily OHLCV frame.

    Steps: flatten/rename columns, coerce numerics, index -> tz-naive IST
    session dates, drop duplicate dates, null out non-positive prices, drop
    bars with no close, fill missing O/H/L from the bar's own close (never
    from other bars, to avoid fabricating history), clamp High/Low, and set
    missing volume to 0.

    Raises:
        NonRetryableDataError: Required columns are missing.
        DataFetchError: Nothing usable remains after cleaning.
    """
    if df is None or df.empty:
        raise DataFetchError(f"{ticker}: empty frame")
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={c: str(c).strip().title() for c in out.columns})
    missing = set(OHLCV) - set(out.columns)
    if missing:
        raise NonRetryableDataError(f"{ticker}: missing columns {sorted(missing)}")
    out = out[OHLCV].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()

    bad_price = (out[PRICE_COLS] <= 0).any(axis=1)
    out.loc[bad_price, PRICE_COLS] = np.nan
    n_before = len(out)
    out = out.dropna(subset=["Close"])
    for col in ("Open", "High", "Low"):
        out[col] = out[col].fillna(out["Close"])
    out["High"] = out[PRICE_COLS].max(axis=1)
    out["Low"] = out[PRICE_COLS].min(axis=1)
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)
    if dropped := n_before - len(out):
        logger.debug("%s: dropped %d bars without a close", ticker, dropped)
    if out.empty:
        raise DataFetchError(f"{ticker}: no usable bars after sanitization")
    return out


def completed_sessions(daily: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return only daily bars strictly before ``as_of``.

    Yahoo can emit a partial bar for today once NSE opens. Using it during
    pre-open would leak information, so it is always removed.
    """
    return daily.loc[daily.index < pd.Timestamp(as_of)]


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class DataProvider(Protocol):
    """Minimal interface every daily-history source implements."""

    name: str

    def get_daily(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...


class YFinanceProvider:
    """Yahoo Finance via ``yfinance`` for NSE symbols (free, unofficial, rate-limited)."""

    name = "yfinance"

    def __init__(self) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise NonRetryableDataError("pip install yfinance") from exc
        self._yf = yf

    def get_daily(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._yf.Ticker(yahoo_symbol(symbol)).history(
            start=start, end=end, interval="1d", auto_adjust=True, actions=False)

    def get_daily_batch(self, symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        """Download many symbols in one request (much faster than one call per symbol)."""
        ysyms = [yahoo_symbol(s) for s in symbols]
        data = self._yf.download(tickers=ysyms, start=start, end=end, interval="1d", auto_adjust=True,
                                 actions=False, group_by="ticker", threads=True, progress=False)
        out: dict[str, pd.DataFrame] = {}
        if data is None or data.empty:
            return out
        if isinstance(data.columns, pd.MultiIndex):
            present = set(data.columns.get_level_values(0))
            for sym, ysym in zip(symbols, ysyms):
                if ysym in present:
                    df = data[ysym].dropna(how="all")
                    if not df.empty:
                        out[sym] = df
        elif len(symbols) == 1:
            out[symbols[0]] = data
        return out


# --------------------------------------------------------------------------- #
# Fetcher
# --------------------------------------------------------------------------- #
class MarketDataFetcher:
    """Safe, throttled access to daily NSE history and pre-open quotes.

    Args:
        config: Data settings (retries, rate limit, sources).
        provider: Daily-history provider; built from ``config.provider`` if omitted.
        preopen: Pre-open source; built from ``config.preopen_source`` if omitted.
        sleeper: Sleep function (overridable for fast tests).
    """

    def __init__(self, config: DataConfig, provider: DataProvider | None = None,
                 preopen: "PreOpenSource | None" = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        from data_ingestion.preopen import build_preopen_source
        self.config = config
        self.provider = provider or self._build_provider(config)
        self.preopen = preopen or build_preopen_source(config)
        self._sleep = sleeper
        self._limiter = RateLimiter(config.requests_per_minute, 60.0, sleeper=sleeper)

    @staticmethod
    def _build_provider(config: DataConfig) -> DataProvider:
        if config.provider == "yfinance":
            return YFinanceProvider()
        if config.provider == "synthetic":
            from data_ingestion.synthetic import SyntheticMarket
            return SyntheticMarket()
        raise NonRetryableDataError(f"Unknown provider {config.provider}")

    def _with_retry(self, fn: Callable[[], T], description: str) -> T:
        def _call() -> T:
            self._limiter.acquire()
            return fn()

        return call_with_backoff(
            _call, max_retries=self.config.max_retries,
            base_delay=self.config.backoff_base_seconds,
            max_delay=self.config.backoff_max_seconds,
            description=description, sleeper=self._sleep,
        )

    def fetch_history(self, symbol: str, lookback_days: int | None = None,
                      now: datetime | None = None) -> pd.DataFrame:
        """Fetch sanitized daily OHLCV (split/bonus adjusted) for one NSE symbol."""
        now = (now or datetime.now(IST)).astimezone(IST)
        days = lookback_days or self.config.history_lookback_days

        def _get() -> pd.DataFrame:
            df = self.provider.get_daily(symbol, now - timedelta(days=days), now + timedelta(days=1))
            if df is None or df.empty:
                # Yahoo answers unknown/delisted symbols with an empty frame; retrying cannot help.
                raise NonRetryableDataError(f"{symbol}: no data from {self.provider.name} (not listed there?)")
            return df

        raw = self._with_retry(_get, f"{self.provider.name}:{symbol}:1d")
        return sanitize_ohlcv(raw, ticker=symbol)

    def fetch_many(self, symbols: Iterable[str], lookback_days: int | None = None,
                   now: datetime | None = None, batch_size: int = 100,
                   progress: Callable[[int, int], None] | None = None) -> dict[str, pd.DataFrame]:
        """Fetch daily history for many symbols; failures are logged and skipped.

        Uses the provider's batch download when available (one request per
        ``batch_size`` symbols), retrying individually only for a few misses.
        """
        symbols = list(symbols)
        results: dict[str, pd.DataFrame] = {}
        requested = len(symbols)
        if symbols and hasattr(self.provider, "get_daily_batch"):
            now_ = (now or datetime.now(IST)).astimezone(IST)
            days = lookback_days or self.config.history_lookback_days
            start, end = now_ - timedelta(days=days), now_ + timedelta(days=1)

            def batch(chunk: list[str], label: str) -> dict[str, pd.DataFrame]:
                # yfinance reports throttling as "no data" rather than an error, so an (almost) empty
                # answer for a whole chunk is treated as a failure and retried with backoff.
                def call() -> dict[str, pd.DataFrame]:
                    raw = self.provider.get_daily_batch(chunk, start, end)
                    if len(chunk) >= 5 and len(raw) < 0.2 * len(chunk):
                        raise DataFetchError(f"only {len(raw)}/{len(chunk)} symbols returned (rate-limited?)")
                    return raw
                return self._with_retry(call, label)

            for i in range(0, len(symbols), batch_size):
                chunk = symbols[i:i + batch_size]
                if progress:
                    progress(i, len(symbols))
                try:
                    raw = batch(chunk, f"{self.provider.name}:batch[{i}:{i + len(chunk)}]")
                except DataFetchError as exc:
                    logger.warning("Batch %d-%d failed (%s); retrying in smaller pieces", i, i + len(chunk), exc)
                    raw = {}
                    for j in range(0, len(chunk), 20):  # smaller, slower requests get through throttling
                        self._sleep(2.0)
                        try:
                            raw.update(batch(chunk[j:j + 20], f"{self.provider.name}:small[{i + j}]"))
                        except DataFetchError as exc2:
                            logger.error("Symbols %d-%d unavailable after retries: %s", i + j, i + j + 20, exc2)
                for sym, df in raw.items():
                    try:
                        results[sym] = sanitize_ohlcv(df, ticker=sym)
                    except DataFetchError as exc:
                        logger.warning("Skipping %s: %s", sym, exc)
            missing = [s for s in symbols if s not in results]
            if len(missing) > 25:
                logger.warning("%d of %d symbols could not be downloaded this time: %s%s", len(missing), requested,
                               ", ".join(missing[:25]), " ..." if len(missing) > 25 else "")
                missing = []
            symbols = missing
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_history(symbol, lookback_days, now)
            except DataFetchError as exc:
                logger.warning("Skipping %s: %s", symbol, exc)
        logger.info("Fetched daily history for %d/%d symbols", len(results), requested)
        return results

    def fetch_preopen(self, symbols: Iterable[str]) -> "dict[str, PreOpenQuote]":
        """Fetch today's pre-open quotes (one request for all symbols, with retries)."""
        symbols = list(symbols)
        quotes = self._with_retry(lambda: self.preopen.fetch(symbols), f"{self.preopen.name}:preopen")
        if missing := sorted(set(symbols) - set(quotes)):
            logger.warning("No pre-open quote for %d symbols: %s", len(missing), ", ".join(missing))
        return quotes

