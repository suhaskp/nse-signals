"""NSE pre-open session data (09:00-09:15 IST).

During pre-open, orders are collected until ~09:08, then a call auction fixes
the indicative equilibrium price (IEP). The IEP becomes the day's opening
price, so a scan run between 09:08 and 09:15 sees the actual open before
continuous trading starts.

Sources:
    ``NSEPreOpenSource``   NSE website JSON. Free but unofficial: it needs
                           browser-like headers and cookies, may change or
                           block automated use, and is subject to NSE's terms.
    ``KitePreOpenSource``  Zerodha Kite Connect quotes (needs an API key and a
                           daily access token). Verify in your account that
                           ``last_price`` reflects the IEP after 09:08.
    Synthetic               See :mod:`data_ingestion.synthetic`.

:class:`PreOpenHistory` stores each day's pre-open quantities so pre-open
relative volume can be computed once enough days accumulate.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from config.config import DataConfig
from data_ingestion.data_fetcher import IST, DataFetchError, NonRetryableDataError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreOpenQuote:
    """One symbol's pre-open state.

    Attributes:
        iep: Indicative equilibrium price (the expected open).
        prev_close: Previous session close as reported by the pre-open feed.
        preopen_quantity: Shares matched at the IEP (NaN if the source lacks it).
        avg_preopen_quantity: Mean pre-open quantity over prior sessions,
            filled from :class:`PreOpenHistory` (NaN if too little history).
    """

    symbol: str
    iep: float
    prev_close: float
    preopen_quantity: float
    as_of: datetime
    source: str
    avg_preopen_quantity: float = float("nan")

    @property
    def preopen_rvol(self) -> float:
        """Today's pre-open quantity relative to its recent average (NaN if unknown)."""
        if not (self.avg_preopen_quantity > 0 and self.preopen_quantity >= 0):
            return float("nan")
        return self.preopen_quantity / self.avg_preopen_quantity


class PreOpenSource(Protocol):
    name: str

    def fetch(self, symbols: list[str]) -> dict[str, PreOpenQuote]: ...


def _num(value: Any) -> float:
    """Parse NSE numbers that may arrive as strings with commas or '-'."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


class NSEPreOpenSource:
    """Pre-open data from NSE's public website JSON (unofficial)."""

    name = "nse"
    BASE = "https://www.nseindia.com"
    PAGE = BASE + "/market-data/pre-open-market-cm-and-emerge-market"
    API = BASE + "/api/market-data-pre-open"
    HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": PAGE,
    }

    def __init__(self, key: str = "ALL", timeout: float = 10.0, session: Any = None) -> None:
        self.key, self.timeout = key, timeout
        self._session = session
        self._primed = False

    def _get_session(self) -> Any:
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update(self.HEADERS)
        if not self._primed:  # NSE sets the cookies the API requires on the HTML page
            self._session.get(self.PAGE, timeout=self.timeout)
            self._primed = True
        return self._session

    @staticmethod
    def parse(payload: dict[str, Any], symbols: list[str], as_of: datetime) -> dict[str, PreOpenQuote]:
        """Parse the ``/api/market-data-pre-open`` payload into quotes."""
        if not isinstance(payload, dict) or "data" not in payload:
            raise DataFetchError("NSE pre-open payload has no 'data' field")
        wanted, out = set(symbols), {}
        for item in payload["data"]:
            md = item.get("metadata", {}) if isinstance(item, dict) else {}
            sym = str(md.get("symbol", "")).upper()
            if sym not in wanted:
                continue
            iep = _num(md.get("iep"))
            if not iep > 0:
                iep = _num(md.get("lastPrice"))
            prev = _num(md.get("previousClose"))
            if not (iep > 0 and prev > 0):
                logger.debug("%s: incomplete NSE pre-open record", sym)
                continue
            out[sym] = PreOpenQuote(sym, iep, prev, _num(md.get("finalQuantity")), as_of, "nse")
        return out

    def fetch(self, symbols: list[str]) -> dict[str, PreOpenQuote]:
        session = self._get_session()
        resp = session.get(self.API, params={"key": self.key}, timeout=self.timeout)
        if resp.status_code in (401, 403):
            self._primed = False  # cookies expired; re-prime on the retry
            raise DataFetchError(f"NSE returned {resp.status_code}; refreshing cookies")
        resp.raise_for_status()
        return self.parse(resp.json(), symbols, datetime.now(IST))


class KitePreOpenSource:
    """Pre-open quotes from Zerodha Kite Connect (read-only; no orders are placed)."""

    name = "kite"
    MAX_INSTRUMENTS = 500

    def __init__(self, api_key: str, access_token: str, client: Any = None) -> None:
        if client is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:  # pragma: no cover
                raise NonRetryableDataError("pip install kiteconnect") from exc
            client = KiteConnect(api_key=api_key)
            client.set_access_token(access_token)
        self._kite = client

    def fetch(self, symbols: list[str]) -> dict[str, PreOpenQuote]:
        now, out = datetime.now(IST), {}
        for i in range(0, len(symbols), self.MAX_INSTRUMENTS):
            batch = [f"NSE:{s}" for s in symbols[i:i + self.MAX_INSTRUMENTS]]
            for key, q in self._kite.quote(batch).items():
                sym = key.split(":", 1)[1]
                iep, prev = _num(q.get("last_price")), _num((q.get("ohlc") or {}).get("close"))
                qty = _num(q.get("volume"))
                if iep > 0 and prev > 0:
                    out[sym] = PreOpenQuote(sym, iep, prev, qty if qty > 0 else float("nan"), now, "kite")
        return out


def build_preopen_source(config: DataConfig) -> PreOpenSource:
    """Instantiate the configured pre-open source."""
    if config.preopen_source == "nse":
        return NSEPreOpenSource(config.nse_preopen_key)
    if config.preopen_source == "kite":
        return KitePreOpenSource(config.kite_api_key or "", config.kite_access_token or "")
    if config.preopen_source == "synthetic":
        from data_ingestion.synthetic import SyntheticMarket
        return SyntheticMarket()
    raise NonRetryableDataError(f"Unknown pre-open source {config.preopen_source}")


class PreOpenHistory:
    """CSV store of daily pre-open quantities, for pre-open relative volume."""

    COLUMNS = ["session_date", "symbol", "iep", "prev_close", "preopen_quantity", "source"]

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.path, dtype={"session_date": str, "symbol": str})

    def record(self, quotes: dict[str, PreOpenQuote], session_date: date) -> None:
        """Store today's quotes, replacing any earlier rows for the same day."""
        rows = pd.DataFrame([{"session_date": session_date.isoformat(), "symbol": q.symbol, "iep": q.iep,
                              "prev_close": q.prev_close, "preopen_quantity": q.preopen_quantity,
                              "source": q.source} for q in quotes.values()], columns=self.COLUMNS)
        hist = self.load()
        hist = hist[hist["session_date"] != session_date.isoformat()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frames = [f for f in (hist, rows) if not f.empty]
        combined = pd.concat(frames, ignore_index=True) if frames else rows
        combined.to_csv(self.path, index=False)

    def average_quantity(self, symbol: str, before: date, lookback: int, min_days: int) -> float:
        """Mean pre-open quantity over the last ``lookback`` sessions before ``before``.

        Returns NaN when fewer than ``min_days`` usable sessions exist.
        """
        hist = self.load()
        if hist.empty:
            return float("nan")
        sel = hist[(hist["symbol"] == symbol) & (hist["session_date"] < before.isoformat())]
        qty = pd.to_numeric(sel.sort_values("session_date")["preopen_quantity"], errors="coerce").dropna()
        qty = qty.tail(lookback)
        return float(qty.mean()) if len(qty) >= min_days and not math.isnan(qty.mean()) else float("nan")
