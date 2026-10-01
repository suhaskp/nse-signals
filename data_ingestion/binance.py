"""Binance public market data (no API key, read-only).

Uses Binance's public market-data endpoint (data-api.binance.vision), falling back to api.binance.com.
Daily candles close at 00:00 UTC (05:30 IST).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from data_ingestion.data_fetcher import DataFetchError, NonRetryableDataError

logger = logging.getLogger(__name__)
BASES = ("https://data-api.binance.vision", "https://api.binance.com")
STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "PYUSD", "EUR", "GBP", "TRY", "BRL", "AEUR", "EURI",
           "USD1", "XUSD", "BFUSD", "WBTC", "WBETH", "PAXG", "USDE"}
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")


class BinanceProvider:
    """``DataProvider`` for Binance spot daily candles, plus universe and 24h ticker helpers."""

    name = "binance"

    def __init__(self, session=None, timeout: float = 15.0) -> None:
        if session is None:
            import requests
            session = requests.Session()
            session.headers.update({"User-Agent": "nse-signals-dashboard/1.0"})
        self.session, self.timeout = session, timeout
        self.lot_steps: dict[str, float] = {}

    def _get(self, path: str, params: dict | None = None):
        last = None
        for base in BASES:
            try:
                r = self.session.get(base + path, params=params, timeout=self.timeout)
                if r.status_code == 400:
                    raise NonRetryableDataError(f"Binance rejected {path} {params}: {r.text[:200]}")
                if r.status_code in (418, 429):
                    raise DataFetchError(f"Binance rate limit ({r.status_code})")
                if r.status_code == 451:
                    last = DataFetchError(f"{base} is not available from this location (HTTP 451)")
                    continue
                r.raise_for_status()
                return r.json()
            except NonRetryableDataError:
                raise
            except Exception as exc:  # noqa: BLE001 - try the next endpoint
                last = exc
        raise DataFetchError(f"Binance unavailable: {last}")

    @staticmethod
    def parse_klines(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["t", "Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
        return df.astype(float)

    def get_daily(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
        end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
        frames = []
        while start_ms < end_ms:
            rows = self._get("/api/v3/klines", {"symbol": symbol, "interval": "1d", "startTime": start_ms,
                                                 "endTime": end_ms, "limit": 1000})
            if not rows:
                break
            frames.append(self.parse_klines(rows))
            start_ms = int(rows[-1][0]) + 86_400_000
            if len(rows) < 1000:
                break
        return pd.concat(frames) if frames else pd.DataFrame()

    def top_symbols(self, n: int = 80, min_quote_volume: float = 0.0) -> list[str]:
        """Most-traded USDT spot pairs, excluding stablecoins and leveraged tokens; remembers lot sizes."""
        info = self._get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
        ok = {}
        for s in info.get("symbols", []):
            base = s.get("baseAsset", "")
            if (s.get("status") != "TRADING" or s.get("quoteAsset") != "USDT" or base in STABLES
                    or base.endswith(LEVERAGED)):
                continue
            step = next((float(f["stepSize"]) for f in s.get("filters", []) if f.get("filterType") == "LOT_SIZE"), 0.0)
            ok[s["symbol"]] = step
        tickers = self._get("/api/v3/ticker/24hr")
        vols = sorted(((t["symbol"], float(t.get("quoteVolume", 0))) for t in tickers if t["symbol"] in ok),
                      key=lambda x: -x[1])
        chosen = [s for s, v in vols if v >= min_quote_volume][:n]
        self.lot_steps.update({s: ok[s] for s in chosen})
        return chosen

    def movers_24h(self, symbols: list[str]) -> pd.DataFrame:
        """24-hour change and quote volume for the given symbols."""
        tickers = self._get("/api/v3/ticker/24hr")
        want = set(symbols)
        rows = [{"Symbol": t["symbol"], "Price": float(t["lastPrice"]), "24h %": float(t["priceChangePercent"]),
                 "24h volume (USDT)": float(t["quoteVolume"])} for t in tickers if t["symbol"] in want]
        return pd.DataFrame(rows).sort_values("24h %", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
