"""Deterministic synthetic NSE data for offline demos and tests. NOT real market data.

:class:`SyntheticMarket` serves both daily history and pre-open quotes.
Roughly one symbol in three is generated as a trending "gapper" with a
1.5-5% pre-open gap and elevated pre-open quantity, so the screener has
something to find.
"""
from __future__ import annotations

import zlib
from datetime import date, datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from data_ingestion.data_fetcher import IST
from data_ingestion.preopen import PreOpenHistory, PreOpenQuote


class SyntheticMarket:
    """Offline daily-history provider *and* pre-open source."""

    name = "synthetic"

    def __init__(self, now: datetime | None = None, n_days: int = 1300, seed: int = 11) -> None:
        self.now = (now or datetime.now(IST)).astimezone(IST)
        self.n_days, self.seed = n_days, seed
        self._daily: dict[str, pd.DataFrame] = {}

    def _rng(self, symbol: str, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(zlib.crc32(symbol.encode()) ^ self.seed ^ salt)

    SECTORS = ("Financial Services", "Information Technology", "Energy", "FMCG", "Automobile",
               "Healthcare", "Capital Goods", "Metals")

    def sector(self, symbol: str) -> str:
        return self.SECTORS[zlib.crc32(symbol.encode()) % len(self.SECTORS)]

    def is_gapper(self, symbol: str) -> bool:
        return (zlib.crc32(symbol.encode()) ^ self.seed) % 3 == 0

    def daily(self, symbol: str) -> pd.DataFrame:
        if symbol in self._daily:
            return self._daily[symbol]
        rng = self._rng(symbol)
        today = self.now.date()
        has_today = today.weekday() < 5 and self.now.time() >= dtime(9, 15)
        idx = pd.bdate_range(end=today if has_today else today - timedelta(days=1), periods=self.n_days)
        vol = rng.uniform(0.006, 0.010) if symbol.startswith("^") else rng.uniform(0.010, 0.022)
        rets = rng.normal(rng.uniform(-0.0002, 0.0007), vol, len(idx))
        if self.is_gapper(symbol):
            rets[-40:] = rng.normal(0.0035, vol * 0.6, 40)
        close = rng.uniform(150, 6_000) * np.exp(np.cumsum(rets))
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * (1 + rng.normal(0, vol / 3, len(idx)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, len(idx))))
        turnover = rng.lognormal(np.log(rng.uniform(4e9, 2e10)), 0.35, len(idx))  # ₹400cr-₹2,000cr/day
        volume = (turnover / close).round()
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
        self._inject_range_break(symbol, df)
        if has_today and self.now.time() < dtime(15, 30):  # today's bar is still forming
            elapsed = (self.now.hour * 60 + self.now.minute - 555) / 375
            df.loc[df.index[-1], "Volume"] = round(df["Volume"].iloc[-1] * max(elapsed, 0.05))
        self._daily[symbol] = df
        return df

    def _inject_range_break(self, symbol: str, df: pd.DataFrame) -> None:
        """Make the last bar a 125-day breakout/breakdown when price is already near the edge (demo realism)."""
        prior = df.iloc[-126:-1]
        hi, lo, last = prior["High"].max(), prior["Low"].min(), df.iloc[-1]
        avg_vol = prior["Volume"].mean()
        i = df.index[-1]
        if last["Close"] >= hi * 0.94:
            new_close = hi * 1.012
            df.loc[i, ["Open", "Close", "High", "Volume"]] = [last["Open"], new_close,
                                                              max(new_close * 1.004, last["High"]), avg_vol * 2.6]
        elif last["Close"] <= lo * 1.06:
            new_close = lo * 0.99
            df.loc[i, ["Open", "Close", "Low", "Volume"]] = [last["Open"], new_close,
                                                             min(new_close * 0.996, last["Low"]), avg_vol * 0.8]
        df.loc[i, "High"] = df.loc[i, ["Open", "High", "Close"]].max()
        df.loc[i, "Low"] = df.loc[i, ["Open", "Low", "Close"]].min()

    def get_daily(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        df = self.daily(symbol)
        return df[(df.index >= pd.Timestamp(start.date())) & (df.index < pd.Timestamp(end.date()))]

    def get_daily_batch(self, symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        return {s: self.get_daily(s, start, end) for s in symbols}

    def _normal_quantity(self, symbol: str, rng: np.random.Generator) -> float:
        return float(self.daily(symbol)["Volume"].tail(20).mean() * 0.012 * rng.lognormal(0, 0.25))

    def fetch(self, symbols: list[str]) -> dict[str, PreOpenQuote]:
        out = {}
        for s in symbols:
            rng = self._rng(s, salt=int(self.now.strftime("%Y%m%d")))
            d = self.daily(s)
            prev = float(d.loc[d.index < pd.Timestamp(self.now.date()), "Close"].iloc[-1])
            if self.is_gapper(s):
                iep, qty = prev * (1 + rng.uniform(0.015, 0.05)), self._normal_quantity(s, rng) * 3.5
            else:
                iep, qty = prev * (1 + rng.normal(0, 0.004)), self._normal_quantity(s, rng)
            out[s] = PreOpenQuote(s, round(iep, 2), prev, round(qty), self.now, "synthetic")
        return out

    def seed_history(self, store: PreOpenHistory, symbols: list[str], n_days: int = 20) -> None:
        """Write ``n_days`` of normal pre-open quantities before today (demo only)."""
        if not store.load().empty:
            return
        days = pd.bdate_range(end=self.now.date() - timedelta(days=1), periods=n_days)
        for d in days:
            quotes = {}
            for s in symbols:
                rng = self._rng(s, salt=int(d.strftime("%Y%m%d")))
                close = self.daily(s)
                prev = float(close.loc[close.index < d, "Close"].iloc[-1])
                quotes[s] = PreOpenQuote(s, prev, prev, round(self._normal_quantity(s, rng)), self.now, "synthetic")
            store.record(quotes, d.date())


def demo_now(market_time: str = "09:10") -> datetime:
    """A time just after price discovery on the most recent weekday (IST)."""
    d: date = datetime.now(IST).date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    hh, mm = (int(x) for x in market_time.split(":"))
    return datetime.combine(d, dtime(hh, mm), tzinfo=IST)


class SyntheticCrypto(SyntheticMarket):
    """Offline stand-in for BinanceProvider (demo and tests). NOT real market data."""

    name = "synthetic-crypto"
    COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
             "DOTUSDT", "TRXUSDT", "LTCUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
             "FILUSDT", "INJUSDT", "SUIUSDT", "AAVEUSDT", "ETCUSDT", "HBARUSDT")

    def __init__(self, now=None, n_days: int = 1300, seed: int = 23) -> None:
        super().__init__(now=now, n_days=n_days, seed=seed)
        self.lot_steps: dict[str, float] = {}

    def top_symbols(self, n: int = 80, min_quote_volume: float = 0.0) -> list[str]:
        syms = list(self.COINS[:n])
        self.lot_steps.update({s: 0.0001 for s in syms})
        return syms

    def movers_24h(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            d = self.daily(s)
            rows.append({"Symbol": s, "Price": float(d["Close"].iloc[-1]),
                         "24h %": round(100 * float(d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1), 2),
                         "24h volume (USDT)": float(d["Close"].iloc[-1] * d["Volume"].iloc[-1])})
        return pd.DataFrame(rows).sort_values("24h %", ascending=False).reset_index(drop=True)
