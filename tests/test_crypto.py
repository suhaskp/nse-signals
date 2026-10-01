from datetime import datetime

import pandas as pd
import pytest

from config.config import load_config
from crypto_engine import CryptoEngine, crypto_rule_version, size_position
from data_ingestion.binance import BinanceProvider
from data_ingestion.data_fetcher import IST
from data_ingestion.synthetic import SyntheticCrypto
from storage import rule_version


class FakeResp:
    def __init__(self, data, code=200):
        self._d, self.status_code, self.text = data, code, ""

    def json(self):
        return self._d

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, routes, blocked=False):
        self.routes, self.blocked, self.calls = routes, blocked, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if self.blocked and "data-api" in url:
            return FakeResp({}, 451)  # first endpoint unavailable from this location -> fall back
        for path, data in self.routes.items():
            if url.endswith(path):
                return FakeResp(data(params) if callable(data) else data)
        raise AssertionError(url)


def test_universe_filters_stablecoins_and_leveraged_tokens():
    info = {"symbols": [
        {"symbol": s, "baseAsset": b, "quoteAsset": "USDT", "status": "TRADING",
         "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}]}
        for s, b in (("BTCUSDT", "BTC"), ("USDCUSDT", "USDC"), ("ETHUPUSDT", "ETHUP"), ("SOLUSDT", "SOL"))]}
    tick = [{"symbol": "BTCUSDT", "quoteVolume": "9e9", "lastPrice": "60000", "priceChangePercent": "1"},
            {"symbol": "SOLUSDT", "quoteVolume": "1e9", "lastPrice": "150", "priceChangePercent": "5"},
            {"symbol": "USDCUSDT", "quoteVolume": "5e9", "lastPrice": "1", "priceChangePercent": "0"}]
    p = BinanceProvider(session=FakeSession({"/exchangeInfo": info, "/ticker/24hr": tick}, blocked=True))
    assert p.top_symbols(10) == ["BTCUSDT", "SOLUSDT"] and p.lot_steps["SOLUSDT"] == 0.001


def test_klines_parse_and_paginate():
    day = 86_400_000
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)

    def klines(params):
        start = params["startTime"]
        n = min(1000, (params["endTime"] - start) // day + 1)
        return [[start + i * day, "1", "2", "0.5", "1.5", "100", 0, "150", 0, 0, 0, 0] for i in range(int(n))]

    p = BinanceProvider(session=FakeSession({"/klines": klines}))
    df = p.get_daily("BTCUSDT", pd.Timestamp(t0, unit="ms", tz="UTC").to_pydatetime(),
                     pd.Timestamp(t0 + 1499 * day, unit="ms", tz="UTC").to_pydatetime())
    assert len(df) == 1500 and list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_unique and str(df.index.tz) == "UTC"


def test_fractional_sizing():
    qty, risk = size_position(60_000, 57_000, 10_000, 0.01, 0.20, 0.00001)
    assert qty == pytest.approx(0.03333, abs=1e-5) and risk <= 100.01      # 1% of 10,000 USDT risked
    qty2, _ = size_position(1.0, 0.95, 10_000, 0.01, 0.20, 1)
    assert qty2 == 2000                                                       # capped at 20% of equity
    assert size_position(100, 99, 40, 0.01, 0.2, 0.01) == (0.0, 0.0)          # 8 USDT: below the ~10 USDT minimum


def test_crypto_engine_end_to_end(tmp_path):
    cfg = load_config(overrides={"cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"),
                                 "data_home": str(tmp_path / "h")})
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    rep = CryptoEngine(cfg, SyntheticCrypto(now=now)).refresh(now)
    assert rep.symbols >= 20 and rep.session < now.date()           # only completed UTC days
    assert set(rep.regime) >= {"risk_on", "btc_vs_200d", "state"}
    assert set(rep.signals["Action"]) <= {"BUY", "WATCH"}
    if not rep.signals.empty:
        sized = rep.signals[rep.signals["Qty"] > 0]
        assert (sized["Risk (INR)"] <= cfg.crypto.account_equity_usdt * cfg.crypto.risk_per_trade_pct + 0.01).all()
        assert sized["Ticker"].is_unique
    assert rep.lab is not None and "summary" in rep.movers_proxy


def test_crypto_does_not_change_the_nse_rule_version():
    a = load_config()
    b = load_config(overrides={"crypto": {"universe_size": 30}})
    assert rule_version(a) == rule_version(b)                            # NSE freeze unaffected
    assert crypto_rule_version(a) != crypto_rule_version(b)


def test_coverage_is_reported(tmp_path):
    cfg = load_config(overrides={"cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"),
                                 "data_home": str(tmp_path / "h")})
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    rep = CryptoEngine(cfg, SyntheticCrypto(now=now)).refresh(now)
    cov = rep.coverage
    assert cov["requested"] >= cov["used"] == rep.symbols
    assert isinstance(cov["short_history"], list) and isinstance(cov["no_data"], list)


def test_crypto_regime_and_strength_vs_btc():
    import numpy as np
    from crypto_engine import crypto_regime, strength_vs_btc
    idx = pd.bdate_range(end="2026-09-30", periods=400)
    btc = pd.DataFrame({"Close": 50_000 * np.exp(np.linspace(0, 0.5, 400))}, index=idx)
    lead = pd.DataFrame({"Close": 100 * np.exp(np.linspace(0, 1.0, 400))}, index=idx)   # beats BTC
    lag = pd.DataFrame({"Close": 100 * np.exp(np.linspace(0, 0.2, 400))}, index=idx)    # rises, but less than BTC
    brd = pd.Series(np.linspace(0.4, 0.9, 400), index=idx)
    reg = crypto_regime(btc, {"BTCUSDT": btc, "A": lead, "B": lag}, brd, "BTCUSDT")
    assert 0 <= reg["score"] <= 100 and len(reg["components"]) == 7 and reg["alt_share_beating_btc_30d"] == 0.5
    vb = strength_vs_btc({"A": lead, "B": lag}, btc).set_index("Ticker")
    assert vb.at["A", "vs BTC 30d %"] > 0 > vb.at["B", "vs BTC 30d %"]           # B rose, but is weaker than BTC


def test_evidence_preliminary_label():
    from research.evidence import strength
    assert strength(49) == "PRELIMINARY" and strength(150) == "MEDIUM" and strength(300) == "HIGH"


def test_crypto_open_alerts(tmp_path):
    from alerts import AlertCenter
    cfg = load_config(overrides={"cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"),
                                 "data_home": str(tmp_path / "h")})
    eng = CryptoEngine(cfg, SyntheticCrypto(now=datetime(2026, 9, 28, 16, 30, tzinfo=IST)))
    res = pd.DataFrame({"session": ["2026-09-26", "2026-09-26"], "Ticker": ["AUSDT", "BUSDT"],
                        "entry_date": ["2026-09-27", "2026-09-27"], "entry": [10.0, None], "stop": [9.0, 8.0],
                        "outcome": ["open", "not entered: SKIP: opened 4.0% above the signal close (chasing)"]})
    eng._open_alerts(res, datetime(2026, 9, 27, 6, 0, tzinfo=IST))
    texts = [a["text"] for a in AlertCenter(tmp_path / "o" / "alerts.json").recent()]
    assert any("entered at 10" in t for t in texts) and any("skip BUSDT" in t and "chasing" in t for t in texts)
