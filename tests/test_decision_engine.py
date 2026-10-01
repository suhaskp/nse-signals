from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from config.config import load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from intelligence_engine import IntelligenceEngine, apply_portfolio_limits
from research.market import regime_detail
from risk_engine.risk_manager import RiskManager
from tests.planted import planted_market


def _bench(path):
    idx = pd.bdate_range(end="2026-09-25", periods=len(path))
    c = np.asarray(path, float)
    return pd.DataFrame({"Open": c, "High": c * 1.002, "Low": c * 0.998, "Close": c, "Volume": 1e6}, index=idx)


def test_regime_states_and_confidence():
    h, _ = planted_market(n_stocks=20, n_days=400)
    up = _bench(20000 * np.exp(np.linspace(0, 0.4, 400)))
    down = _bench(20000 * np.exp(np.linspace(0, -0.4, 400)))
    d_up, d_down = regime_detail(up, h), regime_detail(down, h)
    assert d_up["state"].startswith("Risk-on") and d_down["state"].startswith("Risk-off")
    assert 0 <= d_up["confidence"] <= 100 and len(d_up["components"]) == 8
    assert d_down["exposure"].startswith("0")


def test_concentration_limits_sector_and_correlation():
    cfg = load_config(overrides={"risk": {"max_per_sector": 1, "max_correlation": 0.8}})
    rm = RiskManager(cfg.risk)
    idx = pd.bdate_range(end="2026-09-25", periods=80)
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    twin = base * (1 + rng.normal(0, 0.001, 80))              # moves in lockstep with A
    other = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    hist = {t: pd.DataFrame({"Close": c}, index=idx) for t, c in (("A", base), ("B", twin), ("C", other), ("D", other * 1.1))}
    rows = [{"Action": "BUY", "Ticker": t, "Sector": s, "Qty": 0, "Risk (INR)": 0.0}
            for t, s in (("A", "Metals"), ("B", "IT"), ("C", "Metals"), ("D", "Pharma"))]
    plans = {("BUY", t): rm.plan_from_levels(t, 500.0, 470.0, 590.0, 20.0) for t in "ABCD"}
    recs = apply_portfolio_limits(pd.DataFrame(rows), plans, rm, hist).set_index("Ticker")
    assert recs.at["A", "Qty"] > 0
    assert recs.at["B", "Allocation"].startswith("Skipped: moves with A")
    assert recs.at["C", "Allocation"].startswith("Skipped: sector limit")
    assert recs.at["D", "Qty"] > 0


def _engine(tmp_path, now, **extra):
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}, **extra})
    m = SyntheticMarket(now=now)
    return IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                              auto_universe=False), cfg


def test_watch_history_counts_days_and_expires(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    eng, cfg = _engine(tmp_path, now, buy_quality={"watch_expiry_sessions": 2})
    q = pd.DataFrame({"Ticker": ["X", "Y"], "Readiness": [90.0, 70.0], "State": ["WATCH CLOSELY", "WAIT"],
                      "g_breakout": [False, True]})
    for d in ("2026-09-22", "2026-09-23", "2026-09-24"):
        out = eng._watch_history(q, date.fromisoformat(d))
    assert list(out["Days on watch"]) == [3, 3]
    assert out.set_index("Ticker").at["X", "State"] == "EXPIRED"      # no breakout after 2+ sessions
    assert out.set_index("Ticker").at["Y", "State"] == "WAIT"         # broken out: not expired


def test_report_has_regime_detail_and_card_inputs(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    eng, _ = _engine(tmp_path, now)
    rep = eng.refresh(now)
    assert rep.regime_detail is not None and rep.regime_detail["state"] in {
        "Risk-on", "Risk-on, weakening", "Transition", "Risk-off, improving", "Risk-off"}
    assert {"Days on watch", "SMA50", "Readiness"} <= set(rep.quality.columns)
