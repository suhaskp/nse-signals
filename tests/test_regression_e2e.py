"""Regression tests for bugs found in the end-to-end simulation."""
from datetime import datetime

import numpy as np
import pandas as pd

from config.config import load_config
from data_ingestion.data_fetcher import IST
from intelligence_engine import apply_portfolio_limits
from monitoring.forward_record import held_positions
from risk_engine.risk_manager import RiskManager


def test_no_second_position_in_a_stock_already_held_and_held_risk_counts():
    cfg = load_config(overrides={"risk": {"max_open_positions": 2, "max_portfolio_heat_pct": 0.02}})
    rm = RiskManager(cfg.risk)
    plans = {("BUY", t): rm.plan_from_levels(t, 500.0, 470.0, 590.0, 20.0) for t in ("S31", "A", "B")}
    rows = [{"Action": "BUY", "Ticker": t, "Sector": s, "Qty": 0, "Risk (INR)": 0.0}
            for t, s in (("S31", "IT"), ("A", "Pharma"), ("B", "Energy"))]
    held = [("S31", 10_000.0, "2026-10-07")]
    recs = apply_portfolio_limits(pd.DataFrame(rows), plans, rm, None, held).set_index("Ticker")
    assert recs.at["S31", "Qty"] == 0 and recs.at["S31", "Allocation"].startswith("Already held")
    taken = recs[recs["Qty"] > 0]
    assert len(taken) <= 1                                           # 2 positions max, one already open
    assert taken["Risk (INR)"].sum() <= cfg.risk.account_equity * 0.02 - 10_000 + 1   # held risk counted


def test_held_positions_from_forward_results():
    res = pd.DataFrame({"Ticker": ["X", "Y", "Z"], "session": ["2026-10-06", "2026-10-07", "2026-10-01"],
                        "outcome": ["open", "waiting for the next open", "target"], "risk_amount": [5000.0, np.nan, 1.0]})
    held = held_positions(res, 10_000.0)
    assert [h[0] for h in held] == ["X", "Y"] and held[1][1] == 10_000.0   # missing risk -> default budget


def test_opening_check_uses_the_open_not_the_current_price(tmp_path):
    from data_ingestion import MarketDataFetcher
    from data_ingestion.synthetic import SyntheticMarket
    from intelligence_engine import IntelligenceEngine, morning_check
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "data_home": str(tmp_path / "h"), "buy_quality": {"regime_required": False, "volume_multiple": 1.0,
                                                          "min_close_location": 0.0},
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 30, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    now = datetime(2026, 9, 29, 10, 30, tzinfo=IST)
    m = SyntheticMarket(now=now)
    eng = IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                             auto_universe=False)
    rep = eng.refresh(now)
    tab, note = morning_check(eng, rep, now)
    if tab is not None:
        assert "opening price" in note
        today = pd.Timestamp(now.date())
        for _, r in tab[tab["Today price"].notna()].iterrows():
            assert r["Today price"] == round(float(m.daily(r["Ticker"]).at[today, "Open"]), 2)


def test_engine_path_passes_histories_and_held_positions(tmp_path, monkeypatch):
    """The real engine must call the allocator with price histories (correlation check) and held positions."""
    import intelligence_engine as ie
    from data_ingestion import MarketDataFetcher
    from data_ingestion.synthetic import SyntheticMarket
    seen = {}
    real = ie.apply_portfolio_limits

    def spy(recs, plans, risk, histories=None, held=None):
        seen["histories"], seen["held"] = histories, held
        return real(recs, plans, risk, histories, held)

    monkeypatch.setattr(ie, "apply_portfolio_limits", spy)
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "data_home": str(tmp_path / "h"),
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 30, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    m = SyntheticMarket(now=now)
    ie.IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                          auto_universe=False).refresh(now)
    assert seen["histories"] is not None and len(seen["histories"]) > 10
    assert seen["held"] is not None
