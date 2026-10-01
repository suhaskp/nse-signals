from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.config import RankerConfig, load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from intelligence_engine import IntelligenceEngine
from models.portfolio import BASELINE, Settings, metrics, relabel, simulate, target_weights
from models.ranker import CrossSectionalRanker, build_panel
from monitoring.paper_portfolio import PaperPortfolio
from research.optimizer import optimize
from tests.planted import planted_market

FAST = RankerConfig(n_splits=3, min_train_dates=400, holdout_days=200,
                    params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 20, "n_jobs": 1})


def _day(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"pred": rng.normal(size=n), "vol_21": rng.uniform(0.01, 0.03, n)},
                        index=[f"T{i}" for i in range(n)])


def test_buffer_keeps_holdings_and_regime_goes_to_cash():
    day = _day()
    top = day["pred"].nlargest(10).index
    held_ok = {t: 0.1 for t in day["pred"].rank(pct=True).loc[lambda r: (r >= 0.8) & (r < 0.9)].index}
    w = target_weights(day, held_ok, Settings(top_n=10, buffer=0.8), True)
    assert set(held_ok) <= set(w) and len(w) == 10 and abs(sum(w.values()) - 1) < 1e-9
    assert set(target_weights(day, {}, Settings(top_n=10), True)) == set(top)
    assert target_weights(day, {}, Settings(regime=True), False) == {}
    iv = target_weights(day, {}, Settings(weighting="inv_vol"), True)
    calm = min(iv, key=lambda t: day.at[t, "vol_21"])
    assert iv[calm] == max(iv.values())


def test_turnover_based_costs():
    dates = pd.bdate_range("2025-01-01", periods=40)
    rows = [{"date": d, "ticker": f"T{i}", "pred": float(i), "fwd_ret": 0.01, "idx_fwd": 0.0,
             "idx_above_200": 1.0, "vol_21": 0.02, "turnover_cr": 100} for d in dates for i in range(20)]
    oos = pd.DataFrame(rows)
    sim = simulate(oos, Settings(horizon=10, top_n=5), cost_pct=0.01)
    assert sim["turnover"].iloc[0] == pytest.approx(0.5)       # buying from cash pays only the buy half
    assert (sim["turnover"].iloc[1:] == 0).all()               # same ranks -> nothing traded
    assert sim["ret"].iloc[0] == pytest.approx(0.01 - 0.005)   # half the round-trip cost on entry
    assert sim["ret"].iloc[1] == pytest.approx(0.01)           # no cost when not trading
    m = metrics(sim, 10)
    assert m["periods"] == len(sim) and m["avg_turnover"] < 0.5


def test_relabel_changes_only_labels():
    h, b = planted_market(n_stocks=12, n_days=400)
    p10 = build_panel(h, b, {}, 10)
    p20 = relabel(p10, h, b, 20)
    feats = [c for c in p10.columns if c.startswith("r_")]
    pd.testing.assert_frame_equal(p10.sort_values(["date", "ticker"])[feats].reset_index(drop=True),
                                  p20.sort_values(["date", "ticker"])[feats].reset_index(drop=True))
    x = p20[(p20["ticker"] == "P00")].set_index("date")["fwd_ret"].dropna()
    c, o = h["P00"]["Close"], h["P00"]["Open"]
    d = x.index[100]
    assert x[d] == pytest.approx(c.shift(-20)[d] / o.shift(-1)[d] - 1, rel=1e-5)


def test_optimizer_uses_holdout_and_reports_index():
    h, b = planted_market(n_stocks=40, n_days=1300)
    opt = optimize(build_panel(h, b, {}, 10), h, b, FAST)
    assert opt["holdout_start"] is not None and isinstance(opt["chosen"], Settings)
    assert len(opt["table"]) == 32 and set(opt["reports"]) == {10, 20}
    assert opt["index_note"].startswith("On the unseen final year")
    if not opt["adopted"]:
        assert opt["chosen"] == BASELINE


def test_paper_portfolio_fills_next_open_and_rebalances(tmp_path):
    h, b = planted_market(n_stocks=15, n_days=300)
    dates = h["P00"].index[-12:]
    pp = PaperPortfolio(tmp_path / "pp.json", 1_000_000, 0.0025)
    s = Settings(horizon=5, top_n=5)
    for d in dates:
        scores = pd.DataFrame({"pred": [float(df.loc[:d, "Close"].pct_change(20).iloc[-1]) for df in h.values()],
                               "vol_21": 0.02}, index=list(h))
        pp.update(d.date(), h, b, scores, s, True)
    st = pp.state
    first_buys = [t for t in st["trades"] if t["side"] == "BUY"][:5]
    assert all(t["date"] == str(dates[1].date()) for t in first_buys)           # filled at the next session
    assert all(t["price"] == pytest.approx(h[t["ticker"]].at[dates[1], "Open"], abs=0.01) for t in first_buys)
    assert len(st["history"]) == len(dates) and st["cash"] >= 0
    assert any(t["side"] == "SELL" for t in st["trades"]) or len(st["holdings"]) == 5
    pp.update(dates[-1].date(), h, b, scores, s, True)                          # idempotent
    assert len(pp.state["history"]) == len(dates)
    reloaded = PaperPortfolio(tmp_path / "pp.json", 1, 0)
    assert reloaded.state["capital"] == 1_000_000                                # persisted


def test_engine_reports_plan_and_paper(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "ranker": {"n_splits": 3, "min_train_dates": 400, "holdout_days": 200,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    m = SyntheticMarket(now=now)
    rep = IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                             auto_universe=False).refresh(now)
    assert isinstance(rep.settings, Settings) and rep.optimizer is not None
    assert rep.paper is not None and rep.paper["pending"] is not None
    assert (tmp_path / "o" / "paper_portfolio.json").exists()
    assert list((tmp_path / "c" / "intelligence").glob("optimizer_*.joblib"))
