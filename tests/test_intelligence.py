from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.config import RankerConfig, load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from intelligence_engine import REC_COLUMNS, IntelligenceEngine
from models.ranker import FEATURES, CrossSectionalRanker, build_panel, explain
from monitoring.strategy_lab import lab_summary, run_lab
from research.market import market_breadth, market_regime, sector_strength, stock_profile
from tests.planted import planted_market

FAST = RankerConfig(n_splits=3, min_train_dates=400, params={"n_estimators": 60, "max_depth": 3,
                    "learning_rate": 0.1, "min_child_weight": 20, "n_jobs": 1})


def test_research_snapshots():
    h, bench = planted_market(n_stocks=12, n_days=400)
    reg = market_regime(bench)
    assert reg["state"] in {"bull", "bear", "neutral"} and isinstance(reg["risk_on"], bool)
    br = market_breadth(h)
    assert br["n"] == 12 and 0 <= br["pct_above_200"] <= 1 and br["advancers"] + br["decliners"] <= 12
    sect = sector_strength(h, {s: ("A" if i % 2 else "B") for i, s in enumerate(h)})
    assert list(sect["Rank"]) == [1, 2] and sect["Stocks"].sum() == 12
    prof = stock_profile(next(iter(h.values())), bench)
    assert -1 < prof["max_dd_1y"] <= 0 and "vs_index_3m" in prof


def test_panel_features_have_no_lookahead():
    h, bench = planted_market(n_stocks=15, n_days=500)
    base = build_panel(h, bench, {}, 10)
    cut = h["P00"].index[400]
    tampered = {s: df.copy() for s, df in h.items()}
    for df in tampered.values():
        df.loc[df.index > cut, ["Open", "High", "Low", "Close"]] *= 1.5
    after = build_panel(tampered, bench, {}, 10)
    b = base[base["date"] <= cut].set_index(["date", "ticker"])[FEATURES]
    a = after[after["date"] <= cut].set_index(["date", "ticker"])[FEATURES]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())
    # labels, by contrast, must look ahead
    lab_b = base[base["date"] == base["date"][base["date"] < cut].max()]["fwd_ret"].to_numpy()
    lab_a = after[after["date"] == after["date"][after["date"] < cut].max()]["fwd_ret"].to_numpy()
    assert not np.allclose(lab_a, lab_b, equal_nan=True)


def test_validation_gate_accepts_real_signal_and_rejects_noise():
    h, bench = planted_market(n_stocks=50, n_days=1300)
    rep = CrossSectionalRanker(FAST).walk_forward(build_panel(h, bench, {}, 10))
    assert rep["validated"] and rep["mean_ic"] > 0.02
    cal = rep["calibration"]
    assert cal.loc[cal["decile"] == 10, "avg_excess_pct"].item() > cal.loc[cal["decile"] == 1, "avg_excess_pct"].item()

    m = SyntheticMarket(now=datetime(2026, 9, 28, 16, 30, tzinfo=IST))
    noise = {f"N{i:02d}": m.daily(f"N{i:02d}") for i in range(50)}
    rep2 = CrossSectionalRanker(FAST).walk_forward(build_panel(noise, m.daily("^NSEI"), {}, 10))
    assert not rep2["validated"]


def test_explanations_are_readable():
    h, bench = planted_market(n_stocks=20, n_days=600)
    panel = build_panel(h, bench, {}, 10)
    r = CrossSectionalRanker(FAST).fit(panel)
    last = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    contrib = r.contributions(last)
    assert contrib is not None and list(contrib.columns) == FEATURES
    reasons = explain(last.iloc[0], contrib.iloc[0], +1)
    assert 1 <= len(reasons) <= 3 and all(isinstance(x, str) and x for x in reasons)
    assert all("top 0%" not in x for x in reasons)


def test_strategy_lab_variants():
    m = SyntheticMarket(now=datetime(2026, 9, 28, 16, 30, tzinfo=IST))
    h = {f"L{i:02d}": m.daily(f"L{i:02d}") for i in range(30)}
    trades = run_lab(h, m.daily("^NSEI"), load_config().breakout)
    assert set(trades["variant"]) <= {"original", "regime", "trailing", "no_rsi_cap", "sell_original"}
    summ = lab_summary(trades)
    assert set(summ["Verdict"]) <= {"Passes", "Marginal", "Fails", "Too few trades"}


def test_engine_end_to_end_and_cache(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "ranker": {"n_splits": 3, "min_train_dates": 400,
                   "params": {"n_estimators": 60, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})

    def engine():
        m = SyntheticMarket(now=now)
        return IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                                  auto_universe=False)

    rep = engine().refresh(now)
    assert rep.session == now.date() and rep.regime is not None and rep.validation is not None
    assert list(rep.recommendations.columns) == REC_COLUMNS
    assert set(rep.recommendations["Action"]) <= {"BUY", "SELL", "WATCH"}
    if not rep.validated:
        acted = rep.recommendations[rep.recommendations["Action"] != "WATCH"]
        assert set(acted["Conviction"]) <= {"Unvalidated"}
    assert rep.quality is not None and "passes" in rep.quality and rep.quality_funnel is not None
    buys = rep.recommendations[rep.recommendations["Action"] == "BUY"]
    assert (buys["Stop Loss"] < buys["Price"]).all() and (buys["Target Price"] > buys["Price"]).all()
    assert (tmp_path / "o" / f"intelligence_{rep.session}.csv").exists()
    assert list((tmp_path / "c" / "intelligence").glob("*.joblib"))
    rep2 = engine().refresh(now)  # new engine instance: heavy results come from the disk cache
    pd.testing.assert_frame_equal(rep.recommendations, rep2.recommendations)


def test_panel_features_are_numeric_even_from_object_columns():
    """Older pandas turns boolean-with-NaN into object dtype; the model must still get floats."""
    h, bench = planted_market(n_stocks=12, n_days=400)
    panel = build_panel(h, bench, {}, 10)
    assert all(str(panel[f].dtype) == "float32" for f in FEATURES)
    from models.ranker import _X
    broken = panel.head(50).copy()
    broken["breadth_200"] = broken["breadth_200"].astype(object)
    assert all(str(t) == "float32" for t in _X(broken).dtypes)


def test_failed_model_run_is_not_cached(tmp_path, monkeypatch):
    import intelligence_engine as ie
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    cfg = load_config(overrides={"tickers": [f"F{i:02d}" for i in range(12)],
                                 "data": {"provider": "synthetic", "preopen_source": "synthetic"},
                                 "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"),
                                 "superstar_dir": str(tmp_path / "s")})
    monkeypatch.setattr(ie, "build_panel", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    m = SyntheticMarket(now=now)
    rep = IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                             auto_universe=False).refresh(now)
    assert rep.error == "boom"
    session_caches = [f for f in (tmp_path / "c" / "intelligence").glob("*.joblib")
                      if not f.name.startswith(("robustness_", "optimizer_"))]
    assert not session_caches  # the failed per-session model result is not cached


def test_buy_ideas_respect_portfolio_limits():
    from intelligence_engine import apply_portfolio_limits
    from risk_engine.risk_manager import RiskManager
    cfg = load_config(overrides={"risk": {"max_open_positions": 3, "max_portfolio_heat_pct": 0.025}})
    rm = RiskManager(cfg.risk)
    rows, plans = [], {}
    for i in range(6):
        tk = f"B{i}"
        plans[("BUY", tk)] = rm.plan_from_levels(tk, 500.0, 470.0, 590.0, 20.0)
        rows.append({"Action": "BUY", "Ticker": tk, "Qty": 0, "Risk (INR)": 0.0})
    rows.append({"Action": "SELL", "Ticker": "S0", "Qty": 10, "Risk (INR)": 100.0})
    rows.append({"Action": "WATCH", "Ticker": "W0", "Qty": 0, "Risk (INR)": 0.0})
    recs = apply_portfolio_limits(pd.DataFrame(rows), plans, rm)
    buys = recs[recs["Action"] == "BUY"]
    taken = buys[buys["Qty"] > 0]
    assert len(taken) <= 3 and taken["Risk (INR)"].sum() <= cfg.risk.account_equity * 0.025 + 1
    assert (buys.loc[buys["Qty"] == 0, "Allocation"].str.startswith("Watch only")).all()
    assert recs.loc[recs["Action"] == "SELL", "Allocation"].item() == "Exit / avoid if held"
    assert recs.loc[recs["Action"] == "WATCH", "Allocation"].item() == "Watch only"


def test_reasons_are_stock_specific():
    from models.ranker import MARKET, LABELS
    h, bench = planted_market(n_stocks=20, n_days=600)
    panel = build_panel(h, bench, {}, 10)
    r = CrossSectionalRanker(FAST).fit(panel)
    last = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    contrib = r.contributions(last)
    contrib.loc[:, "idx_ret_21"] = 99.0  # make a market feature dominate
    market_labels = {LABELS[f] for f in MARKET}
    for tk in last.index[:5]:
        reasons = explain(last.loc[tk], contrib.loc[tk], +1)
        assert not any(x.split(":")[0] in market_labels or x.startswith("Nifty") for x in reasons)


def test_watch_list_hides_reward_risk_without_breakout(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    m = SyntheticMarket(now=now)
    rep = IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                             auto_universe=False).refresh(now)
    watch = rep.recommendations[rep.recommendations["Action"] == "WATCH"]
    q = rep.quality.set_index("Ticker")
    for _, r in watch.iterrows():
        if r["Ticker"] in q.index and not q.at[r["Ticker"], "g_breakout"]:
            assert pd.isna(r["R:R"])


def test_trend_leaders_and_regime_evidence(tmp_path):
    from research.market import trend_leaders
    from tests.test_buy_quality import bench, textbook
    good = textbook()
    weak = textbook()
    weak["Close"] = weak["Close"].iloc[::-1].to_numpy()  # a downtrend
    weak["Open"], weak["High"], weak["Low"] = weak["Close"] * 0.998, weak["Close"] * 1.004, weak["Close"] * 0.994
    lead = trend_leaders({"GOOD": good, "WEAK": weak}, bench(), min_turnover_cr=0)
    assert list(lead["Ticker"]) == ["GOOD"] and lead.at[0, "vs Nifty 3M %"] > 0

    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    m = SyntheticMarket(now=now)
    rep = IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                             auto_universe=False).refresh(now)
    assert rep.leaders is not None
    variants = set(rep.lab_trades["variant"]) if rep.lab_trades is not None else set()
    if "quality" in variants:
        assert "quality_no_regime" in variants
        assert (rep.lab_trades["variant"] == "quality_no_regime").sum() >= (rep.lab_trades["variant"] == "quality").sum()


def test_todays_movers():
    from live_engine import todays_movers
    idx = pd.bdate_range(end="2026-09-28", periods=30)
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=idx)
    df.iloc[-1] = [100.0, 106.0, 100.0, 105.5, 8e5]            # +5.5% on 80% of a full day's volume by ~11:00
    quiet = df.copy()
    quiet.iloc[-1] = [100.0, 106.0, 100.0, 105.5, 1e5]         # same move on thin volume
    now = datetime(2026, 9, 28, 11, 0, tzinfo=IST)
    mv = todays_movers({"HOT": df, "THIN": quiet}, now, min_turnover_cr=0)
    assert list(mv["Ticker"]) == ["HOT"] and mv.at[0, "Today %"] == pytest.approx(5.5)
