from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from config.config import BuyQualityConfig, load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from intelligence_engine import IntelligenceEngine
from research.evidence import label, strength, wilson
from research.robustness import deflated_sharpe, execution_stress, health, sensitivity_verdict, stress_verdict


def test_evidence_labels():
    assert strength(20) == "LOW" and strength(50) == "PRELIMINARY" and strength(100) == "MEDIUM" and strength(5000) == "HIGH"
    lo, hi = wilson(0.52, 2400)
    assert 0.49 < lo < 0.52 < hi < 0.55
    assert "LOW" in label(52, 20) and "HIGH" in label(52, 2400)


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.004, 0.03, 200))
    trials = list(rng.normal(0.05, 0.05, 32))
    few, many = deflated_sharpe(r, trials, 10), deflated_sharpe(r, trials, 5000)
    assert 0 <= many["dsr"] < few["dsr"] <= 1


def _trades(avg, n=200, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(avg, 1.0, n)
    return pd.DataFrame({"r_multiple": r, "exit_date": pd.bdate_range("2023-01-02", periods=n),
                         "entry": 100.0, "stop": 95.0, "exit": 100.0})


def test_stress_and_health_verdicts():
    good = _trades(0.5)
    tbl = execution_stress(good, _trades(0.45, seed=1), 0.0025)
    assert list(tbl["Scenario"])[0].startswith("Base") and tbl["Avg R"].iloc[2] < tbl["Avg R"].iloc[0]
    assert stress_verdict(tbl)[0] == "ok"
    assert health(pd.DataFrame(), good)["level"] == "none"                          # not enough live trades
    assert health(_trades(0.5, n=60, seed=3), good)["level"] in ("ok", "caution")
    assert health(_trades(-0.6, n=60, seed=4), good)["level"] == "bad"               # clearly worse than backtest


def test_sensitivity_verdict_flags_isolated_peak():
    rows = [{"Parameter": "Breakout lookback (days)", "Value": v, "Current": v == 55, "Trades": 100, "Avg R": r}
            for v, r in ((45, -0.1), (50, -0.05), (55, 0.4), (60, -0.05), (65, -0.1))]
    assert sensitivity_verdict(pd.DataFrame(rows))[0] == "bad"
    stable = [{**x, "Avg R": 0.2 + 0.01 * i} for i, x in enumerate(rows)]
    assert sensitivity_verdict(pd.DataFrame(stable))[0] == "ok"


def _engine(tmp_path, now):
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "data_home": str(tmp_path / "home"),
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    m = SyntheticMarket(now=now)
    return IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                              auto_universe=False)


def test_changes_journal_and_research_on_the_report(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    eng = _engine(tmp_path, now)
    q = pd.DataFrame({"Ticker": ["A", "B"], "Readiness": [60.0, 90.0], "State": ["WAIT", "WATCH CLOSELY"],
                      "Failed": ["breakout, volume", "market regime"], "g_breakout": [False, True]})
    eng._watch_history(q, date(2026, 9, 24))
    q2 = q.assign(Readiness=[92.0, 40.0], State=["WATCH CLOSELY", "IGNORE"], Failed=["market regime", "trend"])
    eng._watch_history(q2, date(2026, 9, 25))
    ch = eng._changes(date(2026, 9, 25)).set_index("Ticker")
    assert ch.at["A", "Change"] == "↑" and "now passes breakout" in ch.at["A", "Why"]
    assert ch.at["B", "Change"] == "↓" and "now fails trend" in ch.at["B", "Why"]

    rep = eng.refresh(now)
    assert rep.research is not None and "health" in rep.research and rep.research["n_trials"] > 0
    assert rep.journal is not None
    first = pd.read_csv(tmp_path / "o" / "decision_journal.csv") if (tmp_path / "o" / "decision_journal.csv").exists() else None
    eng2 = _engine(tmp_path, now)
    eng2.refresh(now)                                                                 # same session again
    if first is not None:
        again = pd.read_csv(tmp_path / "o" / "decision_journal.csv")
        assert len(again) == len(first)                                              # never rewritten or duplicated
        assert {"Blockers", "Becomes BUY if", "Becomes AVOID if", "Rule version"} <= set(again.columns)


def test_weekly_robustness_runs_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_ROBUSTNESS", "1")
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    rep = _engine(tmp_path, now).refresh(now)
    rs = rep.research
    assert {"sensitivity", "stress", "sensitivity_verdict", "stress_verdict"} <= set(rs)
    assert set(rs["sensitivity"]["Parameter"]) == {"Breakout lookback (days)", "Volume requirement (× average)"}
    assert list(rs["stress"]["Scenario"])[-1] == "Entry one session late"
    assert list((tmp_path / "c" / "intelligence").glob("robustness_*.joblib"))
