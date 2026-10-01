"""End-to-end: every engine driven through several simulated trading days on one consistent market.

Slow (a few minutes), so it runs only when asked:   RUN_E2E=1 pytest tests/test_e2e_week.py
It checks that nothing raises at any time of day and that every record stays consistent.
"""
import json
import os
from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_E2E") != "1", reason="slow end-to-end run; set RUN_E2E=1")


def test_week_of_trading(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_HOME", str(tmp_path / "home"))
    from config import load_config
    from data_ingestion import MarketDataFetcher
    from data_ingestion.data_fetcher import IST
    from data_ingestion.preopen import PreOpenHistory
    from data_ingestion.synthetic import SyntheticMarket
    from intelligence_engine import IntelligenceEngine, morning_check
    from live_engine import LiveSignalEngine
    from preopen_engine import PreOpenEngine
    from storage import rule_version

    cfg = load_config(overrides={
        "tickers": [f"S{i:02d}" for i in range(60)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "preopen_scan_enabled": True,
        "buy_quality": {"regime_required": False, "volume_multiple": 1.0, "min_close_location": 0.3},
        "ranker": {"n_splits": 3, "min_train_dates": 400,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 2, "min_child_weight": 20}},
        "model": {"n_splits": 3, "xgb_params": {"n_estimators": 30, "n_jobs": 2}}})
    market = SyntheticMarket(now=datetime(2026, 10, 9, 16, 30, tzinfo=IST))
    fetch = MarketDataFetcher(cfg.data, provider=market, preopen=market, sleeper=lambda s: None)
    market.seed_history(PreOpenHistory(cfg.data.preopen_history_path), list(cfg.tickers))
    live = LiveSignalEngine(cfg, fetch, auto_universe=False)
    intel = IntelligenceEngine(cfg, fetch, auto_universe=False, live=live)
    pre = PreOpenEngine(cfg, fetch)
    v0 = rule_version(cfg)
    first_journal = None
    for d in ("2026-10-02", "2026-10-03", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"):
        for t in ("09:10", "10:30", "16:30"):
            now = datetime.fromisoformat(f"{d}T{t}").replace(tzinfo=IST)
            market.now = now
            live.refresh(now, force=True)
            rep = intel.refresh(now, force=True)
            morning_check(intel, rep, now)
            pre.refresh(now)
            jp = cfg.output_dir / "decision_journal.csv"
            if first_journal is None and jp.exists():
                first_journal = pd.read_csv(jp)
            # never two positions in one stock: an issued BUY must not already be held
            held = {h[0] for h in getattr(intel, "_held", [])}
            issued = rep.recommendations[(rep.recommendations["Action"] == "BUY") & (rep.recommendations["Qty"] > 0)]
            assert not (set(issued["Ticker"]) & held), (now, set(issued["Ticker"]) & held)

    out = cfg.output_dir
    alerts = json.loads((out / "alerts.json").read_text())
    assert len({a["key"] for a in alerts}) == len(alerts)
    fwd = pd.read_csv(out / "forward_signals.csv")
    assert not fwd.duplicated(["session", "Ticker"]).any()
    journal = pd.read_csv(out / "decision_journal.csv")
    assert not journal.duplicated(["session", "Ticker"]).any()
    assert journal.head(len(first_journal))[first_journal.columns].equals(first_journal)  # append-only
    history = [h["date"] for h in json.loads((out / "paper_portfolio.json").read_text())["history"]]
    assert history == sorted(set(history))
    assert rule_version(cfg) == v0 and (cfg.data_home / "rules_freeze.json").exists()
