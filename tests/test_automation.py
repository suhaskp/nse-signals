from datetime import datetime

import pandas as pd
import pytest

from background import BackgroundRefresher
from config.config import load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from intelligence_engine import MORNING_COLUMNS, IntelligenceEngine, _morning_status, morning_check


def _engine(tmp_path, now):
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "ranker": {"n_splits": 3, "min_train_dates": 400,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    m = SyntheticMarket(now=now)
    return IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                              auto_universe=False)


@pytest.mark.parametrize("action, gap, price, stop, expected", [
    ("BUY", 1.0, 101, 97, "OK"), ("BUY", 5.0, 105, 97, "SKIP: gapped up"), ("BUY", -4.0, 96, 97, "SKIP: opened below"),
    ("BUY", -3.5, 96.5, 90, "CAUTION"), ("SELL", -5.0, 95, 103, "SKIP: gapped down"),
    ("SELL", 4.0, 104, 103, "SKIP: opened above"), ("SELL", 0.5, 100.5, 103, "OK"),
])
def test_morning_status_rules(action, gap, price, stop, expected):
    assert _morning_status(action, gap, price, stop, 3.0).startswith(expected)


def test_morning_check_timeline(tmp_path):
    early = datetime(2026, 9, 29, 8, 40, tzinfo=IST)
    e = _engine(tmp_path, early)
    rep = e.refresh(early)
    table, note = morning_check(e, rep, early)
    assert table is None and "09:00" in note

    preopen = datetime(2026, 9, 29, 9, 10, tzinfo=IST)
    e2 = _engine(tmp_path / "b", preopen)
    rep2 = e2.refresh(preopen)
    table2, note2 = morning_check(e2, rep2, preopen)
    assert list(table2.columns) == MORNING_COLUMNS and "pre-open IEP (final)" in note2
    acted = rep2.recommendations[rep2.recommendations["Action"].isin(["BUY", "SELL"])]
    assert set(table2["Ticker"]) == set(acted["Ticker"])
    skips = table2[table2["Status"].str.startswith("SKIP")]
    assert (skips["Qty"] == 0).all()
    ok = table2[(table2["Status"] == "OK") & (table2["Action"] == "BUY")]
    assert (ok["Stop Loss"] < ok["Entry"]).all()
    assert (tmp_path / "b" / "o" / "morning_check_2026-09-29.csv").exists()
    assert morning_check(e2, rep2, preopen.replace(minute=11))[0] is table2  # throttled

    after = datetime(2026, 9, 29, 16, 30, tzinfo=IST)
    e3 = _engine(tmp_path / "c", after)
    assert morning_check(e3, e3.refresh(after), after) == (None, "")  # ideas already use today's close


def test_background_refresher_survives_errors():
    calls = []

    def ok(now): calls.append(("ok", now))
    def boom(now): raise RuntimeError("feed down")

    r = BackgroundRefresher([("bad", boom), ("good", ok)], lambda: datetime(2026, 9, 29, 9, 10, tzinfo=IST), 1)
    r.run_once()
    assert calls and r.last_run is not None and r.last_error is None  # later job still ran and cleared the error
    r2 = BackgroundRefresher([("bad", boom)], lambda: datetime.now(IST), 1)
    r2.run_once()
    assert "feed down" in r2.last_error


def test_settings_yaml_is_picked_up_automatically(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text("risk:\n  account_equity: 250000\n")
    monkeypatch.chdir(tmp_path)
    assert load_config().risk.account_equity == 250000


def test_single_instance_lock():
    import serve
    first = serve.acquire_lock(18599)
    assert first is not None
    try:
        assert serve.acquire_lock(18599) is None      # a second dashboard is refused
    finally:
        first.close()
    again = serve.acquire_lock(18599)                  # free again once the first one exits
    assert again is not None
    again.close()


def test_preopen_engine_schedule(tmp_path):
    from data_ingestion.preopen import PreOpenHistory
    from preopen_engine import PreOpenEngine
    cfg = load_config(overrides={
        "tickers": ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "LT"],
        "data": {"provider": "synthetic", "preopen_source": "synthetic",
                 "preopen_history_path": str(tmp_path / "po.csv")},
        "output_dir": str(tmp_path / "o"), "log_dir": str(tmp_path / "l"),
        "model": {"model_dir": str(tmp_path / "m"), "n_splits": 3, "xgb_params": {"n_estimators": 20, "n_jobs": 1}}})
    day = datetime(2026, 9, 29, 9, 10, tzinfo=IST)
    m = SyntheticMarket(now=day)
    m.seed_history(PreOpenHistory(cfg.data.preopen_history_path), list(cfg.tickers))
    eng = PreOpenEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None))
    assert eng.refresh(day.replace(hour=9, minute=0)) is None and "09:08" in eng.status   # too early
    first = eng.refresh(day)                                                             # 09:10 -> runs
    assert first is not None and first.run_time == day
    assert eng.refresh(day.replace(minute=11)) is first                                  # within 2 minutes
    second = eng.refresh(day.replace(minute=12))                                         # re-check
    assert second is not first
    assert eng.refresh(day.replace(hour=10)) is second                                   # final after 09:15
    assert (tmp_path / "o" / "preopen_latest.joblib").exists()
    again = PreOpenEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None))
    assert again.peek() is not None                                                      # survives a restart
    sunday = datetime(2026, 9, 27, 9, 10, tzinfo=IST)
    assert eng.refresh(sunday) is second and "weekend" in eng.status.lower()


def test_background_start_does_not_deadlock(monkeypatch):
    """Starting every background thread must return promptly (regression: lock re-entry froze startup)."""
    import threading
    import engines_core
    monkeypatch.setenv("PIPELINE_BACKGROUND", "1")
    monkeypatch.setattr(engines_core, "_refresher", None)
    started = []
    monkeypatch.setattr(engines_core.BackgroundRefresher, "start", lambda self: started.append(self) or self)
    t = threading.Thread(target=engines_core.start_background, args=(True,), daemon=True)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "start_background deadlocked"
    assert len(started) >= 2  # NSE refresher + crypto refresher


def test_heartbeat_and_readiness_rules():
    from datetime import datetime, timedelta
    import pandas as pd
    from data_ingestion.data_fetcher import IST
    from status import heartbeat, readiness, ready

    class Snap:
        def __init__(self, at):
            self.as_of = at
    now = datetime(2026, 10, 1, 13, 0, tzinfo=IST)
    beats = heartbeat(Snap(now), Snap(now - timedelta(hours=3)), None, now, True)
    assert [b["overdue"] for b in beats] == [False, True, False]          # live signals stale in market hours
    assert not heartbeat(Snap(now), Snap(now - timedelta(hours=3)), None, now.replace(hour=20), True)[1]["overdue"]
    res = pd.DataFrame({"rule_version": ["v1"] * 60, "exit": [1.0] * 60, "r_multiple": [0.3] * 60})
    good = {"stress_verdict": ("ok", ""), "sensitivity_verdict": ("ok", "")}
    assert ready(readiness(res, {"level": "ok"}, good, "v1", 50, 0.1))
    assert not ready(readiness(res, {"level": "ok"}, good, "v2", 50, 0.1))   # trades on another version don't count
    assert not ready(readiness(res.head(10), {"level": "ok"}, good, "v1", 50, 0.1))
    assert len(readiness(res, {"level": "ok"}, None, "v1", 50, 0.1)) == 3     # crypto: no robustness items
