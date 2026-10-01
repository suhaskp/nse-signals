import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from alerts import AlertCenter
from config.config import BuyQualityConfig, load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from intelligence_engine import IntelligenceEngine
from monitoring.forward_record import ForwardRecord, summarize
from storage import backup_records, freeze_status, migrate_legacy, pin_freeze_settings, rule_version
from tests.test_buy_quality import textbook


def test_paths_live_outside_the_app_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_HOME", str(tmp_path / "home"))
    cfg = load_config()
    for p in (cfg.output_dir, cfg.cache_dir, cfg.log_dir, cfg.superstar_dir, cfg.model.model_dir,
              cfg.data.preopen_history_path):
        assert str(p).startswith(str(tmp_path / "home"))
    explicit = load_config(overrides={"output_dir": str(tmp_path / "x")})
    assert explicit.output_dir == tmp_path / "x"                       # absolute paths are respected


def test_migration_and_backup(tmp_path):
    app, home = tmp_path / "app", tmp_path / "home"
    (app / "outputs").mkdir(parents=True)
    (app / "outputs" / "forward_signals.csv").write_text("session,Ticker\n2026-09-29,X\n")
    assert migrate_legacy(app, home) == ["outputs"]
    assert (home / "outputs" / "forward_signals.csv").exists()
    assert migrate_legacy(app, home) == []                             # never overwrites
    z = backup_records(home, today=date(2026, 9, 30))
    assert z is not None and z.exists() and backup_records(home, today=date(2026, 9, 30)) is None
    for d in range(1, 20):
        backup_records(home, keep=5, today=date(2026, 10, d))
    assert len(list((home / "backups").glob("*.zip"))) == 5


def test_rule_version_and_freeze(tmp_path):
    cfg = load_config()
    v = rule_version(cfg)
    assert v == rule_version(load_config()) and len(v) == 8
    changed = load_config(overrides={"buy_quality": {"volume_multiple": 2.0}})
    assert rule_version(changed) != v
    fr = freeze_status(cfg, tmp_path, today=date(2026, 9, 30))
    assert fr["active"] and not fr["drifted"] and fr["until"] == "2026-12-13"  # 75 days including 30 Sep
    assert freeze_status(changed, tmp_path, today=date(2026, 10, 5))["drifted"]
    assert not freeze_status(cfg, tmp_path, today=date(2027, 1, 1))["active"]
    pin_freeze_settings(tmp_path, {"horizon": 10, "top_n": 10})
    pin_freeze_settings(tmp_path, {"horizon": 20, "top_n": 20})          # first pin wins
    assert json.loads((tmp_path / "rules_freeze.json").read_text())["settings"]["horizon"] == 10


def test_alerts_dedupe(tmp_path):
    c = AlertCenter(tmp_path / "alerts.json")
    assert c.add("buy:1:X", "buy", "BUY X") and not c.add("buy:1:X", "buy", "BUY X again")
    c.add("skip:1:Y", "skip", "Skip Y")
    assert [a["key"] for a in c.recent()] == ["skip:1:Y", "buy:1:X"] and c.recent()[0]["icon"] == "🔴"


def test_forward_record_follows_signals(tmp_path):
    df = textbook()
    after = pd.DataFrame({"Open": [df["Close"].iloc[-1] * 1.002] * 3, "High": [df["Close"].iloc[-1] * 1.08] * 3,
                          "Low": [df["Close"].iloc[-1]] * 3, "Close": [df["Close"].iloc[-1] * 1.05] * 3,
                          "Volume": [1e6] * 3}, index=pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=3))
    full = pd.concat([df, after])
    from screeners.buy_quality import quality_frame
    from tests.test_buy_quality import bench
    r = quality_frame(df, BuyQualityConfig(), bench()).iloc[-1]
    buys = pd.DataFrame([{"Ticker": "X", "Price": r["close"], "Breakout level": r["level"], "ATR": r["atr"],
                          "Stop Loss": r["stop"], "Target Price": r["target"], "R:R": r["rr"]},
                         {"Ticker": "GAP", "Price": r["close"], "Breakout level": r["level"], "ATR": r["atr"],
                          "Stop Loss": r["stop"], "Target Price": r["target"], "R:R": r["rr"]}])
    fwd = ForwardRecord(tmp_path / "f.csv")
    assert fwd.record(buys, df.index[-1].date(), "v1") == 2 and fwd.record(buys, df.index[-1].date(), "v1") == 0
    gap = full.copy()
    gap.loc[after.index[0], "Open"] = r["close"] * 1.05                   # opens 5% above: must not be entered
    res = fwd.evaluate({"X": full, "GAP": gap}, BuyQualityConfig(), 0.0025).set_index("Ticker")
    assert res.at["X", "outcome"] == "target" and res.at["X", "r_multiple"] > 1.9
    assert res.at["GAP", "outcome"].startswith("not entered")
    s = summarize(res.reset_index())
    assert s.iloc[0]["Signals"] == 2 and s.iloc[0]["Entered"] == 1 and s.iloc[0]["Closed"] == 1


def test_engine_records_signals_and_raises_alerts(tmp_path):
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    cfg = load_config(overrides={
        "tickers": [f"E{i:02d}" for i in range(40)], "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "cache_dir": str(tmp_path / "c"), "output_dir": str(tmp_path / "o"), "superstar_dir": str(tmp_path / "s"),
        "data_home": str(tmp_path / "home"),
        "buy_quality": {"regime_required": False, "volume_multiple": 1.0, "min_close_location": 0.0},
        "ranker": {"n_splits": 3, "min_train_dates": 400, "optimize": False,
                   "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "min_child_weight": 20}}})
    m = SyntheticMarket(now=now)
    rep = IntelligenceEngine(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                             auto_universe=False).refresh(now)
    issued = rep.recommendations[(rep.recommendations["Action"] == "BUY") & (rep.recommendations["Qty"] > 0)]
    keys = [a["key"] for a in AlertCenter(tmp_path / "o" / "alerts.json").recent(100)]
    if len(issued):
        assert (tmp_path / "o" / "forward_signals.csv").exists()
        assert any(k.startswith(f"buy:{rep.session}:") for k in keys)
        log = pd.read_csv(tmp_path / "o" / "forward_signals.csv")
        assert set(log["rule_version"]) == {rule_version(cfg)}
    assert rep.tracking["freeze"]["active"]
