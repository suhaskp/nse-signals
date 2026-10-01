from datetime import datetime

import pytest

from config.config import load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.preopen import PreOpenHistory
from data_ingestion.synthetic import SyntheticMarket
from pipeline import SUMMARY_COLUMNS, DailyPipeline, session_status


def _cfg(tmp_path, **extra):
    return load_config(overrides={
        "tickers": ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "LT", "AXISBANK", "M&M"],
        "data": {"provider": "synthetic", "preopen_source": "synthetic",
                 "preopen_history_path": str(tmp_path / "po.csv")},
        "output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"),
        "model": {"model_dir": str(tmp_path / "m"), "n_splits": 3,
                  "xgb_params": {"n_estimators": 30, "n_jobs": 1}},
        **extra,
    })


@pytest.mark.parametrize("when, trading, phrase", [
    (datetime(2026, 9, 26, 9, 10, tzinfo=IST), False, "weekend"),
    (datetime(2026, 9, 28, 9, 3, tzinfo=IST), True, "not final"),
    (datetime(2026, 9, 28, 9, 10, tzinfo=IST), True, "IEP is final"),
    (datetime(2026, 9, 28, 10, 0, tzinfo=IST), True, "already open"),
])
def test_session_status(tmp_path, when, trading, phrase):
    ok, note = session_status(_cfg(tmp_path), when)
    assert ok is trading and phrase in note


def test_holiday_is_closed(tmp_path):
    cfg = _cfg(tmp_path, market={"holidays": ["2026-09-28"]})
    assert session_status(cfg, datetime(2026, 9, 28, 9, 10, tzinfo=IST))[0] is False


def test_end_to_end_synthetic(tmp_path, now):
    cfg = _cfg(tmp_path)
    market = SyntheticMarket(now=now)
    market.seed_history(PreOpenHistory(cfg.data.preopen_history_path), list(cfg.tickers))
    fetcher = MarketDataFetcher(cfg.data, provider=market, preopen=market, sleeper=lambda s: None)
    result = DailyPipeline(cfg, fetcher, now=now).run(retrain=True, record_paper_orders=True)
    assert list(result.summary.columns) == SUMMARY_COLUMNS
    assert len(result.scan) == 8 and result.cv_report is not None
    assert (tmp_path / "out" / f"watchlist_{now.date()}.csv").exists()
    assert set(result.scan["rvol_source"]) == {"preopen"}
    stored = PreOpenHistory(cfg.data.preopen_history_path).load()
    assert now.date().isoformat() in set(stored["session_date"])
    longs = result.summary[result.summary["Signal"] == "LONG"]
    assert (longs["Stop Loss"] < longs["Current Price"]).all()
