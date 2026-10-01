from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.config import BreakoutConfig, load_config
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket
from data_ingestion.universe import load_symbols
from monitoring.breakout_backtest import _simulate, backtest_breakouts, summarize
from pipeline_eod import SIGNAL_COLUMNS, EodBreakoutPipeline
from screeners.breakout import breakout_frame, scan_breakouts

CFG = BreakoutConfig(lookback=30, min_avg_turnover_cr=0.0)


def _range_bound(n=80, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.sin(np.arange(n) / 3) * 3 + rng.normal(0, 0.3, n)
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": np.full(n, 1e6)}, index=idx)


def test_buy_needs_all_three_conditions():
    df = _range_bound()
    i = df.index[-1]
    df.loc[i, ["Close", "High", "Volume"]] = [df["High"].iloc[:-1].max() * 1.02, df["High"].iloc[:-1].max() * 1.03, 3e6]
    loose = BreakoutConfig(lookback=30, min_avg_turnover_cr=0, bull_rsi_max=90)
    f = breakout_frame(df, loose).iloc[-1]
    assert f["buy"] and 70 < f["rsi"] < 90
    assert not breakout_frame(df, CFG)["buy"].iloc[-1]          # default RSI < 70 rejects this jump
    weak_volume = df.copy(); weak_volume.loc[i, "Volume"] = 1.5e6
    assert not breakout_frame(weak_volume, loose)["buy"].iloc[-1]  # 1.5x volume < 2x requirement


def test_sell_rule_and_volume_variants():
    df = _range_bound()
    i = df.index[-1]
    low = df["Low"].iloc[:-1].min()
    df.loc[i, ["Close", "Low", "Volume"]] = [low * 0.99, low * 0.985, 0.8e6]
    assert breakout_frame(df, BreakoutConfig(lookback=30, min_avg_turnover_cr=0, bear_rsi_min=5))["sell"].iloc[-1]
    hi_vol_rule = BreakoutConfig(lookback=30, min_avg_turnover_cr=0, bear_rsi_min=5, bear_volume_rule="above_multiple")
    assert not breakout_frame(df, hi_vol_rule)["sell"].iloc[-1]  # low volume fails the high-volume variant


def test_prior_range_excludes_today():
    df = _range_bound()
    f = breakout_frame(df, CFG)
    assert f["prior_high"].iloc[-1] == pytest.approx(df["High"].iloc[-31:-1].max())


def test_simulate_stop_target_and_gap():
    idx = pd.bdate_range("2026-01-01", periods=6)
    df = pd.DataFrame({"Open": [100, 100, 101, 104, 90, 90], "High": [101, 101, 107, 105, 91, 91],
                       "Low": [99, 99, 100, 103, 89, 89], "Close": [100, 100, 106, 104, 90, 90],
                       "Volume": [1] * 6}, index=idx)
    flags = pd.Series([True, False, False, False, False, False], index=idx)
    atr_s = pd.Series(2.0, index=idx)
    (tr,) = _simulate(df, flags, atr_s, "long", 1.5, 2.0, 10, 0.0)
    assert tr["outcome"] == "target" and tr["r_multiple"] == pytest.approx(2.0)  # 100 + 2*3 = 106 hit day 3
    flags2 = pd.Series([False, False, True, False, False, False], index=idx)
    (tr2,) = _simulate(df, flags2, atr_s, "long", 1.5, 2.0, 10, 0.0)
    assert tr2["outcome"] == "gap_stop" and tr2["exit"] == 90  # entry 104, stop 101, opens at 90
    (tr3,) = _simulate(df, flags, atr_s, "long", 1.5, 2.0, 10, 0.01)
    assert tr3["r_multiple"] < 2.0  # costs deducted


def test_backtest_and_summary_shapes():
    m = SyntheticMarket(now=datetime(2026, 9, 28, 16, 30, tzinfo=IST))
    hist = {t: m.daily(t) for t in ("RELIANCE", "TCS", "INFY", "SBIN", "ITC", "LT")}
    trades = backtest_breakouts(hist, BreakoutConfig())
    stats = summarize(trades)
    if not trades.empty:
        assert set(trades["side"]) <= {"long", "short"} and (trades["hold_days"] >= 1).all()
        assert {"Rule", "Trades", "Avg R", "Win rate %"} <= set(stats.columns)


def test_load_symbols_skips_rows_without_nse_code(tmp_path):
    p = tmp_path / "ss.csv"
    pd.DataFrame({"Stock": ["Biocon", "Afcom", "M&M"], "NSE Code": ["BIOCON", None, "m&m"]}).to_csv(p, index=False)
    symbols, skipped = load_symbols(p)
    assert symbols == ["BIOCON", "M&M"] and skipped == ["Afcom"]
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_symbols(bad)


def test_eod_pipeline_end_to_end(tmp_path):
    ss = tmp_path / "ss.csv"
    pd.DataFrame({"Stock": ["Arvind", "NoCode"], "NSE Code": ["ARVINDFASN", None]}).to_csv(ss, index=False)
    cfg = load_config(overrides={
        "tickers": ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "LT", "KOTAKBANK", "BAJFINANCE", "NTPC"],
        "data": {"provider": "synthetic", "preopen_source": "synthetic"},
        "output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "logs"), "superstar_file": str(ss),
    })
    now = datetime(2026, 9, 28, 16, 30, tzinfo=IST)
    m = SyntheticMarket(now=now)
    res = EodBreakoutPipeline(cfg, MarketDataFetcher(cfg.data, provider=m, preopen=m, sleeper=lambda s: None),
                              now=now).run()
    assert list(res.signals.columns) == SIGNAL_COLUMNS and res.skipped == ["NoCode"]
    assert "ARVINDFASN" in res.histories and len(res.scan) == 10
    assert (tmp_path / "out" / f"eod_signals_{res.session}.csv").exists()
    sells = res.signals[res.signals["Signal"] == "SELL"]
    assert (sells["Stop Loss"] > sells["Close"]).all() and (sells["Target Price"] < sells["Close"]).all()
    buys = res.signals[res.signals["Signal"] == "BUY"]
    assert (buys["Stop Loss"] < buys["Close"]).all()
    assert set(res.signals.loc[res.signals["Ticker"] == "ARVINDFASN", "Superstar Buy"]) <= {"Yes"}


def test_eod_cutoff_excludes_today_before_ready_time(tmp_path):
    cfg = load_config(overrides={"data": {"provider": "synthetic", "preopen_source": "synthetic"}})
    early = EodBreakoutPipeline(cfg, MarketDataFetcher(cfg.data, provider=SyntheticMarket(), preopen=SyntheticMarket()),
                                now=datetime(2026, 9, 28, 15, 0, tzinfo=IST))
    late = EodBreakoutPipeline(cfg, MarketDataFetcher(cfg.data, provider=SyntheticMarket(), preopen=SyntheticMarket()),
                               now=datetime(2026, 9, 28, 16, 30, tzinfo=IST))
    assert early._cutoff().isoformat() == "2026-09-28" and late._cutoff().isoformat() == "2026-09-29"
