import numpy as np
import pandas as pd
import pytest

from data_ingestion.preopen import PreOpenQuote
from screeners.screener import (add_indicators, atr, evaluate_ticker, gap_pct, get_daily_candidates,
                                relative_volume, rsi, sma)


def test_sma_matches_manual():
    s = pd.Series(np.arange(1, 11, dtype=float))
    assert sma(s, 3).iloc[-1] == pytest.approx(9.0) and sma(s, 3).isna().sum() == 2


def test_rsi_bounds_and_edge_cases(ohlcv):
    r = rsi(ohlcv["Close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert rsi(pd.Series(np.arange(1, 40, dtype=float)), 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(pd.Series(np.full(40, 5.0)), 14).iloc[-1] == pytest.approx(50.0)


def test_atr_constant_range():
    n = 50
    close = pd.Series(np.full(n, 100.0))
    assert atr(close + 1, close - 1, close, 14).iloc[-1] == pytest.approx(2.0)


def test_turnover_filter(ohlcv, ind_cfg, scr_cfg):
    thin = ohlcv.assign(Volume=ohlcv["Volume"] / 1000)
    last = thin["Close"].iloc[-1]
    res = evaluate_ticker("THIN", thin, _snap("THIN", last * 1.03, last), ind_cfg, scr_cfg)
    assert any("turnover" in f for f in res.failures)


def test_rvol_excludes_current_bar():
    vol = pd.Series([100.0] * 20 + [300.0])
    assert relative_volume(vol, 20).iloc[-1] == pytest.approx(3.0)


def test_gap_pct():
    assert gap_pct(103.0, 100.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        gap_pct(1.0, 0.0)


def _snap(ticker, price, prev_close, qty=300_000, avg=100_000):
    return PreOpenQuote(ticker, price, prev_close, qty, pd.Timestamp.now(), "test", avg)


def test_evaluate_and_candidates(ohlcv, ind_cfg, scr_cfg):
    uptrend = ohlcv.copy()
    uptrend["Close"] = uptrend["Close"] * np.linspace(1, 1.6, len(uptrend))
    uptrend[["Open", "High", "Low"]] = uptrend[["Open", "High", "Low"]].mul(np.linspace(1, 1.6, len(uptrend)), axis=0)
    uptrend["High"] = uptrend[["Open", "High", "Close"]].max(axis=1)
    uptrend["Low"] = uptrend[["Open", "Low", "Close"]].min(axis=1)
    last = uptrend["Close"].iloc[-1]
    ind = add_indicators(uptrend, ind_cfg).iloc[-1]
    res = evaluate_ticker("UP", uptrend, _snap("UP", last * 1.04, last), ind_cfg, scr_cfg)
    rsi_ok = scr_cfg.rsi_min <= ind["RSI"] <= scr_cfg.rsi_max
    assert res.gap_pct == pytest.approx(4.0) and res.rvol == pytest.approx(3.0)
    assert res.passed == (rsi_ok and ind[f"EMA_{ind_cfg.ema_fast}"] > ind[f"EMA_{ind_cfg.ema_slow}"]
                          and last * 1.04 > ind[f"SMA_{ind_cfg.sma_slow}"])

    flat = evaluate_ticker("FLAT", uptrend, _snap("FLAT", last * 1.001, last), ind_cfg, scr_cfg)
    assert not flat.passed and any("gap" in f for f in flat.failures)

    stale = evaluate_ticker("STALE", uptrend, _snap("STALE", last * 1.04, last * 0.9), ind_cfg, scr_cfg)
    assert not stale.passed and any("stale feed" in f for f in stale.failures)

    no_history = evaluate_ticker("NH", uptrend, _snap("NH", last * 1.04, last, avg=float("nan")), ind_cfg, scr_cfg)
    assert no_history.rvol_source == "prior_session"

    cands = get_daily_candidates({"UP": uptrend, "FLAT": uptrend},
                                 {"UP": _snap("UP", last * 1.04, last), "FLAT": _snap("FLAT", last, last)},
                                 ind_cfg, scr_cfg)
    assert "FLAT" not in set(cands["ticker"])
