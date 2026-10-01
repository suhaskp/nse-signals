from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.config import BuyQualityConfig, load_config
from data_ingestion.data_fetcher import IST
from screeners.buy_quality import (breakout_status, evaluate_latest, filter_ablation, quality_frame, readiness,
                                   regime_breakdown, regime_state, setup_state)
from tests.test_buy_quality import bench, textbook

Q = BuyQualityConfig()


def test_readiness_and_states():
    r = quality_frame(textbook(), Q, bench()).iloc[-1]
    assert readiness(r, Q) == 100 and setup_state(r, 100) == "BUY NOW"
    weak_vol = quality_frame(textbook(vol_mult=1.0), Q, bench()).iloc[-1]
    rd = readiness(weak_vol, Q)
    assert 85 <= rd < 100 and setup_state(weak_vol, rd) == "WATCH CLOSELY"
    off = quality_frame(textbook(), Q, bench(up=False)).iloc[-1]
    assert readiness(off, Q) == 100 and setup_state(off, 100) != "BUY NOW"  # regime excluded from readiness


def test_breakout_status_labels():
    base = quality_frame(textbook(), Q, bench()).iloc[-1].copy()
    assert breakout_status(base, Q).startswith("✅")
    ext = base.copy()
    ext["extension_atr"] = 3.0
    assert "don't chase" in breakout_status(ext, Q)
    for pct, icon in ((-0.5, "🟢"), (-2.0, "🟡"), (-6.0, "⚪")):
        b = base.copy()
        b["close"] = b["level"] * (1 + pct / 100)
        assert breakout_status(b, Q).startswith(icon)


def test_regime_states():
    idx = pd.bdate_range(end="2026-09-25", periods=330)
    up, down = bench(up=True), bench(up=False)
    assert regime_state(up, None, Q, idx).iloc[-1] == "on"
    assert regime_state(down, None, Q, idx).iloc[-1] in {"off", "neutral"}
    flat = down.copy()
    sma = flat["Close"].rolling(200).mean().iloc[-1]
    flat.iloc[-1, flat.columns.get_loc("Close")] = sma * 0.99      # 1% below the 200-day -> neutral
    assert regime_state(flat, None, Q, idx).iloc[-1] == "neutral"
    breadth = pd.Series(np.r_[np.full(310, 0.2), np.full(20, 0.4)], index=idx)  # +20 pts in 20 sessions
    assert regime_state(down, breadth, Q, idx).iloc[-1] == "neutral"
    three = BuyQualityConfig(regime_mode="three_state")
    r = quality_frame(textbook(), three, flat).iloc[-1]
    assert r["regime"] and r["regime_state"] == "neutral"


def test_evaluate_latest_has_readiness_columns():
    ev = evaluate_latest({"GOOD": textbook()}, Q, bench())
    assert {"Readiness", "State", "Breakout status", "To breakout %", "Regime state"} <= set(ev.columns)


def test_ablation_and_regime_breakdown():
    from data_ingestion.synthetic import SyntheticMarket
    m = SyntheticMarket(now=datetime(2026, 9, 28, 16, 30, tzinfo=IST))
    h = {f"A{i:02d}": m.daily(f"A{i:02d}") for i in range(40)}
    ab = filter_ablation(h, Q, m.daily("^NSEI"), 0.0025)
    assert list(ab["Stage"])[0] == "Breakout only" and len(ab) == 9
    assert (ab["Trades"].diff().dropna() <= 0).all()          # each check can only remove trades
    assert set(ab["Effect"]) <= {"baseline", "improves results", "hurts results", "mostly just cuts trades",
                                 "no trades left"}
    from screeners.buy_quality import backtest_quality
    tr, _ = backtest_quality(h, BuyQualityConfig(regime_required=False), m.daily("^NSEI"), 0.0025)
    rb = regime_breakdown(tr)
    if not tr.empty:
        assert rb["Trades"].sum() == len(tr) and "Regime at signal" in rb


def test_breadth_trend_keys():
    from research.market import market_breadth
    from tests.planted import planted_market
    h, _ = planted_market(n_stocks=10, n_days=300)
    b = market_breadth(h)
    assert {"pct_above_200_5d", "pct_above_200_20d", "pct_above_50_20d"} <= set(b)
