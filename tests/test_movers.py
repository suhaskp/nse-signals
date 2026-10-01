from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data_ingestion.data_fetcher import IST
from research.movers import follow_up, historical_movers, record_movers, verdict


def market(n=120, jump_at=80, follow=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-28", periods=n)
    r = rng.normal(0, 0.005, n)
    r[jump_at] = 0.05
    r[jump_at + 1: jump_at + 11] += follow
    c = 100 * np.exp(np.cumsum(r))
    v = np.full(n, 1e6)
    v[jump_at] = 3e6
    return pd.DataFrame({"Open": np.r_[c[0], c[:-1]], "High": c * 1.003, "Low": c * 0.997, "Close": c, "Volume": v},
                        index=idx)


def bench(n=120):
    idx = pd.bdate_range(end="2026-09-28", periods=n)
    c = np.full(n, 20000.0)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx)


def test_historical_proxy_detects_follow_through():
    hot = {f"H{i}": market(follow=0.004, seed=i) for i in range(10)}
    flat = {f"F{i}": market(follow=0.0, seed=i + 50) for i in range(10)}
    ph = historical_movers(hot, bench(), min_turnover_cr=0, cost_pct=0.0)
    pf = historical_movers(flat, bench(), min_turnover_cr=0, cost_pct=0.0)
    s_hot = ph["summary"].set_index("Hold").loc["5 sessions"]
    s_flat = pf["summary"].set_index("Hold").loc["5 sessions"]
    assert ph["n"] >= 10 and s_hot["Avg return %"] > s_flat["Avg return %"]
    assert verdict(ph)[0] == "ok"
    assert "chasing movers has not beaten" in verdict(pf)[1] or verdict(pf)[0] == "ok"


def test_record_and_follow_up(tmp_path):
    path = tmp_path / "movers_log.csv"
    df = market(n=120, jump_at=100)
    d = df.index[100]
    movers = pd.DataFrame({"Ticker": ["X"], "Price": [float(df["Close"].iloc[100]) * 0.99], "Today %": [4.0],
                           "Volume pace": [2.0]})
    early = datetime.combine(d.date(), datetime.min.time()).replace(hour=11, tzinfo=IST)
    assert not record_movers(movers, early, path)                      # before the 12:00 checkpoint
    noon = early.replace(hour=12, minute=5)
    assert record_movers(movers, noon, path) and not record_movers(movers, noon.replace(minute=30), path)
    assert record_movers(movers, noon.replace(hour=15, minute=20), path)  # the late checkpoint
    fu = follow_up(path, {"X": df}, bench())
    assert fu["days"] == 1 and len(fu["rows"]) == 2
    row = fu["rows"].iloc[0]
    assert row["Held to close %"] == pytest.approx(100 * (1 / 0.99 - 1), abs=0.01)
    assert np.isfinite(row["5d %"]) and np.isfinite(row["10d %"])
    assert set(fu["summary"]["Horizon"]) >= {"Same day (to the close)", "5 sessions"}
