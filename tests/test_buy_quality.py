import numpy as np
import pandas as pd
import pytest

from config.config import BuyQualityConfig, ConfigError, load_config
from screeners.buy_quality import backtest_quality, evaluate_latest, plan_entry, quality_frame
from screeners.screener import atr

Q = BuyQualityConfig()


def textbook(vol_mult=2.0, overhead_high=None, extend_atr=0.5, n=330, seed=4):
    """Uptrend, then a ~60-day sideways base just under its high, then a breakout bar."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-25", periods=n)
    up = n - 61
    c = np.empty(n)
    c[:up] = 100 * np.exp(np.cumsum(0.0018 + rng.normal(0, 0.004, up)))
    top = c[up - 1]
    c[up:n - 1] = top * (0.97 + 0.03 * np.sin(np.arange(n - 1 - up) / 4))  # base up to the prior high
    o = c * 0.998
    h, lo = c * 1.004, c * 0.994
    v = np.full(n, 1e6)
    df = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v}, index=idx)
    if overhead_high:  # an older, higher 52-week high overhead
        df.iloc[n - 150, df.columns.get_loc("High")] = df["Close"].iloc[n - 2] * overhead_high
    a = atr(df["High"], df["Low"], df["Close"], 14).iloc[-2]
    level = df["High"].iloc[-56:-1].max()  # 55-day high; the base reaches the prior top, so this is also the 52-week high
    close = level + extend_atr * a
    df.iloc[-1] = [level - 0.2 * a, close * 1.001, level - 0.3 * a, close, 1e6 * vol_mult]
    return df


def bench(up=True, n=330):
    idx = pd.bdate_range(end="2026-09-25", periods=n)
    drift = np.r_[np.full(n - 70, 0.0012), np.zeros(70)] if up else np.r_[np.zeros(n - 70), np.full(70, -0.004)]
    c = 20000 * np.exp(np.cumsum(drift))
    return pd.DataFrame({"Open": c, "High": c * 1.003, "Low": c * 0.997, "Close": c, "Volume": 1e6}, index=idx)


def last(df, b=None, cfg=Q):
    return quality_frame(df, cfg, b if b is not None else bench()).iloc[-1]


def test_textbook_breakout_passes_every_gate():
    r = last(textbook())
    gates = ["breakout", "close_strength", "not_extended", "volume", "liquidity", "trend", "regime", "stop_ok",
             "reward_risk"]
    assert {g: bool(r[g]) for g in gates} == {g: True for g in gates}
    assert r["buy"] and r["rr"] >= 2 and r["stop"] < r["level"] < r["close"] < r["target"]
    assert r["close"] - r["stop"] == pytest.approx(max(r["close"] - (r["level"] - 0.5 * r["atr"]), 1.5 * r["atr"]))


@pytest.mark.parametrize("kwargs, gate", [
    ({"vol_mult": 1.1}, "volume"),
    ({"extend_atr": 2.0}, "not_extended"),
    ({"overhead_high": 1.02}, "reward_risk"),
])
def test_each_weakness_is_caught(kwargs, gate):
    r = last(textbook(**kwargs))
    assert not r[gate] and not r["buy"]


def test_market_regime_blocks_buys():
    r = last(textbook(), bench(up=False))
    assert not r["regime"] and not r["buy"]
    assert last(textbook(), bench(up=False), BuyQualityConfig(regime_required=False))["regime"]


def test_target_capped_by_resistance_and_atr():
    r = last(textbook(overhead_high=1.02))
    assert r["target"] <= r["resistance"] + 1e-9
    r2 = last(textbook())
    assert r2["target"] <= r2["close"] + Q.target_max_atr * r2["atr"] + 1e-9


def test_no_lookahead_in_gates():
    df = textbook()
    base = quality_frame(df, Q, bench())
    tampered = df.copy()
    tampered.iloc[-1, tampered.columns.get_loc("Volume")] = 1
    after = quality_frame(tampered, Q, bench())
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


@pytest.mark.parametrize("open_px, expect", [(None, "OK"), ("below", "SKIP: opened below"), ("gap", "SKIP: opened"),
                                             ("far", "SKIP")])
def test_next_day_entry_rules(open_px, expect):
    r = last(textbook())
    px = {None: r["close"] * 1.003, "below": r["level"] * 0.995, "gap": r["close"] * 1.03,
          "far": r["close"] * 1.019}[open_px]
    p = plan_entry(r["level"], r["close"], r["atr"], r["target"], px, Q)
    assert p["status"].startswith(expect)
    if p["status"] == "OK":
        assert p["rr"] >= 2 and p["stop"] < px < p["target"]


def test_backtest_enters_next_open_and_funnel_counts():
    df = textbook()
    after = pd.DataFrame({"Open": [df["Close"].iloc[-1] * 1.002] * 3, "High": [df["Close"].iloc[-1] * 1.08] * 3,
                          "Low": [df["Close"].iloc[-1]] * 3, "Close": [df["Close"].iloc[-1] * 1.05] * 3,
                          "Volume": [1e6] * 3}, index=pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=3))
    full = pd.concat([df, after])
    b = bench(n=len(full))
    b.index = full.index
    trades, funnel = backtest_quality({"X": full}, Q, b, 0.0025)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_date"] == after.index[0].date() and t["entry"] == pytest.approx(after["Open"].iloc[0])
    assert t["outcome"] == "target" and t["r_multiple"] > 1.9
    stages = dict(zip(funnel["Stage"], funnel["Signals remaining"]))
    assert stages["raw breakouts"] >= stages["volume"] >= stages["next-day entry OK"] == 1


def test_evaluate_latest_reports_failed_checks():
    ev = evaluate_latest({"GOOD": textbook(), "WEAK": textbook(vol_mult=1.0)}, Q, bench())
    ev = ev.set_index("Ticker")
    assert ev.at["GOOD", "passes"] and ev.at["GOOD", "Failed"] == ""
    assert not ev.at["WEAK", "passes"] and "volume" in ev.at["WEAK", "Failed"]


def test_minimum_two_to_one_is_enforced_in_config():
    with pytest.raises(ConfigError):
        load_config(overrides={"buy_quality": {"min_reward_risk": 1.5}})
