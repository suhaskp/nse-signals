import numpy as np
import pandas as pd
import pytest

from research.stress import stress_test, verdict


def trades(rs, start="2022-01-03", step=10):
    d = pd.bdate_range(start, periods=len(rs), freq=f"{step}B")
    return pd.DataFrame({"r_multiple": rs, "entry_date": d - pd.offsets.BDay(3), "exit_date": d})


def test_historical_drawdown_streak_and_recovery():
    res = stress_test(trades([1, -1, -1, -1, 2, -1, 3, 1]), risk_pct=0.01, n_sims=500)
    assert res.max_dd_r == pytest.approx(-3.0)
    assert res.longest_losing_streak == 3
    assert res.max_dd_pct == pytest.approx((1.01 * 0.99 ** 3) / 1.01 - 1)
    assert res.dd_recovered is not None and res.dd_days_to_recover > 0
    assert res.total_r == pytest.approx(3.0) and res.max_concurrent == 1


def test_not_recovered_and_yearly_table():
    res = stress_test(trades([2, 1, -1, -1, -1, -1, 0.5], step=60), n_sims=300)
    assert res.dd_recovered is None
    assert res.by_year["Trades"].sum() == 7 and set(res.by_year.columns) >= {"Year", "Total R", "Worst drawdown %"}


def test_monte_carlo_ranges_and_sizing():
    rng = np.random.default_rng(1)
    rs = rng.choice([-1.0, 2.0], size=120, p=[0.6, 0.4])
    res = stress_test(trades(list(rs), step=2), risk_pct=0.01, tolerance_pct=0.15, n_sims=2000)
    m = res.mc["one_year"]
    assert m["dd_p95"] <= m["dd_median"] <= 0 and 0 <= m["p_loss"] <= 1
    assert m["ret_p5"] <= m["ret_median"] <= m["ret_p95"]
    assert res.suggested_risk_pct is not None
    worse = stress_test(trades(list(rs), step=2), risk_pct=0.03, n_sims=2000)
    assert worse.mc["one_year"]["dd_p95"] < m["dd_p95"]  # more risk per trade -> deeper drawdowns
    level, text = verdict(res, 0.01, 0.15)
    assert level in {"ok", "caution", "danger"} and "risk at most" in text


def test_overlapping_trades_counted():
    t = pd.DataFrame({"r_multiple": [1, -1, 1, 1, -1], "entry_date": pd.to_datetime(["2024-01-01"] * 5),
                      "exit_date": pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14"])})
    assert stress_test(t, n_sims=100).max_concurrent == 5


def test_too_few_trades_returns_none():
    assert stress_test(trades([1, -1])) is None
