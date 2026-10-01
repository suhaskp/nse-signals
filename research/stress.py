"""Drawdown stress test for a set of closed trades (in R multiples).

Answers "could I live with the bad stretches?" rather than "what is the average?":

* Historical path (trades in exit order): worst drawdown in R and as % of equity at a given risk
  per trade (compounded fixed-fractional), when it started and bottomed, whether and when it
  recovered, longest losing streak, longest time without a new equity high, and every year.
* Monte Carlo: the historical trades are resampled (with replacement) thousands of times into
  one-year and full-length sequences. Because the order of wins and losses is luck, this shows
  the realistic range: typical and bad-case (95th percentile) drawdowns and losing streaks, the
  chance of a losing year, and the spread of outcomes.
* Sizing: the largest risk per trade that keeps the bad-case (95th percentile) one-year drawdown
  within your tolerance.

Caveat: trades are treated one after another. Several open at once can make drawdowns arrive
faster; the maximum number of overlapping trades is reported so this can be judged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StressResult:
    n_trades: int
    years: float
    trades_per_year: float
    total_r: float
    avg_r: float
    max_dd_r: float
    max_dd_pct: float
    dd_peak: str | None
    dd_trough: str | None
    dd_recovered: str | None
    dd_days_to_recover: int | None
    longest_losing_streak: int
    longest_flat_days: int
    max_concurrent: int
    by_year: pd.DataFrame
    curve: pd.DataFrame
    mc: dict = field(default_factory=dict)
    mc_dd_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    suggested_risk_pct: float | None = None
    avg_win_r: float = float("nan")
    avg_loss_r: float = float("nan")
    best_year: str = ""
    worst_year: str = ""


def _streak(losses: np.ndarray) -> int:
    best = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def _max_concurrent(trades: pd.DataFrame) -> int:
    if "entry_date" not in trades or trades.empty:
        return 1
    ev = pd.concat([pd.DataFrame({"d": pd.to_datetime(trades["entry_date"]), "x": 1}),
                    pd.DataFrame({"d": pd.to_datetime(trades["exit_date"]) + pd.Timedelta(days=1), "x": -1})])
    return int(ev.sort_values(["d", "x"])["x"].cumsum().max())


def stress_test(trades: pd.DataFrame, risk_pct: float = 0.01, tolerance_pct: float = 0.15,
                n_sims: int = 5000, seed: int = 7) -> StressResult | None:
    """Run the stress test. ``risk_pct`` is the fraction of equity risked per trade (1R)."""
    if trades is None or trades.empty or len(trades) < 5:
        return None
    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t = t.sort_values("exit_date").reset_index(drop=True)
    r = t["r_multiple"].to_numpy(dtype=float)
    span_days = max((t["exit_date"].max() - t["exit_date"].min()).days, 1)
    years = max(span_days / 365.25, 1 / 12)
    tpy = len(r) / years

    cum_r = np.cumsum(r)
    peak_r = np.maximum.accumulate(np.r_[0.0, cum_r])[1:]
    dd_r = cum_r - peak_r
    eq = np.cumprod(1 + r * risk_pct)
    peak_eq = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    dd_pct = eq / peak_eq - 1
    trough = int(np.argmin(dd_pct))
    peak_idx = int(np.argmax(eq[: trough + 1])) if eq[: trough + 1].max() >= 1 else None
    peak_val = eq[peak_idx] if peak_idx is not None else 1.0
    rec = next((i for i in range(trough + 1, len(eq)) if eq[i] >= peak_val), None)
    dates = t["exit_date"]
    start = dates.iloc[peak_idx] if peak_idx is not None else dates.iloc[0] - pd.Timedelta(days=1)

    # longest stretch (calendar days) without a new equity high
    highs = [dates.iloc[0] - pd.Timedelta(days=1)] + [dates.iloc[i] for i in range(len(eq))
                                                      if eq[i] >= np.max(np.r_[1.0, eq[:i]])]
    highs.append(dates.iloc[-1])
    flat = max((b - a).days for a, b in zip(highs[:-1], highs[1:])) if len(highs) > 1 else 0

    yr = t.assign(year=dates.dt.year)
    rows = []
    for y, g in yr.groupby("year"):
        rr = g["r_multiple"].to_numpy(float)
        e = np.cumprod(1 + rr * risk_pct)
        rows.append({"Year": int(y), "Trades": len(rr), "Win %": round(100 * (rr > 0).mean(), 1),
                     "Total R": round(rr.sum(), 2), "Return %": round(100 * (e[-1] - 1), 1),
                     "Worst drawdown %": round(100 * (e / np.maximum.accumulate(np.r_[1.0, e])[1:] - 1).min(), 1)})
    by_year = pd.DataFrame(rows)

    # ---- Monte Carlo
    rng = np.random.default_rng(seed)
    n_year = max(int(round(tpy)), 5)
    out: dict = {}
    for label, n in (("one_year", n_year), ("full", len(r))):
        s = rng.choice(r, size=(n_sims, n), replace=True)
        e = np.cumprod(1 + s * risk_pct, axis=1)
        pk = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), e], axis=1), axis=1)[:, 1:]
        dd = (e / pk - 1).min(axis=1)
        streaks = np.array([_streak(row <= 0) for row in s])
        final = e[:, -1] - 1
        out[label] = {"trades": n, "dd_median": float(np.median(dd)), "dd_p95": float(np.percentile(dd, 5)),
                      "dd_p99": float(np.percentile(dd, 1)), "streak_median": float(np.median(streaks)),
                      "streak_p95": float(np.percentile(streaks, 95)), "p_loss": float((final < 0).mean()),
                      "ret_p5": float(np.percentile(final, 5)), "ret_median": float(np.median(final)),
                      "ret_p95": float(np.percentile(final, 95)),
                      "p_dd_beyond_tolerance": float((dd < -tolerance_pct).mean())}
        if label == "one_year":
            mc_dd = dd
            s_year = s

    # ---- risk-per-trade that keeps the 95th-percentile one-year drawdown within tolerance
    suggested = None
    for rp in np.arange(0.0025, 0.0501, 0.0025):
        e = np.cumprod(1 + s_year * rp, axis=1)
        pk = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), e], axis=1), axis=1)[:, 1:]
        if np.percentile((e / pk - 1).min(axis=1), 5) >= -tolerance_pct:
            suggested = float(rp)
        else:
            break

    curve = pd.DataFrame({"date": dates, "equity": eq, "drawdown_pct": 100 * dd_pct, "cum_r": cum_r})
    return StressResult(
        n_trades=len(r), years=years, trades_per_year=tpy, total_r=float(r.sum()), avg_r=float(r.mean()),
        max_dd_r=float(dd_r.min()), max_dd_pct=float(dd_pct.min()),
        dd_peak=str(start.date()), dd_trough=str(dates.iloc[trough].date()),
        dd_recovered=str(dates.iloc[rec].date()) if rec is not None else None,
        dd_days_to_recover=int((dates.iloc[rec] - start).days) if rec is not None else None,
        longest_losing_streak=_streak(r <= 0), longest_flat_days=int(flat),
        max_concurrent=_max_concurrent(t), by_year=by_year, curve=curve, mc=out, mc_dd_pct=mc_dd,
        suggested_risk_pct=suggested,
        avg_win_r=float(r[r > 0].mean()) if (r > 0).any() else float("nan"),
        avg_loss_r=float(r[r <= 0].mean()) if (r <= 0).any() else float("nan"),
        best_year=(f"{int(by_year.loc[by_year['Return %'].idxmax(), 'Year'])} "
                   f"({by_year['Return %'].max():+.1f}%)") if not by_year.empty else "",
        worst_year=(f"{int(by_year.loc[by_year['Return %'].idxmin(), 'Year'])} "
                    f"({by_year['Return %'].min():+.1f}%)") if not by_year.empty else "")


def verdict(res: StressResult, risk_pct: float, tolerance_pct: float) -> tuple[str, str]:
    """(level, plain-language summary) where level is ok | caution | danger."""
    m1 = res.mc["one_year"]
    bad = -m1["dd_p95"]
    lines = [f"At {100 * risk_pct:.2f}% risk per trade (about {res.trades_per_year:.0f} trades a year): "
             f"the worst historical drawdown was {100 * -res.max_dd_pct:.1f}% of equity "
             f"({res.max_dd_r:.1f}R)"
             + (f", recovered after {res.dd_days_to_recover} days." if res.dd_recovered else ", not yet recovered."),
             f"In a typical year expect a drawdown of about {100 * -m1['dd_median']:.1f}%; in a bad year "
             f"(1 in 20) about {100 * bad:.1f}%. Losing streaks of {m1['streak_median']:.0f} trades are normal and "
             f"{m1['streak_p95']:.0f} can happen. The chance of a losing year is about {100 * m1['p_loss']:.0f}%."]
    if res.suggested_risk_pct is not None:
        lines.append(f"To keep a bad-year drawdown within {100 * tolerance_pct:.0f}%, risk at most "
                     f"{100 * res.suggested_risk_pct:.2f}% of equity per trade.")
    else:
        lines.append(f"Even at 0.25% risk per trade, a bad year could exceed a {100 * tolerance_pct:.0f}% drawdown.")
    level = "ok" if bad <= tolerance_pct and m1["p_loss"] < 0.3 else ("caution" if bad <= 1.5 * tolerance_pct
                                                                       else "danger")
    return level, " ".join(lines)
