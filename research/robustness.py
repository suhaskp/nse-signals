"""Is the edge real or fragile? Robustness checks, the multiple-testing adjustment, and live health.

* Parameter sensitivity: the BUY rules re-run with the breakout lookback and the volume requirement nudged
  up and down. A real effect degrades gently; a fluke works only at one exact setting.
* Execution stress: costs x1.5 and x2, extra slippage, and entering one session late.
* Research trials and the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the probability that the
  chosen setup's true Sharpe ratio is above zero after allowing for how many setups were tried and for
  non-normal returns.
* Strategy health: the forward record compared with the backtest, only once there are enough trades.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import BuyQualityConfig
from screeners.buy_quality import _stats, backtest_quality

EULER = 0.5772156649


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float], n_trials: int) -> dict:
    """Deflated Sharpe Ratio of ``returns`` (per-period), given the per-period Sharpe ratios of the trials."""
    r = pd.Series(returns).dropna()
    t = len(r)
    if t < 10 or r.std() == 0:
        return {"dsr": np.nan, "sr": np.nan, "sr0": np.nan, "n_trials": n_trials, "periods": t}
    sr = r.mean() / r.std()
    skew, kurt = float(r.skew()), float(r.kurt()) + 3  # pandas kurt is excess kurtosis
    var_trials = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    n = max(n_trials, 2)
    sr0 = math.sqrt(max(var_trials, 0)) * ((1 - EULER) * _norm_ppf(1 - 1 / n) + EULER * _norm_ppf(1 - 1 / (n * math.e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return {"dsr": _norm_cdf((sr - sr0) * math.sqrt(t - 1) / denom), "sr": sr, "sr0": sr0, "n_trials": n_trials,
            "periods": t}


def count_trials(path: Path, experiments: list[str]) -> int:
    """Remember every distinct experiment ever run (tuning setups, lab variants, sensitivity runs)."""
    try:
        seen = set(json.loads(path.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        seen = set()
    seen |= set(experiments)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen)))
    return len(seen)


def sensitivity(histories: Mapping[str, pd.DataFrame], cfg: BuyQualityConfig, benchmark, cost_pct: float,
                breadth=None) -> pd.DataFrame:
    """BUY rules with one parameter nudged at a time."""
    rows = []
    for name, field, values in (("Breakout lookback (days)", "breakout_lookback", (45, 50, 55, 60, 65)),
                                ("Volume requirement (× average)", "volume_multiple", (1.25, 1.5, 1.75, 2.0))):
        for v in values:
            tr, _ = backtest_quality(histories, replace(cfg, **{field: v}), benchmark, cost_pct, breadth)
            rows.append({"Parameter": name, "Value": v, "Current": v == getattr(cfg, field), **_stats(tr)})
    return pd.DataFrame(rows)


def sensitivity_verdict(sens: pd.DataFrame) -> tuple[str, str]:
    if sens.empty:
        return "none", "No sensitivity results yet."
    notes, level = [], "ok"
    for name, g in sens.groupby("Parameter", sort=False):
        g = g[g["Trades"] >= 30]
        if len(g) < 3:
            notes.append(f"{name}: too few trades to judge")
            continue
        signs = set(np.sign(g["Avg R"].round(3)))
        spread = g["Avg R"].max() - g["Avg R"].min()
        cur = g.loc[g["Current"], "Avg R"]
        best_is_isolated = len(cur) and cur.iloc[0] == g["Avg R"].max() and \
            (g["Avg R"].sort_values().iloc[-2] < cur.iloc[0] - 0.15 if len(g) > 1 else False)
        if len(signs - {0}) > 1 or best_is_isolated:
            level = "bad"
            notes.append(f"{name}: results change sign or peak only at the current setting (possible overfitting)")
        elif spread > 0.15:
            level = "caution" if level == "ok" else level
            notes.append(f"{name}: results vary noticeably ({spread:.2f}R between settings)")
        else:
            notes.append(f"{name}: stable across nearby settings")
    return level, "; ".join(notes) + "."


def execution_stress(base_trades: pd.DataFrame, delayed_trades: pd.DataFrame, cost_pct: float) -> pd.DataFrame:
    """Profit factor / average R under worse execution."""
    if base_trades.empty:
        return pd.DataFrame()

    def with_extra(extra: float) -> pd.DataFrame:
        t = base_trades.copy()
        risk = (t["entry"] - t["stop"]).where(lambda x: x > 0)
        t["r_multiple"] = t["r_multiple"] - extra * t["entry"] / risk
        return t.dropna(subset=["r_multiple"])

    cases = [("Base case (estimated costs)", base_trades), ("Costs ×1.5", with_extra(0.5 * cost_pct)),
             ("Costs ×2", with_extra(cost_pct)), ("Extra 0.2% slippage per trade", with_extra(0.002)),
             ("Entry one session late", delayed_trades)]
    rows = [{"Scenario": n, **_stats(t)} for n, t in cases]
    return pd.DataFrame(rows)


def stress_verdict(tbl: pd.DataFrame) -> tuple[str, str]:
    if tbl.empty or tbl["Trades"].iloc[0] < 30:
        return "none", "Too few trades for an execution stress verdict."
    worst = tbl["Avg R"].min()
    x2 = tbl.set_index("Scenario").at["Costs ×2", "Avg R"]
    if worst > 0.05:
        return "ok", "ROBUST: every scenario still made money per trade."
    if x2 > 0:
        return "caution", "MODERATELY ROBUST: still positive at double costs, but at least one scenario is close to zero."
    return "bad", "FRAGILE: the edge disappears under realistic execution problems."


def health(forward: pd.DataFrame, backtest_trades: pd.DataFrame, min_trades: int = 30) -> dict:
    """Is the live (forward) record consistent with the backtest?"""
    closed = forward[forward["exit"].notna()] if forward is not None and not forward.empty else pd.DataFrame()
    bt = _stats(backtest_trades) if backtest_trades is not None and not backtest_trades.empty else {}
    out = {"closed": len(closed), "backtest": bt, "live": _stats(closed.rename(columns={}))
           if len(closed) else {}, "min_trades": min_trades}
    if len(closed) < min_trades or not bt:
        out.update(level="none", text=f"Not enough live trades yet ({len(closed)} of {min_trades} needed) to judge "
                                      "whether the strategy is behaving as the backtest expects.")
        return out
    r = closed["r_multiple"].to_numpy(float)
    se = r.std(ddof=1) / math.sqrt(len(r))
    gap = r.mean() - bt["Avg R"]
    if gap < -2 * se:
        out.update(level="bad", text=f"DEGRADATION SUSPECTED: live trades average {r.mean():+.2f}R vs {bt['Avg R']:+.2f}R "
                                     f"in the backtest ({len(r)} trades; more than 2 standard errors worse). Reduce or "
                                     "stop new deployment and review.")
    elif gap < -se:
        out.update(level="caution", text=f"WATCH: live trades average {r.mean():+.2f}R vs {bt['Avg R']:+.2f}R in the "
                                         f"backtest ({len(r)} trades): somewhat worse, within normal variation so far.")
    else:
        out.update(level="ok", text=f"HEALTHY: live trades average {r.mean():+.2f}R vs {bt['Avg R']:+.2f}R in the "
                                    f"backtest ({len(r)} trades), consistent with expectations.")
    return out
