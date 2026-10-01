"""Portfolio construction on top of the ranker's out-of-sample predictions.

Levers tested (all with turnover-based costs, so trading less is rewarded):

* ``top_n``     how many stocks to hold (10 or 20)
* ``buffer``    hold-buffer: keep an existing holding while its rank stays in the top
                (1 - buffer) of stocks, instead of replacing it whenever it drops out of
                the top ``top_n``. Cuts turnover and costs.
* ``regime``    sit in cash when the Nifty closed below its 200-day average
* ``weighting`` equal weights or inverse-volatility weights (calmer stocks get more)
* ``horizon``   holding period / rebalance interval in sessions (10 or 20)

Costs: ``cost_pct`` is a full round trip (buy + sell). Each rebalance pays
``cost_pct x turnover`` with turnover = sum(|new weight - old weight|) / 2, so replacing the whole
portfolio = 1.0 (a full sell and a full buy), while buying in from cash = 0.5 (only the buy side).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Settings:
    horizon: int = 10
    top_n: int = 10
    buffer: float | None = None
    regime: bool = False
    weighting: str = "equal"

    def label(self) -> str:
        parts = [f"{self.horizon}-day hold", f"top {self.top_n}",
                 f"keep while in top {round(100 * (1 - self.buffer))}%" if self.buffer else "no hold-buffer",
                 "cash when Nifty < 200-day" if self.regime else "always invested",
                 "volatility-scaled weights" if self.weighting == "inv_vol" else "equal weights"]
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE = Settings()


def relabel(panel: pd.DataFrame, histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None,
            horizon: int) -> pd.DataFrame:
    """Copy of the panel with labels (fwd_ret, target, fwd_excess, idx_fwd) for another horizon.

    Features are unchanged, so no feature can see the future; only the labels move.
    """
    fwd = pd.concat([(df["Close"].shift(-horizon) / df["Open"].shift(-1) - 1).rename("fwd_ret").to_frame()
                     .assign(ticker=t) for t, df in histories.items()]).rename_axis("date").reset_index()
    out = panel.drop(columns=["fwd_ret", "target", "fwd_excess", "idx_fwd"], errors="ignore")
    out = out.merge(fwd, on=["date", "ticker"], how="left")
    g = out.groupby("date")["fwd_ret"]
    out["target"] = g.rank(pct=True) - 0.5
    out["fwd_excess"] = out["fwd_ret"] - g.transform("median")
    if benchmark is not None:
        idx = (benchmark["Close"].shift(-horizon) / benchmark["Open"].shift(-1) - 1).rename("idx_fwd")
        out = out.merge(idx, left_on="date", right_index=True, how="left")
    else:
        out["idx_fwd"] = np.nan
    return out


def target_weights(day: pd.DataFrame, held: Mapping[str, float], s: Settings, risk_on: bool) -> dict[str, float]:
    """Target portfolio for one rebalance date from that date's predictions."""
    if s.regime and not risk_on:
        return {}
    ranks = day["pred"].rank(pct=True)
    keep: list[str] = []
    if s.buffer:
        keep = [t for t in held if t in ranks.index and ranks[t] >= s.buffer]
        keep = sorted(keep, key=lambda t: -day.at[t, "pred"])[: s.top_n]
    fill = day.drop(index=keep).nlargest(max(s.top_n - len(keep), 0), "pred").index.tolist()
    names = keep + fill
    if not names:
        return {}
    if s.weighting == "inv_vol":
        iv = 1 / day.loc[names, "vol_21"].astype(float).clip(lower=1e-4).fillna(day["vol_21"].median())
        w = iv / iv.sum()
    else:
        w = pd.Series(1 / len(names), index=names)
    return w.to_dict()


def simulate(oos: pd.DataFrame, s: Settings, cost_pct: float, min_turnover_cr: float = 0.0) -> pd.DataFrame:
    """Rebalance every ``s.horizon`` dates; returns one row per holding period."""
    d = oos[oos["fwd_ret"].notna()]
    if min_turnover_cr and "turnover_cr" in d:
        d = d[d["turnover_cr"] >= min_turnover_cr]
    by_date = {k: g.set_index("ticker") for k, g in d.groupby("date")}
    dates = sorted(by_date)[:: s.horizon]
    held: dict[str, float] = {}
    rows = []
    for dt in dates:
        day = by_date[dt]
        if len(day) < s.top_n:
            continue
        ro = day["idx_above_200"].iloc[0]
        risk_on = bool(ro) if pd.notna(ro) else True
        new = target_weights(day, held, s, risk_on)
        names = set(new) | set(held)
        turnover = sum(abs(new.get(t, 0.0) - held.get(t, 0.0)) for t in names) / 2
        gross = float(sum(w * day.at[t, "fwd_ret"] for t, w in new.items())) if new else 0.0
        idx = day["idx_fwd"].iloc[0] if "idx_fwd" in day else np.nan
        rows.append({"date": pd.Timestamp(dt), "ret": gross - cost_pct * turnover, "gross": gross,
                     "turnover": turnover, "invested": float(sum(new.values())), "n": len(new),
                     "nifty": float(idx) if pd.notna(idx) else np.nan})
        held = new
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["ret", "gross", "turnover", "invested", "n", "nifty"])


def metrics(sim: pd.DataFrame, horizon: int) -> dict[str, float]:
    """CAGR, volatility, Sharpe, drawdown, turnover and excess vs the Nifty for one simulation."""
    if sim.empty:
        return {k: np.nan for k in ("cagr", "vol", "sharpe", "max_dd", "nifty_cagr", "excess_cagr",
                                    "avg_turnover", "beat_nifty", "periods")}
    ppy = 252 / horizon
    r = sim["ret"]
    eq = (1 + r).cumprod()
    years = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / years) - 1
    nif = sim["nifty"].dropna()
    n_cagr = (1 + nif).prod() ** (1 / (len(nif) / ppy)) - 1 if len(nif) else np.nan
    return {"cagr": float(cagr), "vol": float(r.std() * np.sqrt(ppy)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else np.nan,
            "max_dd": float((eq / eq.cummax() - 1).min()), "nifty_cagr": float(n_cagr),
            "excess_cagr": float(cagr - n_cagr) if np.isfinite(n_cagr) else np.nan,
            "avg_turnover": float(sim["turnover"].mean()),
            "beat_nifty": float((sim["ret"] > sim["nifty"]).mean()) if len(nif) else np.nan,
            "periods": int(len(r))}
