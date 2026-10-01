"""Cross-sectional stock ranker: which stocks are likely to beat the rest over the next N sessions.

Method
------
For every stock and every date *t* (using data up to the close of *t* only):

* **Features** (~20): momentum over 1 week to 12 months (including 12-1 month momentum),
  distance from the 52-week high/low, volatility and its trend, RSI, volume surge,
  liquidity, trend (price vs 50/200-day averages), strength vs the Nifty and vs the
  stock's sector, a 125-day breakout flag, and market-wide context (index trend,
  index volatility, market breadth).
* Stock-level features are converted to **percentile ranks within each date**, so the
  model learns "stronger than peers" rather than absolute levels that drift over time.
* **Target**: percentile rank (within the date) of the return from the next day's open
  to the close ``horizon`` sessions later, i.e. relative performance you could actually
  capture by acting on a signal after the close.

Validation
----------
Purged walk-forward: train on the past, test on the next block of dates, with an
embargo of ``horizon + 5`` sessions between them so overlapping labels cannot leak.
Training uses every ``train_stride``-th date to reduce label overlap. Out-of-sample
predictions are graded on:

* **Rank IC**: daily Spearman correlation between predicted and realised returns. For
  equity ranking models a mean IC of 0.02-0.05 is typical of a real, small edge.
* **Decile spread**: realised return of the top 10% minus the bottom 10%.
* **Portfolio simulation**: every ``horizon`` sessions buy the top ``top_n`` liquid
  stocks, equal weight, after costs; compared with holding the Nifty over the same
  windows, with and without a regime filter (cash when the Nifty is below its
  200-day average).

Explanations use XGBoost's per-feature contributions (SHAP values) for each prediction.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.config import RankerConfig
from models.ml_engine import PurgedWalkForwardSplit
from research.market import breadth_series
from screeners.screener import rsi

logger = logging.getLogger(__name__)

RANKED = ["ret_5", "ret_21", "ret_63", "ret_126", "mom_12_1", "dist_high_252", "dist_low_252",
          "vol_21", "vol_63", "vol_ratio", "rsi14", "vol_surge", "log_turnover", "c_sma50", "c_sma200",
          "sma50_200", "rel_index_63", "sector_rel_63"]
MARKET = ["idx_above_200", "idx_ret_21", "idx_vol_21", "breadth_200"]
FLAGS = ["breakout_125"]
FEATURES = [f"r_{f}" for f in RANKED] + MARKET + FLAGS

LABELS = {
    "ret_5": "1-week return", "ret_21": "1-month return", "ret_63": "3-month return",
    "ret_126": "6-month return", "mom_12_1": "12-month momentum (ex last month)",
    "dist_high_252": "closeness to 52-week high", "dist_low_252": "distance above 52-week low",
    "vol_21": "1-month volatility", "vol_63": "3-month volatility", "vol_ratio": "volatility trend",
    "rsi14": "RSI(14)", "vol_surge": "recent volume surge", "log_turnover": "liquidity (turnover)",
    "c_sma50": "price vs 50-day average", "c_sma200": "price vs 200-day average",
    "sma50_200": "50-day vs 200-day average", "rel_index_63": "3-month strength vs Nifty",
    "sector_rel_63": "3-month strength vs its sector", "idx_above_200": "Nifty above its 200-day average",
    "idx_ret_21": "Nifty 1-month return", "idx_vol_21": "Nifty volatility", "breadth_200": "market breadth",
    "breakout_125": "fresh 125-day breakout",
}


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def stock_features(df: pd.DataFrame, horizon: int, index_close: pd.Series | None = None) -> pd.DataFrame:
    """Time-series features known at each close, plus the forward label (future data)."""
    c, h, lo, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    lr = np.log(c).diff()
    sma50, sma200 = c.rolling(50, min_periods=50).mean(), c.rolling(200, min_periods=200).mean()
    vol21, vol63 = lr.rolling(21, min_periods=21).std(), lr.rolling(63, min_periods=63).std()
    turnover = (c * v).rolling(20, min_periods=20).mean() / 1e7
    out = pd.DataFrame({
        "ret_5": c / c.shift(5) - 1, "ret_21": c / c.shift(21) - 1, "ret_63": c / c.shift(63) - 1,
        "ret_126": c / c.shift(126) - 1, "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "dist_high_252": c / h.rolling(252, min_periods=126).max() - 1,
        "dist_low_252": c / lo.rolling(252, min_periods=126).min() - 1,
        "vol_21": vol21, "vol_63": vol63, "vol_ratio": vol21 / vol63, "rsi14": rsi(c, 14),
        "vol_surge": np.log((v.rolling(5, min_periods=5).mean() / v.rolling(63, min_periods=63).mean())
                            .where(lambda x: x > 0)),
        "log_turnover": np.log(turnover.clip(lower=1e-6)), "turnover_cr": turnover,
        "c_sma50": c / sma50 - 1, "c_sma200": c / sma200 - 1, "sma50_200": sma50 / sma200 - 1,
        "breakout_125": (c > h.shift(1).rolling(125, min_periods=125).max()).astype(float),
        "close": c,
        # Label: next open -> close `horizon` sessions later. Uses FUTURE data; never a feature.
        "fwd_ret": c.shift(-horizon) / o.shift(-1) - 1,
    }, index=df.index)
    if index_close is not None:
        ic = index_close.reindex(df.index).ffill()
        out["rel_index_63"] = out["ret_63"] - (ic / ic.shift(63) - 1)
    else:
        out["rel_index_63"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def index_forward_return(benchmark: pd.DataFrame, horizon: int) -> pd.Series:
    """Index return from the next open to the close ``horizon`` sessions later."""
    return benchmark["Close"].shift(-horizon) / benchmark["Open"].shift(-1) - 1


def build_panel(histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None,
                sectors: Mapping[str, str], horizon: int) -> pd.DataFrame:
    """Long panel (date, ticker) with ranked features, market context and labels."""
    idx_close = benchmark["Close"] if benchmark is not None else None
    frames = []
    for sym, df in histories.items():
        if len(df) < 260:
            continue
        f = stock_features(df, horizon, idx_close)
        f["ticker"], f["sector"] = sym, sectors.get(sym, "Unknown")
        frames.append(f)
    if not frames:
        raise ValueError("No stock has the ~1 year of history the ranker needs")
    panel = pd.concat(frames).rename_axis("date").reset_index()
    for col in RANKED + ["fwd_ret"]:
        if col in panel:
            panel[col] = panel[col].astype("float32")

    med = panel.groupby(["date", "sector"])["ret_63"].transform("median")
    panel["sector_rel_63"] = panel["ret_63"] - med

    if benchmark is not None:
        bc = benchmark["Close"]
        mk = pd.DataFrame({
            "idx_above_200": (bc > bc.rolling(200, min_periods=200).mean()).astype(float),
            "idx_ret_21": bc / bc.shift(21) - 1,
            "idx_vol_21": np.log(bc).diff().rolling(21).std() * np.sqrt(252),
            "idx_fwd": index_forward_return(benchmark, horizon),
        })
    else:
        mk = pd.DataFrame(columns=["idx_above_200", "idx_ret_21", "idx_vol_21", "idx_fwd"])
    mk["breadth_200"] = breadth_series(histories)["pct_above_200"]
    panel = panel.merge(mk, left_on="date", right_index=True, how="left")

    g = panel.groupby("date")
    for col in RANKED:
        panel[f"r_{col}"] = g[col].rank(pct=True).astype("float32")
    panel["target"] = g["fwd_ret"].rank(pct=True) - 0.5
    panel["fwd_excess"] = panel["fwd_ret"] - g["fwd_ret"].transform("median")
    panel[FEATURES] = panel[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float32")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _X(rows: pd.DataFrame) -> pd.DataFrame:
    """Model inputs as float32, whatever dtype pandas produced upstream."""
    return rows[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float32")


def _daily_ic(df: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    """Spearman rank IC per date between prediction and realised forward return."""
    d = df.dropna(subset=[pred_col, "fwd_ret"])
    d = d.assign(rp=d.groupby("date")[pred_col].rank(), rf=d.groupby("date")["fwd_ret"].rank())
    ic = d.groupby("date")[["rp", "rf"]].corr().unstack().iloc[:, 1]
    counts = d.groupby("date").size()
    return ic[counts.reindex(ic.index) >= 10].rename("ic")


def _decile_spread(df: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    d = df.dropna(subset=[pred_col, "fwd_ret"])
    q = d.groupby("date")[pred_col].rank(pct=True)
    top = d[q >= 0.9].groupby("date")["fwd_ret"].mean()
    bot = d[q <= 0.1].groupby("date")["fwd_ret"].mean()
    return (top - bot).dropna().rename("spread")


def _perf(rets: pd.Series, periods_per_year: float) -> dict[str, float]:
    if rets.empty:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan, "total": np.nan}
    eq = (1 + rets).cumprod()
    years = len(rets) / periods_per_year
    return {"cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            "vol": float(rets.std() * np.sqrt(periods_per_year)),
            "sharpe": float(rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else np.nan,
            "max_dd": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1)}


class CrossSectionalRanker:
    """XGBoost regressor on within-date return ranks."""

    def __init__(self, cfg: RankerConfig) -> None:
        self.cfg = cfg
        self.model_: Any = None
        self.backend = "xgboost"
        self.metadata_: dict[str, Any] = {}

    def _make(self) -> Any:
        try:
            from xgboost import XGBRegressor
            self.backend = "xgboost"
            return XGBRegressor(**self.cfg.params, random_state=42)
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.backend = "sklearn_hgb"
            return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=42)

    def _train_rows(self, panel: pd.DataFrame) -> pd.DataFrame:
        d = panel.dropna(subset=["target"])
        d = d[d["turnover_cr"] >= self.cfg.min_turnover_cr]
        dates = np.sort(d["date"].unique())
        keep = set(dates[:: self.cfg.train_stride])
        return d[d["date"].isin(keep)]

    def walk_forward(self, panel: pd.DataFrame) -> dict[str, Any]:
        """Out-of-sample evaluation. Returns metrics, per-year table and equity curves."""
        cfg = self.cfg
        labelled = panel.dropna(subset=["target"]).reset_index(drop=True)
        labelled = labelled[labelled["turnover_cr"] >= cfg.min_turnover_cr].reset_index(drop=True)
        splitter = PurgedWalkForwardSplit(cfg.n_splits, cfg.horizon_days + 5, cfg.min_train_dates)
        oos = []
        for k, (tr, te) in enumerate(splitter.split(labelled["date"])):
            train = self._train_rows(labelled.iloc[tr])
            if len(train) < 1000:
                continue
            model = self._make().fit(_X(train), train["target"])
            test = labelled.iloc[te].copy()
            test["pred"] = model.predict(_X(test))
            test["fold"] = k
            oos.append(test)
            logger.info("Ranker fold %d: train %d rows to %s, test %s..%s", k, len(train),
                        train["date"].max().date(), test["date"].min().date(), test["date"].max().date())
        if not oos:
            raise ValueError("Not enough history for walk-forward validation (needs ~3 years)")
        oos_df = pd.concat(oos, ignore_index=True)
        return self._grade(oos_df)

    def _grade(self, oos: pd.DataFrame) -> dict[str, Any]:
        cfg = self.cfg
        ic = _daily_ic(oos)
        spread = _decile_spread(oos)
        n_eff = max(len(ic) / cfg.horizon_days, 1.0)
        ic_t = float(ic.mean() / ic.std() * np.sqrt(n_eff)) if ic.std() > 0 else 0.0

        # Portfolio: rebalance every `horizon` sessions into the top_n liquid names.
        dates = np.sort(oos["date"].unique())[:: cfg.horizon_days]
        rows = []
        for d in dates:
            day = oos[(oos["date"] == d) & oos["fwd_ret"].notna()]
            if len(day) < cfg.top_n:
                continue
            picks = day.nlargest(cfg.top_n, "pred")
            idx_ret = float(day["idx_fwd"].iloc[0]) if "idx_fwd" in day and pd.notna(day["idx_fwd"].iloc[0]) else np.nan
            risk_on = bool(day["idx_above_200"].iloc[0]) if pd.notna(day["idx_above_200"].iloc[0]) else True
            strat = float(picks["fwd_ret"].mean()) - cfg.cost_pct
            rows.append({"date": pd.Timestamp(d), "strategy": strat, "strategy_regime": strat if risk_on else 0.0,
                         "nifty": idx_ret, "universe": float(day["fwd_ret"].mean())})
        port = pd.DataFrame(rows).set_index("date")
        ppy = 252 / cfg.horizon_days
        perf = {k: _perf(port[k].dropna(), ppy) for k in ("strategy", "strategy_regime", "nifty", "universe")}
        beat = float((port["strategy"] > port["nifty"]).mean()) if port["nifty"].notna().any() else np.nan

        years = pd.DataFrame({"ic": ic, "spread": spread})
        years["year"] = years.index.year
        by_year = years.groupby("year").agg(ic=("ic", "mean"), spread=("spread", "mean"), days=("ic", "size"))
        py = port.assign(year=port.index.year).groupby("year")
        by_year["strategy_%"] = py["strategy"].apply(lambda r: (1 + r).prod() - 1) * 100
        by_year["nifty_%"] = py["nifty"].apply(lambda r: (1 + r.dropna()).prod() - 1) * 100
        by_year["spread"] = by_year["spread"] * 100
        by_year = by_year.round({"ic": 3, "spread": 2, "strategy_%": 1, "nifty_%": 1})
        by_year = by_year.rename(columns={"spread": "top-bottom %"})

        pos_years = float((by_year["top-bottom %"] > 0).mean()) if len(by_year) else 0.0
        # Calibration: how stocks in each predicted decile actually did (out of sample).
        cal = oos.dropna(subset=["fwd_ret"]).copy()
        cal["decile"] = (cal.groupby("date")["pred"].rank(pct=True) * 10).clip(upper=9.999).astype(int) + 1
        calibration = cal.groupby("decile").agg(avg_excess_pct=("fwd_excess", "mean"),
                                                beat_median_pct=("fwd_excess", lambda x: (x > 0).mean()),
                                                observations=("fwd_excess", "size"))
        calibration[["avg_excess_pct", "beat_median_pct"]] *= 100
        calibration = calibration.round(2).reset_index()
        checks = {
            f"Mean IC >= {cfg.min_ic}": ic.mean() >= cfg.min_ic,
            f"IC t-stat >= {cfg.min_ic_tstat}": ic_t >= cfg.min_ic_tstat,
            f"Top-bottom spread positive in >= {cfg.min_positive_years_pct:.0%} of years":
                pos_years >= cfg.min_positive_years_pct and len(by_year) >= 2,
        }
        return {"mean_ic": float(ic.mean()), "ic_tstat": ic_t, "ic_hit": float((ic > 0).mean()),
                "spread_mean": float(spread.mean()), "positive_years": pos_years, "portfolio": port,
                "calibration": calibration,
                "perf": perf, "beat_nifty": beat, "by_year": by_year.reset_index(), "checks": checks,
                "validated": all(checks.values()), "oos_start": str(pd.Timestamp(oos["date"].min()).date()),
                "oos_end": str(pd.Timestamp(oos["date"].max()).date()), "n_oos_dates": int(oos["date"].nunique()),
                "oos": oos[[c for c in ("date", "ticker", "pred", "fwd_ret", "idx_fwd", "idx_above_200", "vol_21",
                            "turnover_cr") if c in oos]].reset_index(drop=True)}

    def fit(self, panel: pd.DataFrame) -> "CrossSectionalRanker":
        train = self._train_rows(panel)
        self.model_ = self._make().fit(_X(train), train["target"])
        self.metadata_ = {"trained_at": datetime.now(timezone.utc).isoformat(), "rows": len(train),
                          "train_end": str(pd.Timestamp(train["date"].max()).date())}
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Ranker is not fitted")
        return self.model_.predict(_X(rows))

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame | None:
        """Per-feature contributions to each prediction (SHAP), or None if unavailable."""
        if self.backend != "xgboost" or self.model_ is None:
            return None
        import xgboost as xgb
        contrib = self.model_.get_booster().predict(xgb.DMatrix(_X(rows)), pred_contribs=True)
        return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=rows.index)

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model_, "backend": self.backend, "meta": {**self.metadata_, **(extra or {})}}, path)

    @classmethod
    def load(cls, cfg: RankerConfig, path: Path) -> "CrossSectionalRanker":
        blob = joblib.load(path)
        r = cls(cfg)
        r.model_, r.backend, r.metadata_ = blob["model"], blob["backend"], blob["meta"]
        return r


def explain(row: pd.Series, contrib: pd.Series | None, direction: int, k: int = 3) -> list[str]:
    """Plain-language reasons: the features that pushed the score in ``direction`` (+1 up, -1 down)."""
    def describe(feat: str) -> str:
        base = feat[2:] if feat.startswith("r_") else feat
        label = LABELS.get(base, base)
        val = row.get(feat)
        if feat.startswith("r_") and pd.notna(val):
            if val >= 0.5:
                return f"{label}: top {max(1, round(100 * (1 - val)))}% of stocks"
            return f"{label}: bottom {max(1, round(100 * val))}% of stocks"
        if feat == "idx_above_200":
            return "Nifty above its 200-day average" if val == 1 else "Nifty below its 200-day average"
        if feat == "breakout_125":
            return "fresh 125-day breakout" if val == 1 else "no recent breakout"
        if feat == "breadth_200" and pd.notna(val):
            return f"market breadth: {100 * val:.0f}% of stocks above 200-day average"
        return label

    if contrib is not None:
        # Market-wide features are identical for every stock that day, so they don't explain why
        # THIS stock ranks where it does; reasons come from stock-specific features only.
        contrib = contrib.drop(labels=[f for f in MARKET if f in contrib.index])
        order = contrib.sort_values(ascending=direction < 0)
        feats = [f for f, v in order.items() if v * direction > 0][:k]
    else:  # fallback: the most extreme ranked features in the right direction
        ranks = row[[f"r_{f}" for f in RANKED]].astype(float)
        feats = list((ranks - 0.5).mul(direction).sort_values(ascending=False).index[:k])
    return [describe(f) for f in feats]
