"""Predictive classifier for P(intraday return > 0).

Target
------
For session *t*: ``y_t = 1 if Close_t / Open_t - 1 > 0 else 0`` (open-to-close).

Leakage controls
----------------
1. **Feature timing** - every feature for row *t* uses information available
   at the *open* of *t*: end-of-day features are computed on day *t-1* and
   shifted forward one bar; the only same-day input is ``Open_t`` (the gap),
   which is known at the open. On NSE the pre-open call auction's IEP *is*
   the opening price, so at inference (09:08-09:15 IST) the gap feature is
   exact rather than a proxy.
2. **Train/serve parity** - inference appends a synthetic "today" row and
   runs the *same* ``build_features`` function used in training.
3. **Purged walk-forward CV** - folds split on calendar dates (not rows, so
   different tickers on the same day never straddle train/test), train on
   the past only, and drop an embargo of dates before each test block to
   limit leakage from overlapping rolling windows.
4. Imputation lives inside the sklearn ``Pipeline`` so it is fitted on
   training folds only.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from config.config import IndicatorConfig, ModelConfig
from screeners.screener import atr, ema, relative_volume, rsi, sma

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "gap_pct", "gap_atr", "ret_1d", "ret_5d", "ret_20d",
    "close_sma_fast", "close_sma_slow", "ema_fast_slow",
    "rsi", "atr_pct", "realized_vol_10", "realized_vol_20", "vol_ratio_10_20",
    "rvol", "volume_zscore", "range_pct", "close_location", "day_of_week",
]


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _end_of_day_features(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Features known at the *close* of each row (to be shifted before use)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    log_ret = np.log(c).diff()
    a = atr(h, l, c, cfg.atr_period)
    rv10 = log_ret.rolling(10, min_periods=10).std() * np.sqrt(252)
    rv20 = log_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_mean = v.rolling(20, min_periods=20).mean()
    vol_std = v.rolling(20, min_periods=20).std()
    rng = h - l
    return pd.DataFrame({
        "ret_1d": c.pct_change(1),
        "ret_5d": c.pct_change(5),
        "ret_20d": c.pct_change(20),
        "close_sma_fast": c / sma(c, cfg.sma_fast) - 1,
        "close_sma_slow": c / sma(c, cfg.sma_slow) - 1,
        "ema_fast_slow": ema(c, cfg.ema_fast) / ema(c, cfg.ema_slow) - 1,
        "rsi": rsi(c, cfg.rsi_period),
        "atr_pct": a / c,
        "realized_vol_10": rv10,
        "realized_vol_20": rv20,
        "vol_ratio_10_20": rv10 / rv20,
        "rvol": relative_volume(v, cfg.rvol_lookback),
        "volume_zscore": (v - vol_mean) / vol_std,
        "range_pct": rng / c,
        "close_location": ((c - l) / rng).where(rng > 0, 0.5),
        "_close": c,
        "_atr": a,
    }, index=df.index)


def build_features(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Build the model feature matrix for one ticker's daily OHLCV.

    Row *t* contains only information available at the open of session *t*.
    """
    prior = _end_of_day_features(df, cfg).shift(1)
    feats = prior.drop(columns=["_close", "_atr"])
    feats["gap_pct"] = df["Open"] / prior["_close"] - 1
    feats["gap_atr"] = (df["Open"] - prior["_close"]) / prior["_atr"]
    feats["day_of_week"] = df.index.dayofweek.astype(float)
    return feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Open-to-close return and binary label (NaN where the session is incomplete)."""
    intraday_ret = df["Close"] / df["Open"] - 1
    label = (intraday_ret > 0).astype(float).where(intraday_ret.notna())
    return pd.DataFrame({"intraday_return": intraday_ret, "target": label}, index=df.index)


def build_training_dataset(histories: Mapping[str, pd.DataFrame], cfg: IndicatorConfig
                           ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pool all tickers into (X, y, meta), sorted by date.

    Warm-up rows (any NaN feature) are dropped. ``meta`` holds ``date``,
    ``ticker`` and ``intraday_return`` aligned with X.
    """
    frames = []
    for ticker, df in histories.items():
        if len(df) < cfg.sma_slow + 25:
            logger.debug("%s: too short for training (%d bars)", ticker, len(df))
            continue
        block = pd.concat([build_features(df, cfg), build_target(df)], axis=1)
        block["ticker"], block["date"] = ticker, block.index
        frames.append(block)
    if not frames:
        raise ValueError("No ticker had enough history to build a training set")
    data = pd.concat(frames, ignore_index=True).dropna(subset=[*FEATURE_COLUMNS, "target"])
    data = data.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    return data[FEATURE_COLUMNS], data["target"].astype(int), data[["date", "ticker", "intraday_return"]]


def build_inference_row(history: pd.DataFrame, open_price: float, as_of: date,
                        cfg: IndicatorConfig) -> pd.DataFrame:
    """Feature row for today, using the NSE pre-open IEP as the open.

    Args:
        history: Completed daily sessions only (strictly before ``as_of``).
        open_price: Today's pre-open indicative equilibrium price.
        as_of: Today's session date.

    Raises:
        ValueError: If ``history`` contains ``as_of`` or later (lookahead).
    """
    ts = pd.Timestamp(as_of)
    if not history.empty and history.index.max() >= ts:
        raise ValueError("history must only contain sessions before as_of")
    today = pd.DataFrame({"Open": [open_price], "High": [np.nan], "Low": [np.nan],
                          "Close": [np.nan], "Volume": [np.nan]}, index=pd.DatetimeIndex([ts]))
    return build_features(pd.concat([history, today]), cfg).iloc[[-1]]


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
class PurgedWalkForwardSplit:
    """Expanding-window walk-forward splitter over unique dates with an embargo.

    Args:
        n_splits: Number of test blocks.
        embargo: Dates removed from the end of each training window.
        min_train_dates: Dates reserved before the first test block.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 5, min_train_dates: int = 250) -> None:
        self.n_splits, self.embargo, self.min_train_dates = n_splits, embargo, min_train_dates

    def split(self, dates: pd.Series | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` row positions."""
        d = pd.to_datetime(pd.Series(dates)).to_numpy()
        unique = np.unique(d)
        test_size = (len(unique) - self.min_train_dates) // self.n_splits
        if test_size < 1:
            raise ValueError(f"Not enough dates ({len(unique)}) for {self.n_splits} splits "
                             f"after {self.min_train_dates} warm-up dates")
        for k in range(self.n_splits):
            start = self.min_train_dates + k * test_size
            stop = start + test_size if k < self.n_splits - 1 else len(unique)
            train_end = start - self.embargo
            if train_end <= 0:
                continue
            train_idx = np.flatnonzero(d < unique[train_end])
            test_idx = np.flatnonzero((d >= unique[start]) & (d <= unique[stop - 1]))
            yield train_idx, test_idx


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class MomentumClassifier:
    """XGBoost (fallback: sklearn HistGradientBoosting) wrapped in a Pipeline."""

    def __init__(self, cfg: ModelConfig, feature_columns: list[str] | None = None) -> None:
        self.cfg = cfg
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.pipeline_: Pipeline | None = None
        self.metadata_: dict[str, Any] = {}

    def _make_pipeline(self) -> Pipeline:
        try:
            from xgboost import XGBClassifier
            estimator: Any = XGBClassifier(**self.cfg.xgb_params, random_state=self.cfg.random_state)
            backend = "xgboost"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            logger.warning("xgboost not installed; falling back to HistGradientBoostingClassifier")
            estimator = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                                       random_state=self.cfg.random_state)
            backend = "sklearn_hgb"
        self.metadata_["backend"] = backend
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", estimator)])

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> dict[str, Any]:
        """Purged walk-forward evaluation. Returns per-fold and mean metrics."""
        splitter = PurgedWalkForwardSplit(self.cfg.n_splits, self.cfg.embargo_days, self.cfg.min_train_dates)
        folds: list[dict[str, Any]] = []
        X = X[self.feature_columns]
        for k, (tr, te) in enumerate(splitter.split(dates)):
            if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
                logger.warning("Fold %d skipped: single-class labels", k)
                continue
            model = self._make_pipeline().fit(X.iloc[tr], y.iloc[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            yt = y.iloc[te]
            folds.append({
                "fold": k, "n_train": len(tr), "n_test": len(te),
                "test_start": str(pd.Timestamp(dates.iloc[te].min()).date()),
                "test_end": str(pd.Timestamp(dates.iloc[te].max()).date()),
                "auc": roc_auc_score(yt, p),
                "log_loss": log_loss(yt, p, labels=[0, 1]),
                "brier": brier_score_loss(yt, p),
                "accuracy": accuracy_score(yt, p >= 0.5),
                "precision_at_threshold": precision_score(yt, p >= self.cfg.probability_threshold, zero_division=0),
                "signals_at_threshold": int((p >= self.cfg.probability_threshold).sum()),
                "base_rate": float(yt.mean()),
            })
        if not folds:
            raise ValueError("Cross-validation produced no valid folds")
        keys = ["auc", "log_loss", "brier", "accuracy", "precision_at_threshold", "base_rate"]
        summary = {f"mean_{k}": float(np.mean([f[k] for f in folds])) for k in keys}
        for f in folds:
            logger.info("CV fold %d [%s..%s] AUC=%.3f Brier=%.4f Prec@thr=%.3f (base %.3f, n=%d)",
                        f["fold"], f["test_start"], f["test_end"], f["auc"], f["brier"],
                        f["precision_at_threshold"], f["base_rate"], f["n_test"])
        logger.info("CV mean AUC=%.3f Brier=%.4f", summary["mean_auc"], summary["mean_brier"])
        return {"folds": folds, "summary": summary}

    def fit(self, X: pd.DataFrame, y: pd.Series, cv_report: dict[str, Any] | None = None,
            train_end: date | None = None) -> "MomentumClassifier":
        """Fit on all supplied rows (call after cross-validation)."""
        self.pipeline_ = self._make_pipeline().fit(X[self.feature_columns], y)
        self.metadata_.update({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_end": str(train_end) if train_end else None,
            "n_rows": int(len(X)), "feature_columns": self.feature_columns,
            "cv_summary": (cv_report or {}).get("summary"),
        })
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(intraday return > 0) for each row."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted")
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        return self.pipeline_.predict_proba(X[self.feature_columns])[:, 1]

    def is_stale(self, max_age_days: int) -> bool:
        """True if the model is older than ``max_age_days`` or has no timestamp."""
        ts = self.metadata_.get("trained_at")
        if not ts:
            return True
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).days > max_age_days

    def save(self, path: Path | None = None) -> Path:
        """Persist the pipeline and metadata with joblib."""
        if self.pipeline_ is None:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(path or self.cfg.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline_, "metadata": self.metadata_}, path)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, cfg: ModelConfig, path: Path | None = None) -> "MomentumClassifier":
        """Load a persisted model. Only load artifacts you created yourself (pickle)."""
        payload = joblib.load(Path(path or cfg.model_path))
        model = cls(cfg, payload["metadata"].get("feature_columns"))
        model.pipeline_, model.metadata_ = payload["pipeline"], payload["metadata"]
        return model


def train_and_evaluate(histories: Mapping[str, pd.DataFrame], ind_cfg: IndicatorConfig,
                       model_cfg: ModelConfig) -> tuple[MomentumClassifier, dict[str, Any]]:
    """Build the dataset, run purged walk-forward CV, then fit on all data.

    Raises:
        ValueError: If there are fewer than ``model_cfg.min_train_rows`` rows.
    """
    X, y, meta = build_training_dataset(histories, ind_cfg)
    if len(X) < model_cfg.min_train_rows:
        raise ValueError(f"Only {len(X)} training rows (< {model_cfg.min_train_rows})")
    logger.info("Training set: %d rows, %d tickers, %s..%s, base rate %.3f",
                len(X), meta["ticker"].nunique(), meta["date"].min().date(),
                meta["date"].max().date(), y.mean())
    model = MomentumClassifier(model_cfg)
    report = model.cross_validate(X, y, meta["date"])
    model.fit(X, y, report, train_end=meta["date"].max().date())
    return model, report
