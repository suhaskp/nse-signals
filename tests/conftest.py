"""Shared fixtures: deterministic synthetic OHLCV data."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.config import AppConfig, IndicatorConfig, ModelConfig, RiskConfig, ScreenerConfig
from data_ingestion.data_fetcher import IST


def make_ohlcv(n: int = 400, seed: int = 0, drift: float = 0.0005, vol: float = 0.02,
               end: str = "2026-09-25") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    close = 1000 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1 + rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(np.log(5e5), 0.3, n).round()  # ~₹50cr/day at ₹1,000
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


@pytest.fixture(autouse=True)
def _no_background_thread(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("PIPELINE_BACKGROUND", "0")
    monkeypatch.setenv("PIPELINE_HOME", str(tmp_path_factory.getbasetemp() / "home"))  # never touch ~/NSE_Signals
    monkeypatch.setenv("PIPELINE_ROBUSTNESS", "0")
    monkeypatch.setenv("PIPELINE_SIMPLE_VIEW", "0")  # tests exercise every tab; the simple view has its own test  # weekly robustness backtests are tested directly, not per engine


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 28, 9, 10, tzinfo=IST)


@pytest.fixture
def ind_cfg() -> IndicatorConfig:
    return IndicatorConfig(use_talib=False)


@pytest.fixture
def scr_cfg() -> ScreenerConfig:
    return ScreenerConfig()


@pytest.fixture
def risk_cfg() -> RiskConfig:
    return RiskConfig(account_equity=1_000_000, risk_per_trade_pct=0.01, atr_stop_multiplier=1.5,
                      reward_risk_ratio=2.0, max_position_pct=0.25, max_open_positions=3,
                      max_portfolio_heat_pct=0.025)


@pytest.fixture
def model_cfg(tmp_path) -> ModelConfig:
    return ModelConfig(n_splits=3, embargo_days=5, min_train_dates=120, min_train_rows=200,
                       model_dir=tmp_path, xgb_params={"n_estimators": 40, "max_depth": 2,
                                                       "learning_rate": 0.1, "n_jobs": 1})


@pytest.fixture
def app_cfg(tmp_path) -> AppConfig:
    return AppConfig()
