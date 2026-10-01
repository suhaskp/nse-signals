import pytest

from config.config import ConfigError, load_config


def test_defaults_valid():
    cfg = load_config()
    assert cfg.risk.atr_stop_multiplier == 1.5 and cfg.execution.paper_only


def test_overrides_and_symbol_normalization():
    cfg = load_config(overrides={"tickers": ["reliance.ns", " NSE:tcs", "RELIANCE", "m&m"],
                                 "risk": {"account_equity": 500000}})
    assert cfg.tickers == ("RELIANCE", "TCS", "M&M") and cfg.risk.account_equity == 500000
    assert cfg.market.timezone == "Asia/Kolkata"


@pytest.mark.parametrize("override", [
    {"risk": {"risk_per_trade_pct": 0.2}},
    {"risk": {"reward_risk_ratio": 1.5}},
    {"execution": {"paper_only": False}},
    {"data": {"preopen_source": "kite"}},          # no credentials
    {"market": {"holidays": ["02-10-2026"]}},     # not ISO
    {"market": {"price_discovery_end": "09:20"}},  # after the open
    {"screener": {"rsi_min": 90, "rsi_max": 80}},
    {"unknown_key": 1},
])
def test_invalid_config_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=override)


def test_env_secret_not_in_repr(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "SECRET123")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "TOKEN456")
    cfg = load_config()
    assert "SECRET123" not in repr(cfg) and "TOKEN456" not in repr(cfg)
