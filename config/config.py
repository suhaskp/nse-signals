"""System configuration for the NSE (India) pre-open screening pipeline.

Every tunable parameter lives here as an immutable (frozen) dataclass.
Values are resolved in increasing order of precedence:

1. Dataclass defaults (this file)
2. A YAML file (see ``config/settings.example.yaml``)
3. Environment variables (secrets and a few operational knobs)
4. Programmatic overrides passed to :func:`load_config` (e.g. CLI flags)

Secrets (API keys) are read from the environment only and are excluded
from ``repr`` so they never leak into logs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Liquid NSE large caps (plain NSE symbols, no ".NS" suffix). Index membership
# changes over time: check the current NIFTY 50 list before relying on this.
DEFAULT_TICKERS: tuple[str, ...] = (
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "BHARTIARTL", "SBIN", "LT",
    "ITC", "HINDUNILVR", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "M&M", "TITAN", "ULTRACEMCO", "NTPC", "HCLTECH",
)

SUPPORTED_PROVIDERS = {"yfinance", "synthetic"}
SUPPORTED_PREOPEN = {"nse", "kite", "synthetic"}

# NSE equity tick sizes by price band: (upper price bound, tick). Verify against the
# latest NSE circular; an order rejected for "invalid tick" means this table is stale.
DEFAULT_TICK_TABLE: tuple[tuple[float, float], ...] = (
    (250.0, 0.01), (1_000.0, 0.05), (5_000.0, 0.10),
    (10_000.0, 0.50), (20_000.0, 1.00), (float("inf"), 5.00),
)
SUPPORTED_SIZING = {"fixed_fractional", "volatility_scaled"}


class ConfigError(ValueError):
    """Raised when configuration values are missing or invalid."""


def _parse_hhmm(value: str) -> tuple[int, int]:
    hh, mm = value.split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(value)
    return h, m


@dataclass(frozen=True)
class MarketConfig:
    """NSE session timings (IST) and trading holidays.

    Pre-open runs 09:00-09:15: order entry until ~09:08, then price discovery
    fixes the indicative equilibrium price (IEP), which becomes the open. Run
    the scan between ``price_discovery_end`` and ``session_open``.

    Attributes:
        holidays: ISO dates (``"2026-10-02"``) from NSE's trading-holiday
            circular. Not bundled because the list changes every year.
    """

    exchange: str = "NSE"
    timezone: str = "Asia/Kolkata"
    currency_symbol: str = "₹"
    preopen_start: str = "09:00"
    price_discovery_end: str = "09:08"
    session_open: str = "09:15"
    session_close: str = "15:30"
    holidays: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataConfig:
    """Data-ingestion settings.

    Attributes:
        provider: Daily history source: ``yfinance`` (NSE via ``.NS`` symbols)
            or ``synthetic`` (offline demo / tests).
        preopen_source: Pre-open IEP source: ``nse`` (NSE website JSON,
            free but unofficial), ``kite`` (Zerodha Kite Connect, needs
            credentials) or ``synthetic``.
        nse_preopen_key: NSE pre-open segment (``ALL``, ``NIFTY``, ``FO``...).
        history_lookback_days: Calendar days of daily history (~5 years).
        preopen_history_path: CSV where each day's pre-open quantities are
            stored; pre-open RVOL needs this history to build up.
        max_retries / backoff_*: Exponential backoff settings.
        requests_per_minute: Client-side rate limit (keep NSE traffic low).
    """

    provider: str = "yfinance"
    preopen_source: str = "nse"
    nse_preopen_key: str = "ALL"
    history_lookback_days: int = 1900
    preopen_history_path: Path = Path("data/preopen_history.csv")
    max_retries: int = 4
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    requests_per_minute: int = 30
    kite_api_key: str | None = field(default=None, repr=False)
    kite_access_token: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class IndicatorConfig:
    """Technical indicator windows."""

    sma_fast: int = 20
    sma_slow: int = 50
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    rvol_lookback: int = 20
    min_preopen_history_days: int = 10  # below this, fall back to prior-session RVOL
    use_talib: bool = True  # used only if the TA-Lib C library is installed


@dataclass(frozen=True)
class ScreenerConfig:
    """Momentum screening thresholds in INR (all must pass).

    ``min_avg_turnover_cr`` is the 20-day average traded value in crore
    (1 crore = 1,00,00,000). Turnover compares liquidity fairly across
    ₹100 and ₹10,000 stocks, unlike share volume.
    ``max_prev_close_mismatch_pct`` rejects a ticker when the pre-open feed's
    previous close disagrees with history (stale feed or corporate action).
    """

    min_price: float = 50.0
    max_price: float = 50_000.0
    min_avg_turnover_cr: float = 25.0
    min_gap_pct: float = 1.0
    max_gap_pct: float = 8.0
    min_rvol: float = 1.5
    rsi_min: float = 50.0
    rsi_max: float = 80.0
    require_above_sma_slow: bool = True
    require_ema_trend: bool = True
    min_history_bars: int = 60
    max_prev_close_mismatch_pct: float = 2.0


@dataclass(frozen=True)
class BreakoutConfig:
    """End-of-day 125-day breakout / breakdown scans (Chartink guide logic).

    BUY  (bullish): close > max(high of previous ``lookback`` days)
                    AND volume > ``volume_multiple`` x SMA(volume, lookback) as of yesterday
                    AND RSI(14) < ``bull_rsi_max``
    SELL (bearish): close < min(low of previous ``lookback`` days)
                    AND volume rule (see ``bear_volume_rule``)
                    AND RSI(14) > ``bear_rsi_min``

    ``bear_volume_rule``: ``below_avg`` reproduces the saved Chartink scan in the guide
    (volume < SMA(volume, 125)); ``above_multiple`` requires a high-volume breakdown
    (volume > volume_multiple x SMA), the mirror of the bullish rule. The guide's text
    and screenshot disagree, so both are available.
    """

    lookback: int = 125
    volume_multiple: float = 2.0
    bull_rsi_max: float = 70.0
    bear_rsi_min: float = 30.0
    bear_volume_rule: str = "below_avg"
    min_avg_turnover_cr: float = 10.0
    # Backtest of the same rules: next-day open entry, ATR stop, R-multiple target, time exit.
    backtest_max_hold_days: int = 20
    # Breakout trades are held overnight (delivery/CNC): STT is 0.1% on both buy and
    # sell, so costs are higher than intraday. Set from your broker's charge sheet.
    est_round_trip_cost_pct: float = 0.0025
    eod_ready_time: str = "16:00"  # after this IST, today's daily bar is treated as complete


@dataclass(frozen=True)
class BuyQualityConfig:
    """Quality-filtered BUY: every gate must pass. See screeners/buy_quality.py."""

    breakout_lookback: int = 55          # close above the prior N-day high
    min_close_location: float = 0.6      # close in the top 40% of the day's range
    max_extension_atr: float = 1.0       # at most 1 ATR beyond the breakout level (no chasing)
    volume_window: int = 50
    volume_multiple: float = 1.5         # volume >= 1.5x the prior 50-day average
    min_turnover_cr: float = 10.0        # 20-day average turnover, Rs crore
    sma200_slope_days: int = 20          # 200-day average must be higher than 20 sessions ago
    require_relative_strength: bool = True  # 3-month return above the Nifty's
    regime_required: bool = True         # no new BUYs while the Nifty is below its 200-day average
    regime_mode: str = "binary"          # "binary": BUY only above the 200-day; "three_state": also in a NEUTRAL
                                         # market (Nifty within 3% below its 200-day, or breadth recovering) at half size
    neutral_band_pct: float = 3.0
    neutral_breadth_gain: float = 0.05   # breadth up >= 5 points over 20 sessions counts as recovering
    min_breadth: float = 0.0             # optional: share of stocks above 200-day (0 = off)
    atr_period: int = 14
    stop_buffer_atr: float = 0.5         # stop sits 0.5 ATR below the breakout level
    min_stop_atr: float = 1.5            # ...but never closer than 1.5 ATR to the entry (daily noise)
    max_stop_atr: float = 2.5            # skip if the stop would be wider than 2.5 ATR
    max_stop_pct: float = 0.10           # ...or wider than 10% of the price
    resistance_lookback: int = 252       # the 52-week high caps the target when it is overhead
    target_max_atr: float = 6.0          # target no further than ~a typical move over the holding period
    min_reward_risk: float = 2.0
    entry_max_gap_pct: float = 2.0       # next open at most 2% above the signal close
    min_model_score: float = 0.5         # when the ranker is validated: model score >= 50/100
    max_hold_days: int = 30              # backtest/tracking time exit
    watch_expiry_sessions: int = 10      # a setup that has not broken out after this many sessions on watch expires


@dataclass(frozen=True)
class CryptoConfig:
    """Crypto (Binance spot, paper only). Same rules as NSE, adapted to crypto: BTC is the market, 24/7 trading,
    daily candles close at 00:00 UTC, fractional quantities, higher costs and correlations."""

    enabled: bool = True
    universe_size: int = 80
    min_quote_volume_usdt: float = 20_000_000      # 24h volume floor for the universe
    benchmark: str = "BTCUSDT"
    account_equity_usdt: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.20
    max_open_positions: int = 4
    max_portfolio_heat_pct: float = 0.03
    max_correlation: float = 0.9                  # coins move together far more than stocks
    cost_pct: float = 0.003                       # ~0.1% fee per side + slippage, round trip
    history_days: int = 1500
    refresh_minutes: int = 15
    mover_min_gain: float = 0.08                  # "mover day" for the historical test
    # quality BUY rules, crypto-tuned (turnover unit = 1e7 USDT, so 2.0 = $20M a day)
    breakout_lookback: int = 55
    volume_multiple: float = 1.5
    min_turnover: float = 2.0
    max_stop_pct: float = 0.20
    entry_max_gap_pct: float = 3.0


def _default_ranker_params() -> dict[str, Any]:
    return {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 50, "reg_lambda": 5.0,
            "tree_method": "hist", "n_jobs": -1}


@dataclass(frozen=True)
class RankerConfig:
    """Cross-sectional ranking model: which stocks should beat the rest over ``horizon_days``.

    Validation gate (all must hold out-of-sample before suggestions are marked validated):
    mean daily rank IC >= ``min_ic``, IC t-stat >= ``min_ic_tstat`` (overlap-adjusted), and the
    top-minus-bottom decile spread positive in at least ``min_positive_years_pct`` of test years.
    """

    horizon_days: int = 10
    top_n: int = 10
    n_splits: int = 6
    min_train_dates: int = 500
    train_stride: int = 5          # train on every 5th date to cut overlap between labels
    min_turnover_cr: float = 10.0
    cost_pct: float = 0.0025       # round trip per rebalance (delivery)
    min_ic: float = 0.02
    min_ic_tstat: float = 2.0
    min_positive_years_pct: float = 0.6
    risk_off_buy: str = "flag"     # "flag" lowers conviction when Nifty < 200DMA; "block" hides BUYs
    max_entry_gap_pct: float = 3.0  # morning check: skip ideas that open more than this far from the close
    optimize: bool = True           # weekly tuning of holding period / portfolio construction
    tune_horizons: tuple[int, ...] = (10, 20)
    holdout_days: int = 250         # final ~12 months kept unseen for confirming the tuned settings
    params: dict[str, Any] = field(default_factory=_default_ranker_params)


def _default_xgb_params() -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }


@dataclass(frozen=True)
class ModelConfig:
    """ML classifier settings."""

    n_splits: int = 5
    embargo_days: int = 5
    min_train_dates: int = 250
    min_train_rows: int = 1_000
    probability_threshold: float = 0.55
    max_model_age_days: int = 7
    model_dir: Path = Path("artifacts/models")
    model_filename: str = "momentum_classifier.joblib"
    random_state: int = 42
    xgb_params: dict[str, Any] = field(default_factory=_default_xgb_params)

    @property
    def model_path(self) -> Path:
        """Full path of the persisted model artifact."""
        return self.model_dir / self.model_filename


@dataclass(frozen=True)
class RiskConfig:
    """Risk and position-sizing limits.

    Attributes:
        account_equity: Total account equity used for sizing.
        risk_per_trade_pct: Fraction of equity risked between entry and stop.
        atr_stop_multiplier: Stop distance in ATR units (1.5 x ATR).
        reward_risk_ratio: Take-profit distance as a multiple of the stop distance.
        min_reward_risk_ratio: Hard floor; config is rejected below this.
        sizing_method: ``fixed_fractional`` or ``volatility_scaled``.
        target_volatility_pct: For volatility scaling, the dollar value of one
            ATR move per position as a fraction of equity.
        max_position_pct: Cap on a single position's notional / equity.
        max_open_positions: Maximum simultaneous positions.
        max_portfolio_heat_pct: Cap on combined open risk / equity.
        tick_table: NSE price-band tick sizes used to round stops/targets.
        est_round_trip_cost_pct: Estimated intraday round-trip costs as a
            fraction of trade value (brokerage, STT, exchange charges, GST,
            SEBI fee, stamp duty). Set it from your broker's charge sheet.
    """

    account_equity: float = 1_000_000.0  # ₹10 lakh
    risk_per_trade_pct: float = 0.01
    atr_stop_multiplier: float = 1.5
    reward_risk_ratio: float = 2.0
    min_reward_risk_ratio: float = 2.0
    sizing_method: str = "fixed_fractional"
    target_volatility_pct: float = 0.0075
    max_position_pct: float = 0.20
    max_open_positions: int = 5
    max_portfolio_heat_pct: float = 0.04
    max_per_sector: int = 2              # portfolio concentration: at most this many new BUYs per sector
    max_correlation: float = 0.8         # skip a BUY that moves almost in lockstep with one already chosen
    tick_table: tuple[tuple[float, float], ...] = DEFAULT_TICK_TABLE
    est_round_trip_cost_pct: float = 0.0006


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution guardrails.

    The system never places broker orders. "Paper orders" are written to a
    local ledger (``outputs/paper_ledger.csv``) and evaluated after the close.
    Automated order placement in India also falls under SEBI's framework for
    retail algorithmic trading via brokers, which this build does not implement.
    """

    paper_only: bool = True
    record_paper_orders: bool = False   # automatic pre-open scan writes LONG signals to the paper ledger
    kill_switch_path: Path = Path("KILL_SWITCH")
    max_orders_per_day: int = 5


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    market: MarketConfig = field(default_factory=MarketConfig)
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    buy_quality: BuyQualityConfig = field(default_factory=BuyQualityConfig)
    benchmark: str = "^NSEI"            # Nifty 50 index on Yahoo
    universe_file: Path | None = None   # CSV with an "NSE Code" or "Symbol" column
    auto_universe: str = "nifty500"     # download NSE's Nifty 500 list weekly; "none" = use tickers
    superstar_dir: Path = Path("data/superstar")  # drop Trendlyne exports here; newest file is used
    cache_dir: Path = Path("data/cache")
    live_refresh_seconds: int = 300     # dashboard refresh while the market is open
    data_home: Path | None = None       # where records live; default PIPELINE_HOME or ~/NSE_Signals
    rules_freeze_days: int = 75         # keep the rules fixed this long so the forward record measures one version
    preopen_scan_enabled: bool = False  # the experimental pre-open gap scan (retired by default)
    real_money_min_trades: int = 50     # "ready for real money?" checklist thresholds, fixed in advance
    real_money_min_avg_r: float = 0.10
    allow_network_access: bool = False  # False: the dashboard is reachable only from this computer
    simple_view_default: bool = True
    crypto: "CryptoConfig" = field(default_factory=lambda: CryptoConfig())
    superstar_file: Path | None = None  # Trendlyne "Buys by Superstar Investors" export
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def validate(self) -> None:
        """Validate cross-field constraints.

        Raises:
            ConfigError: With every violation listed, so all can be fixed at once.
        """
        e: list[str] = []
        d, i, s, m, r, x = (self.data, self.indicators, self.screener,
                            self.model, self.risk, self.execution)
        mk = self.market

        if not self.tickers:
            e.append("tickers must not be empty")
        if d.provider not in SUPPORTED_PROVIDERS:
            e.append(f"data.provider must be one of {sorted(SUPPORTED_PROVIDERS)}")
        if d.preopen_source not in SUPPORTED_PREOPEN:
            e.append(f"data.preopen_source must be one of {sorted(SUPPORTED_PREOPEN)}")
        if d.preopen_source == "kite" and not (d.kite_api_key and d.kite_access_token):
            e.append("KITE_API_KEY / KITE_ACCESS_TOKEN env vars are required for preopen_source=kite")
        try:
            times = [_parse_hhmm(t) for t in (mk.preopen_start, mk.price_discovery_end,
                                               mk.session_open, mk.session_close)]
            if times != sorted(times):
                e.append("market times must be in order: preopen_start < price_discovery_end "
                         "< session_open < session_close")
        except ValueError:
            e.append("market times must be HH:MM")
        for h in mk.holidays:
            try:
                date.fromisoformat(str(h))
            except ValueError:
                e.append(f"market.holidays entry {h!r} is not an ISO date (YYYY-MM-DD)")
        if d.max_retries < 0 or d.requests_per_minute <= 0:
            e.append("data.max_retries must be >= 0 and data.requests_per_minute > 0")

        if not (1 < i.sma_fast < i.sma_slow) or not (1 < i.ema_fast < i.ema_slow):
            e.append("indicator fast windows must be > 1 and smaller than slow windows")
        if i.rsi_period < 2 or i.atr_period < 2 or i.rvol_lookback < 2:
            e.append("rsi_period, atr_period and rvol_lookback must be >= 2")

        if not (0 <= s.rsi_min < s.rsi_max <= 100):
            e.append("screener RSI bounds must satisfy 0 <= rsi_min < rsi_max <= 100")
        if not (s.min_gap_pct < s.max_gap_pct):
            e.append("screener.min_gap_pct must be < max_gap_pct")
        if not (0 < s.min_price < s.max_price):
            e.append("screener price bounds must satisfy 0 < min_price < max_price")
        if s.min_rvol <= 0 or s.min_avg_turnover_cr < 0:
            e.append("screener.min_rvol must be > 0 and min_avg_turnover_cr >= 0")

        if not (0.5 <= m.probability_threshold < 1.0):
            e.append("model.probability_threshold must be in [0.5, 1.0)")
        if m.n_splits < 2 or m.embargo_days < 0:
            e.append("model.n_splits must be >= 2 and embargo_days >= 0")

        if r.account_equity <= 0:
            e.append("risk.account_equity must be > 0")
        if not (0 < r.risk_per_trade_pct <= 0.05):
            e.append("risk.risk_per_trade_pct must be in (0, 0.05] (5% hard ceiling)")
        if r.atr_stop_multiplier <= 0:
            e.append("risk.atr_stop_multiplier must be > 0")
        if r.min_reward_risk_ratio < 1 or r.reward_risk_ratio < r.min_reward_risk_ratio:
            e.append("risk.reward_risk_ratio must be >= min_reward_risk_ratio (>= 1)")
        if r.sizing_method not in SUPPORTED_SIZING:
            e.append(f"risk.sizing_method must be one of {sorted(SUPPORTED_SIZING)}")
        if not (0 < r.max_position_pct <= 1):
            e.append("risk.max_position_pct must be in (0, 1]")
        if r.max_open_positions < 1:
            e.append("risk.max_open_positions must be >= 1")
        if r.max_portfolio_heat_pct < r.risk_per_trade_pct:
            e.append("risk.max_portfolio_heat_pct must be >= risk_per_trade_pct")
        bounds = [b for b, _ in r.tick_table]
        if not r.tick_table or bounds != sorted(bounds) or any(t <= 0 for _, t in r.tick_table):
            e.append("risk.tick_table must be ascending (upper_bound, tick>0) pairs")
        if not (0 <= r.est_round_trip_cost_pct < 0.01):
            e.append("risk.est_round_trip_cost_pct must be in [0, 0.01)")

        if not x.paper_only:
            e.append("execution.paper_only=False is not supported: this build never places broker orders")

        b = self.breakout
        if b.lookback < 20 or b.volume_multiple <= 0 or not (0 < b.bear_rsi_min < b.bull_rsi_max < 100):
            e.append("breakout: lookback >= 20, volume_multiple > 0 and 0 < bear_rsi_min < bull_rsi_max < 100")
        if b.bear_volume_rule not in {"below_avg", "above_multiple"}:
            e.append("breakout.bear_volume_rule must be 'below_avg' or 'above_multiple'")
        rk = self.ranker
        if rk.horizon_days < 1 or rk.top_n < 1 or rk.n_splits < 2 or rk.risk_off_buy not in {"flag", "block"}:
            e.append("ranker: horizon_days >= 1, top_n >= 1, n_splits >= 2, risk_off_buy in {flag, block}")
        q = self.buy_quality
        if q.min_reward_risk < 2.0:
            e.append("buy_quality.min_reward_risk must be >= 2.0 (minimum 2:1 reward/risk)")
        if q.regime_mode not in {"binary", "three_state"}:
            e.append("buy_quality.regime_mode must be 'binary' or 'three_state'")
        if not (0 < q.min_stop_atr <= q.max_stop_atr) or q.breakout_lookback < 10 or q.volume_multiple <= 0:
            e.append("buy_quality: need 0 < min_stop_atr <= max_stop_atr, breakout_lookback >= 10, volume_multiple > 0")
        if self.auto_universe not in {"nifty500", "none"}:
            e.append("auto_universe must be 'nifty500' or 'none'")
        if self.live_refresh_seconds < 60:
            e.append("live_refresh_seconds must be >= 60 (be gentle with free data sources)")
        if b.backtest_max_hold_days < 1:
            e.append("breakout.backtest_max_hold_days must be >= 1")

        if e:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(e))


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a (partial) dict, rejecting unknown keys."""
    defaults = cls()
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value, default = data[f.name], getattr(defaults, f.name)
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise ConfigError(f"'{f.name}' must be a mapping")
            kwargs[f.name] = _build(type(default), value)
        elif isinstance(default, Path) or f.name.endswith("_file") or f.name == "data_home":
            kwargs[f.name] = Path(value) if value is not None else None
        elif isinstance(default, tuple):
            kwargs[f.name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        elif isinstance(default, dict):
            kwargs[f.name] = {**default, **value}
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to load YAML config files") from exc
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level YAML config must be a mapping")
    return loaded


def _env_overrides() -> dict[str, Any]:
    """Read secrets and operational overrides from the environment."""
    try:
        from dotenv import load_dotenv  # optional dependency
        load_dotenv()
    except ImportError:
        pass
    out: dict[str, Any] = {}
    if key := os.getenv("KITE_API_KEY"):
        out.setdefault("data", {})["kite_api_key"] = key
    if token := os.getenv("KITE_ACCESS_TOKEN"):
        out.setdefault("data", {})["kite_access_token"] = token
    if equity := os.getenv("PIPELINE_ACCOUNT_EQUITY"):
        try:
            out.setdefault("risk", {})["account_equity"] = float(equity)
        except ValueError as exc:
            raise ConfigError("PIPELINE_ACCOUNT_EQUITY must be numeric") from exc
    if tickers := os.getenv("PIPELINE_TICKERS"):
        out["tickers"] = [t for t in tickers.split(",") if t.strip()]
    if level := os.getenv("PIPELINE_LOG_LEVEL"):
        out["log_level"] = level
    return out


def _anchor_paths(cfg: "AppConfig") -> "AppConfig":
    """Put every relative record/cache path under the data home, outside the application folder."""
    from storage import default_home
    home = Path(cfg.data_home) if cfg.data_home else default_home()
    fix = lambda p: p if p is None or Path(p).is_absolute() else home / p  # noqa: E731
    return replace(cfg, data_home=home, output_dir=fix(cfg.output_dir), cache_dir=fix(cfg.cache_dir),
                   log_dir=fix(cfg.log_dir), superstar_dir=fix(cfg.superstar_dir),
                   model=replace(cfg.model, model_dir=fix(cfg.model.model_dir)),
                   data=replace(cfg.data, preopen_history_path=fix(cfg.data.preopen_history_path)))


def normalize_symbol(symbol: str) -> str:
    """Canonical NSE symbol: upper-case, without ``.NS`` suffix or ``NSE:`` prefix."""
    s = symbol.strip().upper()
    if s.startswith("^"):
        return s  # index symbol (e.g. ^NSEI), passed to Yahoo unchanged
    if s.startswith("NSE:"):
        s = s[4:]
    if s.endswith(".NS"):
        s = s[:-3]
    return s


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load, merge and validate the application configuration.

    Args:
        path: Optional YAML file with partial overrides of the defaults.
        overrides: Optional nested dict applied last (e.g. from CLI flags).

    Returns:
        A validated, immutable :class:`AppConfig`.

    Raises:
        ConfigError: If the file is malformed or any value is invalid.
    """
    if path is None and Path("config/settings.yaml").exists():
        path = "config/settings.yaml"  # personal settings (equity, risk %, universe), picked up automatically
    raw: dict[str, Any] = _read_yaml(Path(path)) if path else {}
    raw = _deep_merge(raw, _env_overrides())
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg: AppConfig = _build(AppConfig, raw)
    tickers = tuple(dict.fromkeys(normalize_symbol(t) for t in cfg.tickers if t.strip()))
    cfg = replace(cfg, tickers=tickers)
    cfg = _anchor_paths(cfg)
    cfg.validate()
    logger.debug("Configuration loaded: %s", cfg)
    return cfg
