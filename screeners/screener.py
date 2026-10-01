"""Technical indicators and the daily momentum screen.

All indicators are pure pandas (Wilder smoothing for RSI/ATR). If the TA-Lib
C library is installed and ``IndicatorConfig.use_talib`` is True, TA-Lib is
used for RSI/ATR instead; both use Wilder's method, so results agree closely
(pandas uses an EWM seed rather than an SMA seed for the first value).

Timing convention (important for avoiding lookahead):
    Indicators are computed on *completed* daily sessions only. The gap is the
    NSE pre-open IEP versus the last completed close. RVOL is today's pre-open
    matched quantity versus its recent average once enough pre-open history
    has been stored; until then it falls back to the last completed session's
    RVOL.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from collections.abc import Mapping

import numpy as np
import pandas as pd

from config.config import IndicatorConfig, ScreenerConfig
from data_ingestion.preopen import PreOpenQuote

logger = logging.getLogger(__name__)

try:  # optional accelerated backend
    import talib  # type: ignore[import-not-found]
    _HAS_TALIB = True
except ImportError:
    talib = None
    _HAS_TALIB = False


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average (NaN until ``window`` observations exist)."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with span ``window`` (no bias adjustment)."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14, use_talib: bool = False) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100].

    Edge cases: only gains -> 100; flat prices -> 50.
    """
    if use_talib and _HAS_TALIB:
        return pd.Series(talib.RSI(close.to_numpy(dtype=float), timeperiod=period), index=close.index)
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(H-L, |H-prevC|, |L-prevC|). First bar uses H-L."""
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
        use_talib: bool = False) -> pd.Series:
    """Wilder's Average True Range."""
    if use_talib and _HAS_TALIB:
        values = talib.ATR(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=period)
        return pd.Series(values, index=close.index)
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Volume divided by the average of the *previous* ``lookback`` bars.

    The average is shifted by one bar so a bar is never compared with itself.
    """
    baseline = volume.shift(1).rolling(lookback, min_periods=lookback).mean()
    return volume / baseline.replace(0, np.nan)


def gap_pct(current_price: float, prev_close: float) -> float:
    """Percentage gap of ``current_price`` versus ``prev_close``.

    Raises:
        ValueError: If ``prev_close`` is not a positive finite number.
    """
    if not np.isfinite(prev_close) or prev_close <= 0:
        raise ValueError(f"prev_close must be positive, got {prev_close}")
    return (current_price / prev_close - 1.0) * 100.0


def add_indicators(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Return a copy of daily OHLCV with indicator columns appended.

    Columns added: ``SMA_{fast}``, ``SMA_{slow}``, ``EMA_{fast}``, ``EMA_{slow}``,
    ``RSI``, ``ATR``, ``AvgVol``, ``AvgTurnover`` (INR), ``RVOL``.
    """
    out = df.copy()
    c = out["Close"]
    use_talib = cfg.use_talib and _HAS_TALIB
    out[f"SMA_{cfg.sma_fast}"] = sma(c, cfg.sma_fast)
    out[f"SMA_{cfg.sma_slow}"] = sma(c, cfg.sma_slow)
    out[f"EMA_{cfg.ema_fast}"] = ema(c, cfg.ema_fast)
    out[f"EMA_{cfg.ema_slow}"] = ema(c, cfg.ema_slow)
    out["RSI"] = rsi(c, cfg.rsi_period, use_talib)
    out["ATR"] = atr(out["High"], out["Low"], c, cfg.atr_period, use_talib)
    out["AvgVol"] = out["Volume"].rolling(cfg.rvol_lookback, min_periods=cfg.rvol_lookback).mean()
    out["AvgTurnover"] = (c * out["Volume"]).rolling(cfg.rvol_lookback, min_periods=cfg.rvol_lookback).mean()
    out["RVOL"] = relative_volume(out["Volume"], cfg.rvol_lookback)
    return out


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
@dataclass
class ScreenResult:
    """Outcome of screening one ticker. ``failures`` lists every failed rule."""

    ticker: str
    passed: bool
    price: float
    prev_close: float
    gap_pct: float
    rvol: float
    rvol_source: str
    rsi: float
    atr: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    avg_turnover_cr: float
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failures"] = "; ".join(self.failures)
        return d


def evaluate_ticker(ticker: str, daily: pd.DataFrame, quote: PreOpenQuote,
                    ind_cfg: IndicatorConfig, scr_cfg: ScreenerConfig) -> ScreenResult:
    """Apply every momentum rule to one NSE symbol.

    Args:
        daily: *Completed* daily sessions (see ``completed_sessions``).
        quote: Today's pre-open quote (IEP, previous close, pre-open quantity).
    """
    ind = add_indicators(daily, ind_cfg)
    last = ind.iloc[-1]
    hist_close = float(last["Close"])
    price = quote.iep
    # History is the reference close; the feed's figure is only a consistency check.
    try:
        gap = gap_pct(price, hist_close)
    except ValueError:
        gap = float("nan")
    po_rvol = quote.preopen_rvol
    rvol, rvol_src = (po_rvol, "preopen") if np.isfinite(po_rvol) else (float(last["RVOL"]), "prior_session")
    turnover_cr = float(last["AvgTurnover"]) / 1e7

    f: list[str] = []
    s = scr_cfg
    if np.isfinite(quote.prev_close) and quote.prev_close > 0:
        mismatch = abs(quote.prev_close / hist_close - 1) * 100
        if mismatch > s.max_prev_close_mismatch_pct:
            f.append(f"pre-open prev close {quote.prev_close:.2f} vs history {hist_close:.2f} "
                     f"({mismatch:.1f}% apart: stale feed or corporate action)")
    if len(ind) < max(s.min_history_bars, ind_cfg.sma_slow):
        f.append(f"history<{max(s.min_history_bars, ind_cfg.sma_slow)} bars")
    if not (s.min_price <= price <= s.max_price):
        f.append(f"price {price:.2f} outside [{s.min_price:g}, {s.max_price:g}]")
    if not turnover_cr >= s.min_avg_turnover_cr:
        f.append(f"avg turnover ₹{turnover_cr:.1f}cr < ₹{s.min_avg_turnover_cr:g}cr")
    if not (s.min_gap_pct <= gap <= s.max_gap_pct):
        f.append(f"gap {gap:.2f}% outside [{s.min_gap_pct}, {s.max_gap_pct}]")
    if not rvol >= s.min_rvol:
        f.append(f"rvol {rvol:.2f} < {s.min_rvol}")
    if not (s.rsi_min <= last["RSI"] <= s.rsi_max):
        f.append(f"RSI {last['RSI']:.1f} outside [{s.rsi_min}, {s.rsi_max}]")
    sma_slow_v = float(last[f"SMA_{ind_cfg.sma_slow}"])
    ema_f, ema_s = float(last[f"EMA_{ind_cfg.ema_fast}"]), float(last[f"EMA_{ind_cfg.ema_slow}"])
    if s.require_above_sma_slow and not price > sma_slow_v:
        f.append(f"price below SMA{ind_cfg.sma_slow}")
    if s.require_ema_trend and not ema_f > ema_s:
        f.append(f"EMA{ind_cfg.ema_fast} <= EMA{ind_cfg.ema_slow}")

    return ScreenResult(
        ticker=ticker, passed=not f, price=price, prev_close=hist_close, gap_pct=gap,
        rvol=rvol, rvol_source=rvol_src, rsi=float(last["RSI"]), atr=float(last["ATR"]),
        sma_slow=sma_slow_v, ema_fast=ema_f, ema_slow=ema_s,
        avg_turnover_cr=turnover_cr, failures=f,
    )


def scan_universe(histories: Mapping[str, pd.DataFrame],
                  quotes: Mapping[str, PreOpenQuote],
                  ind_cfg: IndicatorConfig, scr_cfg: ScreenerConfig) -> pd.DataFrame:
    """Screen every ticker and return one row per ticker (passed or not)."""
    rows = []
    for ticker, daily in histories.items():
        quote = quotes.get(ticker)
        if quote is None:
            logger.warning("%s: no pre-open quote; skipped", ticker)
            continue
        try:
            rows.append(evaluate_ticker(ticker, daily, quote, ind_cfg, scr_cfg).to_dict())
        except Exception:  # noqa: BLE001 - one bad ticker must not stop the scan
            logger.exception("%s: screening failed", ticker)
    if not rows:
        return pd.DataFrame(columns=list(ScreenResult.__dataclass_fields__))
    scan = pd.DataFrame(rows).sort_values(["passed", "rvol", "gap_pct"], ascending=False)
    return scan.reset_index(drop=True)


def get_daily_candidates(histories: Mapping[str, pd.DataFrame],
                         quotes: Mapping[str, PreOpenQuote],
                         ind_cfg: IndicatorConfig, scr_cfg: ScreenerConfig) -> pd.DataFrame:
    """Return only symbols meeting every momentum criterion, ranked by RVOL then gap."""
    scan = scan_universe(histories, quotes, ind_cfg, scr_cfg)
    candidates = scan[scan["passed"]].reset_index(drop=True) if not scan.empty else scan
    logger.info("Screen: %d/%d tickers passed", len(candidates), len(scan))
    return candidates
