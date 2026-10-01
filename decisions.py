"""Rule-driven explanations shared by the decision card, the daily briefing and the decision journal.

Everything here is derived from the BUY checks; nothing adds or changes a rule. Keeping it in one place means
the card, the briefing and the journal can never give different reasons for the same decision.
"""
from __future__ import annotations

import pandas as pd

from config.config import BuyQualityConfig


def hard_checks(r: pd.Series, q: BuyQualityConfig, data_stale: bool, market: str = "Nifty") -> list[tuple[bool, str]]:
    return [
        (bool(r["g_trend"]), f"Trend: above the 50-day and a rising 200-day average, beating {market}"),
        (bool(r["g_liquidity"]), "Liquidity"),
        (bool(r["g_breakout"]), f"Breakout: {r['Breakout status']} (level ₹{r['Breakout level']:,.2f})"),
        (bool(r["g_close_strength"]), "Strong close"),
        (bool(r["g_not_extended"]), "Not extended beyond the breakout"),
        (bool(r["g_volume"]), f"Volume {r['Volume x avg']:.1f}× average (needs {q.volume_multiple:g}×)"),
        (bool(r["g_stop_ok"]), "A valid ATR stop exists"),
        (bool(r["g_reward_risk"]) if r["g_breakout"] else False,
         f"Reward/risk {r['R:R']:.1f}:1 (needs {q.min_reward_risk:g}:1)" if r["g_breakout"] else
         "Reward/risk: measured once it breaks out"),
        (not data_stale, "Data is fresh"),
        (bool(r["g_regime"]), f"Market regime allows BUYs (your rule; now: {r['Regime state']})"),
    ]


def becomes_buy(r: pd.Series, q: BuyQualityConfig, data_stale: bool, market: str = "Nifty") -> list[str]:
    out = []
    if not r["g_breakout"]:
        out.append(f"a close above ₹{r['Breakout level']:,.2f} ({-r['To breakout %']:+.1f}% from here)")
    if not r["g_volume"]:
        out.append(f"volume of at least {q.volume_multiple:g}× average on that day")
    if not r["g_close_strength"]:
        out.append("a close in the upper part of the day's range")
    if not r["g_not_extended"]:
        out.append(f"a pullback towards ₹{r['Breakout level']:,.2f} (it is too far above the breakout to chase)")
    if r["g_breakout"] and not r["g_reward_risk"]:
        out.append("more room to the target (the 52-week high caps it right now)")
    if not r["g_trend"]:
        out.append(f"the trend repairing (price above a rising 50-day and 200-day average, beating {market})")
    if not r["g_regime"]:
        out.append(f"the market regime turning permissible ({market} back above its 200-day average)")
    if data_stale:
        out.append("fresh price data")
    return out


def becomes_avoid(r: pd.Series, market: str = "Nifty") -> list[str]:
    out = [f"a close below its 50-day average (₹{r['SMA50']:,.2f}): the trend breaks",
           f"its 3-month strength vs {market} turning negative"]
    if r["g_breakout"]:
        out.insert(0, f"a close back below ₹{r['Stop']:,.2f}: the breakout has failed")
    return out


def blockers(r: pd.Series, q: BuyQualityConfig, data_stale: bool, market: str = "Nifty") -> list[str]:
    return [t for ok, t in hard_checks(r, q, data_stale, market) if not ok]


def short_blocker(text: str) -> str:
    """'Market regime allows BUYs (your rule; now: off)' -> 'Market regime'."""
    return text.split(":")[0].split(" (")[0].replace(" allows BUYs", "").strip()
