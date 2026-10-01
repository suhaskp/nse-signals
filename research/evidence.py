"""How much to trust a historical percentage: sample size, a 95% range, and a plain label.

The same "52%" means little from 20 trades and a lot from 2,000. Every hit rate shown in the dashboard should
come with this.
"""
from __future__ import annotations

import math


def strength(n: int) -> str:
    """LOW < 30 trades <= PRELIMINARY < 100 <= MEDIUM < 300 <= HIGH."""
    if n < 30:
        return "LOW"
    if n < 100:
        return "PRELIMINARY"
    if n < 300:
        return "MEDIUM"
    return "HIGH"


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion ``p`` (0-1) observed over ``n`` trials."""
    if n <= 0:
        return (math.nan, math.nan)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def label(p_pct: float, n: int) -> str:
    """e.g. '52% (95% range 45-59%, 180 trades, evidence MEDIUM)'."""
    if n <= 0 or p_pct != p_pct:
        return "no history"
    lo, hi = wilson(p_pct / 100, n)
    return f"{p_pct:.0f}% (95% range {100 * lo:.0f}–{100 * hi:.0f}%, {n:,} trades, evidence {strength(n)})"
