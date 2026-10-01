"""App version, refresh heartbeat, and the fixed "ready for real money?" checklist."""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta

import pandas as pd

APP_VERSION = "2026.10.01.1"
STALE_AFTER = timedelta(hours=2)


def heartbeat(intel_rep, live_snap, crypto_rep, now: datetime, nse_trading_day: bool) -> list[dict]:
    """Last successful refresh per engine, and whether it is overdue."""
    in_hours = nse_trading_day and dtime(9, 15) <= now.time() <= dtime(15, 30)
    rows = []
    for name, obj, must_be_fresh in (("NSE research", intel_rep, False), ("Live breakout signals", live_snap, in_hours),
                                     ("Crypto", crypto_rep, crypto_rep is not None)):
        at = getattr(obj, "as_of", None)
        overdue = bool(at is not None and must_be_fresh and now - at > STALE_AFTER)
        rows.append({"engine": name, "last": at, "overdue": overdue})
    return rows


def readiness(forward_results: pd.DataFrame | None, health: dict | None, research: dict | None, version: str,
              min_trades: int, min_avg_r: float) -> list[tuple[bool | None, str]]:
    """The checklist that must be all ✅ before risking real money. None = cannot be assessed yet."""
    res = forward_results if forward_results is not None else pd.DataFrame()
    mine = res[(res.get("rule_version") == version) & res["exit"].notna()] if not res.empty else res
    n = len(mine)
    avg = float(mine["r_multiple"].mean()) if n else float("nan")
    h = (health or {}).get("level")
    rs = research or {}
    stress = (rs.get("stress_verdict") or (None, ""))[0]
    sens = (rs.get("sensitivity_verdict") or (None, ""))[0]
    checks = [
        (n >= min_trades, f"At least {min_trades} closed forward trades on rule version {version} ({n} so far)"),
        ((avg > min_avg_r) if n >= min_trades else None,
         f"Average above {min_avg_r:+.2f}R per closed live trade after costs"
         + (f" (now {avg:+.2f}R)" if n else "")),
        ((h == "ok") if h not in (None, "none") else None, "Live results consistent with the backtest (strategy health)"),
        ((stress != "bad") if stress not in (None, "none") else None, "Not fragile under execution stress"),
        ((sens != "bad") if sens not in (None, "none") else None, "No sign of overfitting in parameter sensitivity"),
    ]
    return checks if research is not None else checks[:3]  # robustness research exists for NSE only


def ready(checks: list[tuple[bool | None, str]]) -> bool:
    return all(ok is True for ok, _ in checks)
