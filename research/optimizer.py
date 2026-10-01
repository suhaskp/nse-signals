"""Tune holding period and portfolio construction without fooling ourselves.

1. Walk-forward out-of-sample predictions are produced for each holding period.
2. The OOS period is split: everything except the last ``holdout_days`` sessions is the
   SELECTION period; the final ``holdout_days`` sessions are the HOLDOUT, never used to choose.
3. Every combination in a small grid (top 10/20, hold-buffer on/off, regime filter on/off,
   equal/inverse-vol weights) is simulated with turnover-based costs.
4. The best combination by selection-period Sharpe (with positive excess return over the
   Nifty and a validated model) is ADOPTED only if, on the holdout, it matches or beats the
   current baseline on BOTH Sharpe ratio and return. Otherwise the baseline is kept.
   Whether the adopted setup beat simply holding the Nifty is reported separately.

Testing ~32 combinations on the same data inflates the best result by luck, which is why
the holdout confirmation, not the selection score, decides.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from config.config import RankerConfig
from models.portfolio import BASELINE, Settings, metrics, relabel, simulate
from models.ranker import CrossSectionalRanker

logger = logging.getLogger(__name__)


def grid(horizon: int) -> list[Settings]:
    return [Settings(horizon, n, b, r, w)
            for n, b, r, w in product((10, 20), (None, 0.8), (False, True), ("equal", "inv_vol"))]


def optimize(panel: pd.DataFrame, histories: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame | None,
             cfg: RankerConfig, status=lambda _m: None) -> dict[str, Any]:
    reports: dict[int, dict] = {}
    rows = []
    holdout_start = None
    for h in cfg.tune_horizons:
        status(f"Tuning: walk-forward test of the {h}-day holding period")
        p = panel if h == cfg.horizon_days else relabel(panel, histories, benchmark, h)
        rep = CrossSectionalRanker(replace(cfg, horizon_days=h)).walk_forward(p)
        oos = rep.pop("oos")
        reports[h] = rep
        dates = np.sort(oos["date"].unique())
        if len(dates) <= cfg.holdout_days + 120:
            logger.warning("Only %d OOS dates: too few for a separate holdout; tuning skipped", len(dates))
            return {"adopted": False, "chosen": BASELINE, "table": pd.DataFrame(), "reports": reports,
                    "reason": "Not enough out-of-sample history for a separate holdout year; using the baseline.",
                    "holdout_start": None, "curves": {}}
        cut = pd.Timestamp(dates[-cfg.holdout_days])
        holdout_start = cut
        sel, hold = oos[oos["date"] < cut], oos[oos["date"] >= cut]
        for s in grid(h):
            ms = metrics(simulate(sel, s, cfg.cost_pct, cfg.min_turnover_cr), h)
            mh = metrics(simulate(hold, s, cfg.cost_pct, cfg.min_turnover_cr), h)
            rows.append({"settings": s, "Setup": s.label(), "validated": bool(rep["validated"]),
                         **{f"sel_{k}": v for k, v in ms.items()}, **{f"hold_{k}": v for k, v in mh.items()}})
        reports[h]["_oos_for_curves"] = oos

    table = pd.DataFrame(rows)
    base = table[table["settings"] == BASELINE].iloc[0]
    cands = table[table["validated"] & (table["sel_excess_cagr"] > 0)]
    best = cands.sort_values("sel_sharpe").iloc[-1] if not cands.empty else None
    if best is None:
        adopted, chosen = False, BASELINE
        reason = "No setting had a validated model and positive excess return in the selection period."
    elif best["settings"] == BASELINE:
        adopted, chosen = False, BASELINE
        reason = "The current setup was already the best in the selection period."
    elif best["hold_sharpe"] >= base["hold_sharpe"] and best["hold_cagr"] >= base["hold_cagr"]:
        adopted, chosen = True, best["settings"]
        reason = (f"Adopted: best in the selection period (Sharpe {best['sel_sharpe']:.2f} vs baseline "
                  f"{base['sel_sharpe']:.2f}) and confirmed on the unseen final year (Sharpe "
                  f"{best['hold_sharpe']:.2f} vs {base['hold_sharpe']:.2f}; annualised return "
                  f"{100 * best['hold_cagr']:+.1f}% vs {100 * base['hold_cagr']:+.1f}%).")
    else:
        adopted, chosen = False, BASELINE
        failed = [f"Sharpe {best['hold_sharpe']:.2f} vs baseline {base['hold_sharpe']:.2f}"
                  if best["hold_sharpe"] < base["hold_sharpe"] else None,
                  f"return {100 * best['hold_cagr']:+.1f}% vs baseline {100 * base['hold_cagr']:+.1f}%"
                  if best["hold_cagr"] < base["hold_cagr"] else None]
        reason = (f"Not adopted: '{best['Setup']}' looked best in the selection period but was worse than the "
                  f"baseline on the unseen final year ({'; '.join(f for f in failed if f)}). Keeping the baseline.")
    curves = {}
    for name, s in (("Chosen setup", chosen), ("Baseline", BASELINE)):
        oos_h = reports[s.horizon]["_oos_for_curves"]
        curves[name] = simulate(oos_h, s, cfg.cost_pct, cfg.min_turnover_cr)
    for rep in reports.values():
        rep.pop("_oos_for_curves", None)
    show = table.drop(columns=["settings"]).copy()
    pct_cols = [c for c in show.columns if c.endswith(("cagr", "max_dd", "vol", "avg_turnover", "beat_nifty"))]
    show[pct_cols] = show[pct_cols] * 100
    show = show.round(2).sort_values("sel_sharpe", ascending=False).reset_index(drop=True)
    ch = table[table["settings"] == chosen].iloc[0]
    beats_index = bool(ch["hold_cagr"] > ch["hold_nifty_cagr"]) if np.isfinite(ch["hold_nifty_cagr"]) else None
    index_note = ("" if beats_index is None else
                  f"On the unseen final year the chosen setup returned {100 * ch['hold_cagr']:+.1f}% a year vs "
                  f"{100 * ch['hold_nifty_cagr']:+.1f}% for the Nifty 50, "
                  + ("so it beat simply holding an index fund." if beats_index else
                     "so simply holding a Nifty index fund would have done better that year."))
    logger.info("Tuning: %s %s", reason, index_note)
    return {"adopted": adopted, "chosen": chosen, "table": show, "reports": reports, "reason": reason,
            "holdout_start": str(holdout_start.date()) if holdout_start is not None else None, "curves": curves,
            "best": best["Setup"] if best is not None else None, "beats_index": beats_index,
            "index_note": index_note}
