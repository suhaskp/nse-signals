"""Watch-list logging and after-the-fact signal evaluation.

Every run writes ``outputs/watchlist_YYYY-MM-DD.csv`` and appends to
``outputs/watchlist_log.csv``. ``evaluate`` later replays each logged plan
against that session's daily bar to measure hit-rate, R-multiples and model
calibration.

Evaluation assumptions (conservative): fill at the session open (on NSE the
pre-open IEP); if both stop and target fall inside the day's range, the stop
is assumed hit first; otherwise exit at the close, standing in for intraday
(MIS) square-off. Estimated round-trip costs are deducted from every trade.
Daily bars cannot resolve intraday order, so treat results as approximate.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Persists daily watch lists and evaluates past signals."""

    LOG_NAME = "watchlist_log.csv"

    def __init__(self, output_dir: Path, cost_pct: float = 0.0) -> None:
        self.output_dir = Path(output_dir)
        self.cost_pct = cost_pct
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self.output_dir / self.LOG_NAME

    def log_watchlist(self, summary: pd.DataFrame, run_date: date) -> Path:
        """Write today's file and append to the master log (replacing same-day rows)."""
        daily_path = self.output_dir / f"watchlist_{run_date.isoformat()}.csv"
        frame = summary.copy()
        frame.insert(0, "run_date", run_date.isoformat())
        frame.to_csv(daily_path, index=False)
        if self.log_path.exists():
            log = pd.read_csv(self.log_path)
            log = log[log["run_date"] != run_date.isoformat()]
            frame = pd.concat([log, frame], ignore_index=True)
        frame.to_csv(self.log_path, index=False)
        logger.info("Watch list written to %s", daily_path)
        return daily_path

    def log_scan(self, scan: pd.DataFrame, run_date: date) -> Path:
        """Save the full screen (every symbol, with failure reasons) for the day."""
        path = self.output_dir / f"scan_{run_date.isoformat()}.csv"
        scan.to_csv(path, index=False)
        return path

    @staticmethod
    def evaluate_outcome(plan: pd.Series, bar: pd.Series, cost_pct: float = 0.0) -> dict:
        """Replay one long plan against its session's daily bar, net of costs."""
        fill, stop, target = float(bar["Open"]), float(plan["Stop Loss"]), float(plan["Target Price"])
        risk = fill - stop
        if risk <= 0:
            return {"outcome": "gap_through_stop", "exit": fill, "r_multiple": np.nan}
        hit_stop, hit_target = bar["Low"] <= stop, bar["High"] >= target
        if hit_stop:
            outcome, exit_px = ("stop_assumed_first" if hit_target else "stop"), stop
        elif hit_target:
            outcome, exit_px = "target", target
        else:
            outcome, exit_px = "close", float(bar["Close"])
        cost = cost_pct * fill
        return {"outcome": outcome, "exit": exit_px, "r_multiple": (exit_px - fill - cost) / risk,
                "intraday_up": float(bar["Close"] > bar["Open"])}

    def evaluate(self, histories: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
        """Evaluate every logged LONG signal whose session is in ``histories``."""
        if not self.log_path.exists():
            return pd.DataFrame(), {}
        log = pd.read_csv(self.log_path)
        rows = []
        for _, plan in log[log["Signal"] == "LONG"].iterrows():
            df = histories.get(plan["Ticker"])
            ts = pd.Timestamp(plan["run_date"])
            if df is None or ts not in df.index:
                continue
            rows.append({**plan.to_dict(), **self.evaluate_outcome(plan, df.loc[ts], self.cost_pct)})
        results = pd.DataFrame(rows)
        if results.empty:
            return results, {}
        prob = results["Model Probability Score"].astype(float)
        summary = {
            "n_trades": int(len(results)),
            "target_hit_rate": float((results["outcome"] == "target").mean()),
            "avg_r_multiple": float(results["r_multiple"].mean()),
            "total_r": float(results["r_multiple"].sum()),
            "directional_accuracy": float(results["intraday_up"].mean()),
            "brier_score": float(((prob - results["intraday_up"]) ** 2).mean()),
        }
        results.to_csv(self.output_dir / "signal_evaluation.csv", index=False)
        return results, summary
