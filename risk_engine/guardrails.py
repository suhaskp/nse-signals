"""Execution guardrails. This build never places broker orders.

"Paper orders" are written to a local ledger CSV and evaluated after the
close by :mod:`monitoring.performance_log`. Layers of protection:

1. Config validation rejects ``execution.paper_only=False`` outright.
2. ``ExecutionGuard.assert_paper_only`` re-checks this at runtime.
3. A kill-switch file (default ``./KILL_SWITCH``) halts every run if present.
4. Every paper order is re-validated (geometry, risk budget, notional cap, R:R).
5. A per-run order cap.

Automating real order placement in India also brings in SEBI's framework for
retail algorithmic trading through brokers (broker approval, order tagging and
related controls). Check the current rules with your broker before adding it.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import ExecutionConfig, RiskConfig
from risk_engine.risk_manager import Side, TradePlan

logger = logging.getLogger(__name__)
_TOL = 1e-6


class GuardrailViolation(RuntimeError):
    """An action was blocked by a safety guardrail."""


class ExecutionGuard:
    """Validates that paper orders respect every hard limit."""

    def __init__(self, exec_cfg: ExecutionConfig, risk_cfg: RiskConfig, account_equity: float) -> None:
        self.exec_cfg, self.risk_cfg, self.equity = exec_cfg, risk_cfg, account_equity
        self._orders_this_run = 0

    def check_kill_switch(self) -> None:
        if self.exec_cfg.kill_switch_path.exists():
            raise GuardrailViolation(f"Kill switch present at {self.exec_cfg.kill_switch_path}; halting")

    def assert_paper_only(self) -> None:
        if not self.exec_cfg.paper_only:
            raise GuardrailViolation("Broker order placement is not supported by this pipeline")

    def validate_plan(self, plan: TradePlan) -> None:
        """Raise if the plan breaks any hard limit."""
        problems = []
        if plan.units <= 0:
            problems.append("units must be > 0")
        if plan.side is Side.LONG and not (plan.stop_loss < plan.entry_price < plan.take_profit):
            problems.append("long requires stop < entry < target")
        if plan.side is Side.SHORT and not (plan.take_profit < plan.entry_price < plan.stop_loss):
            problems.append("short requires target < entry < stop")
        if plan.risk_amount > self.equity * self.risk_cfg.risk_per_trade_pct * (1 + _TOL) + 0.01:
            problems.append(f"risk ₹{plan.risk_amount:,.2f} exceeds per-trade budget")
        if plan.notional > self.equity * self.risk_cfg.max_position_pct * (1 + _TOL) + 0.01:
            problems.append(f"notional ₹{plan.notional:,.2f} exceeds max position size")
        if plan.reward_risk_ratio + 0.01 < self.risk_cfg.min_reward_risk_ratio:
            problems.append(f"R:R {plan.reward_risk_ratio} below minimum")
        if problems:
            raise GuardrailViolation(f"{plan.ticker}: " + "; ".join(problems))

    def register_order(self) -> None:
        self._orders_this_run += 1
        if self._orders_this_run > self.exec_cfg.max_orders_per_day:
            raise GuardrailViolation(f"Order cap {self.exec_cfg.max_orders_per_day} reached")


class PaperLedger:
    """Records validated paper orders to ``paper_ledger.csv`` (no network, no broker)."""

    FILENAME = "paper_ledger.csv"

    def __init__(self, output_dir: Path, guard: ExecutionGuard) -> None:
        self.path = Path(output_dir) / self.FILENAME
        self.guard = guard

    def record(self, plans: list[TradePlan], session_date: date) -> list[dict[str, Any]]:
        """Validate and record each plan. Returns one status record per plan."""
        self.guard.check_kill_switch()
        self.guard.assert_paper_only()
        records: list[dict[str, Any]] = []
        for plan in plans:
            rec: dict[str, Any] = {"session_date": session_date.isoformat(), **plan.to_dict()}
            try:
                self.guard.validate_plan(plan)
                self.guard.register_order()
                rec["status"] = "paper_recorded"
                logger.info("[PAPER] %s %d %s @ ₹%.2f | stop ₹%.2f | target ₹%.2f", plan.side.value.upper(),
                            plan.units, plan.ticker, plan.entry_price, plan.stop_loss, plan.take_profit)
            except GuardrailViolation as exc:
                logger.error("Blocked: %s", exc)
                rec.update(status="blocked", reason=str(exc))
                records.append(rec)
                if "Order cap" in str(exc):
                    break
                continue
            records.append(rec)
        if records:
            new = pd.DataFrame(records)
            if self.path.exists():
                old = pd.read_csv(self.path)
                old = old[old["session_date"] != session_date.isoformat()]
                new = pd.concat([old, new], ignore_index=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new.to_csv(self.path, index=False)
        return records
