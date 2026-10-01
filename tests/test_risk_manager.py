import pytest

from config.config import ExecutionConfig
from risk_engine.guardrails import ExecutionGuard, GuardrailViolation, PaperLedger
from risk_engine.risk_manager import RiskError, RiskManager, Side
from dataclasses import replace
from datetime import date

import pandas as pd


def test_stop_and_target(risk_cfg):
    rm = RiskManager(risk_cfg)
    stop = rm.stop_loss(100.0, 2.0)
    assert stop == pytest.approx(97.0)
    assert rm.take_profit(100.0, stop) == pytest.approx(106.0)
    assert rm.stop_loss(100.0, 2.0, Side.SHORT) == pytest.approx(103.0)
    with pytest.raises(RiskError):
        rm.take_profit(100.0, stop, reward_risk=1.5)


def test_fixed_fractional_sizing(risk_cfg):
    plan = RiskManager(risk_cfg).build_trade_plan("X", 100.0, 2.0)
    # ₹10,000 risk / ₹3 per share = 3,333 units; notional ₹3.33L > 25% cap (₹2.5L) => 2,500
    assert plan.units == 2500 and plan.binding_constraint == "max_position_pct"
    plan2 = RiskManager(risk_cfg).build_trade_plan("Y", 500.0, 40.0)
    # ₹10,000 / ₹60 = 166 units, notional ₹83k within cap
    assert plan2.units == 166 and plan2.risk_amount <= 10_000
    assert plan2.est_cost == pytest.approx(166 * 500 * risk_cfg.est_round_trip_cost_pct, abs=0.01)


def test_nse_tick_rounding(risk_cfg):
    rm = RiskManager(risk_cfg)
    assert rm.tick_size(180) == 0.01 and rm.tick_size(1336) == 0.10 and rm.tick_size(12_450) == 1.0
    plan = rm.build_trade_plan("X", 12_450.4, 211.7)
    for price in (plan.entry_price, plan.stop_loss, plan.take_profit):
        assert price == pytest.approx(round(price))  # whole rupees in the ₹10k-₹20k band


def test_volatility_scaled_never_exceeds_risk_budget(risk_cfg):
    rm = RiskManager(replace(risk_cfg, sizing_method="volatility_scaled", target_volatility_pct=0.05))
    plan = rm.build_trade_plan("X", 500.0, 40.0)
    assert plan.risk_amount <= 10_000 + 1e-6


def test_invalid_inputs(risk_cfg):
    rm = RiskManager(risk_cfg)
    for entry, a in [(0, 1), (100, 0), (100, -1), (1.0, 5.0)]:
        with pytest.raises(RiskError):
            rm.build_trade_plan("X", entry, a)


def test_allocate_enforces_heat_and_count(risk_cfg):
    rm = RiskManager(risk_cfg)  # heat 2.5% = ₹25,000; max 3 positions
    plans = [rm.build_trade_plan(t, 500.0, 40.0) for t in "ABCD"]  # ~₹9,960 risk each
    out = rm.allocate(plans)
    total = sum(p.risk_amount for p in out)
    assert total <= 25_000 and sum(p.units > 0 for p in out) <= 3
    assert out[-1].units == 0


def test_guardrails(tmp_path, risk_cfg):
    plan = RiskManager(risk_cfg).build_trade_plan("X", 100.0, 2.0)
    guard = ExecutionGuard(ExecutionConfig(kill_switch_path=tmp_path / "KILL"), risk_cfg, 1_000_000)
    guard.validate_plan(plan)
    live = ExecutionGuard(replace(ExecutionConfig(), paper_only=False), risk_cfg, 1_000_000)
    with pytest.raises(GuardrailViolation):
        live.assert_paper_only()
    (tmp_path / "KILL").touch()
    with pytest.raises(GuardrailViolation):
        guard.check_kill_switch()
    with pytest.raises(GuardrailViolation):
        guard.validate_plan(replace(plan, units=100_000, risk_amount=300_000, notional=1e7))


def test_paper_ledger_records_and_caps(tmp_path, risk_cfg):
    exec_cfg = ExecutionConfig(kill_switch_path=tmp_path / "KILL", max_orders_per_day=2)
    ledger = PaperLedger(tmp_path, ExecutionGuard(exec_cfg, risk_cfg, 1_000_000))
    plans = [RiskManager(risk_cfg).build_trade_plan(t, 500.0, 40.0) for t in "ABC"]
    records = ledger.record(plans, date(2026, 9, 28))
    assert [r["status"] for r in records] == ["paper_recorded", "paper_recorded", "blocked"]
    saved = pd.read_csv(ledger.path)
    assert len(saved) == 3 and (saved["session_date"] == "2026-09-28").all()
