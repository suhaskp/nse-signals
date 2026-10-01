"""Stop-loss, take-profit and position sizing.

Long trade:  stop = entry - k*ATR,  target = entry + R * (entry - stop)
Short trade: stop = entry + k*ATR,  target = entry - R * (stop - entry)

Sizing methods
--------------
fixed_fractional:   units = floor(equity * risk_pct / risk_per_share)
volatility_scaled:  units = floor(equity * target_vol_pct / ATR), then capped
                    so rupee risk never exceeds the fixed-fractional budget.

Both are then capped by max notional per position (``max_position_pct``).
Prices are rounded to NSE's price-band tick size, and each plan carries an
estimated round-trip cost (brokerage, STT, exchange charges, GST, stamp duty).
All amounts are in INR.
Portfolio-level limits (max positions, total "heat") are applied by
:meth:`RiskManager.allocate`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from enum import Enum

from config.config import RiskConfig

logger = logging.getLogger(__name__)


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class RiskError(ValueError):
    """Raised for inputs that cannot produce a valid trade plan."""


@dataclass(frozen=True)
class TradePlan:
    """Fully specified, sized trade."""

    ticker: str
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    risk_per_share: float
    reward_per_share: float
    reward_risk_ratio: float
    units: int
    notional: float
    risk_amount: float
    est_cost: float
    sizing_method: str
    binding_constraint: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        return d


class RiskManager:
    """Turns (entry, ATR) into a sized trade plan under portfolio limits."""

    def __init__(self, cfg: RiskConfig, account_equity: float | None = None) -> None:
        self.cfg = cfg
        self.equity = float(account_equity if account_equity is not None else cfg.account_equity)
        if self.equity <= 0:
            raise RiskError("account equity must be positive")

    def tick_size(self, price: float) -> float:
        """NSE tick size for ``price`` from the configured price-band table."""
        for upper, tick in self.cfg.tick_table:
            if price < upper:
                return tick
        return self.cfg.tick_table[-1][1]

    def _round(self, price: float, reference: float | None = None) -> float:
        """Round to the tick of the band that ``reference`` (default: price) falls in."""
        tick = self.tick_size(reference if reference is not None else price)
        return round(round(price / tick) * tick, 6)

    @staticmethod
    def _check(entry: float, atr_value: float) -> None:
        if not (math.isfinite(entry) and entry > 0):
            raise RiskError(f"entry must be positive, got {entry}")
        if not (math.isfinite(atr_value) and atr_value > 0):
            raise RiskError(f"ATR must be positive, got {atr_value}")

    def stop_loss(self, entry: float, atr_value: float, side: Side = Side.LONG) -> float:
        """ATR-based stop: entry -/+ ``atr_stop_multiplier`` x ATR."""
        self._check(entry, atr_value)
        dist = self.cfg.atr_stop_multiplier * atr_value
        stop = entry - dist if side is Side.LONG else entry + dist
        if stop <= 0:
            raise RiskError(f"stop {stop:.4f} <= 0: ATR too large relative to price")
        return self._round(stop, entry)

    def take_profit(self, entry: float, stop: float, side: Side = Side.LONG,
                    reward_risk: float | None = None) -> float:
        """Target at ``reward_risk`` x the stop distance."""
        rr = reward_risk if reward_risk is not None else self.cfg.reward_risk_ratio
        if rr < self.cfg.min_reward_risk_ratio:
            raise RiskError(f"reward:risk {rr} below minimum {self.cfg.min_reward_risk_ratio}")
        risk = abs(entry - stop)
        return self._round(entry + rr * risk if side is Side.LONG else entry - rr * risk, entry)

    def size_fixed_fractional(self, risk_per_share: float) -> int:
        """Units such that a stop-out loses ``risk_per_trade_pct`` of equity."""
        if risk_per_share <= 0:
            raise RiskError("risk_per_share must be positive")
        return max(0, math.floor(self.equity * self.cfg.risk_per_trade_pct / risk_per_share))

    def size_volatility_scaled(self, atr_value: float, risk_per_share: float) -> int:
        """Units such that a 1-ATR move equals ``target_volatility_pct`` of equity."""
        vol_units = math.floor(self.equity * self.cfg.target_volatility_pct / atr_value)
        return max(0, min(vol_units, self.size_fixed_fractional(risk_per_share)))

    def build_trade_plan(self, ticker: str, entry: float, atr_value: float,
                         side: Side = Side.LONG) -> TradePlan:
        """Compute stop, target and size for a single trade (before portfolio limits)."""
        entry = self._round(entry)
        stop = self.stop_loss(entry, atr_value, side)
        target = self.take_profit(entry, stop, side)
        risk_ps = abs(entry - stop)
        reward_ps = abs(target - entry)

        if self.cfg.sizing_method == "volatility_scaled":
            units, binding = self.size_volatility_scaled(atr_value, risk_ps), "volatility_target"
        else:
            units, binding = self.size_fixed_fractional(risk_ps), "risk_per_trade"
        notional_cap = math.floor(self.equity * self.cfg.max_position_pct / entry)
        if units > notional_cap:
            units, binding = notional_cap, "max_position_pct"
        if units == 0:
            binding = "insufficient_equity"

        return TradePlan(
            ticker=ticker, side=side, entry_price=entry, stop_loss=stop, take_profit=target,
            atr=round(atr_value, 4), risk_per_share=round(risk_ps, 4), reward_per_share=round(reward_ps, 4),
            reward_risk_ratio=round(reward_ps / risk_ps, 2), units=units,
            notional=round(units * entry, 2), risk_amount=round(units * risk_ps, 2),
            est_cost=round(units * entry * self.cfg.est_round_trip_cost_pct, 2),
            sizing_method=self.cfg.sizing_method, binding_constraint=binding,
        )

    def plan_from_levels(self, ticker: str, entry: float, stop: float, target: float, atr_value: float,
                         side: Side = Side.LONG) -> TradePlan:
        """Size a trade whose stop and target were set by a strategy (e.g. breakout structure)."""
        self._check(entry, atr_value)
        entry = self._round(entry)
        stop, target = self._round(stop, entry), self._round(target, entry)
        risk_ps = entry - stop if side is Side.LONG else stop - entry
        reward_ps = target - entry if side is Side.LONG else entry - target
        if risk_ps <= 0 or reward_ps <= 0:
            raise RiskError(f"{ticker}: stop/target on the wrong side of the entry")
        units, binding = self.size_fixed_fractional(risk_ps), "risk_per_trade"
        cap = math.floor(self.equity * self.cfg.max_position_pct / entry)
        if units > cap:
            units, binding = cap, "max_position_pct"
        if units == 0:
            binding = "insufficient_equity"
        return TradePlan(ticker=ticker, side=side, entry_price=entry, stop_loss=stop, take_profit=target,
                         atr=round(atr_value, 4), risk_per_share=round(risk_ps, 4), reward_per_share=round(reward_ps, 4),
                         reward_risk_ratio=round(reward_ps / risk_ps, 2), units=units, notional=round(units * entry, 2),
                         risk_amount=round(units * risk_ps, 2),
                         est_cost=round(units * entry * self.cfg.est_round_trip_cost_pct, 2),
                         sizing_method="fixed_fractional", binding_constraint=binding)

    def allocate(self, plans: list[TradePlan], used_heat: float = 0.0, taken: int = 0) -> list[TradePlan]:
        """Apply portfolio limits in priority order (callers sort best-first).

        Enforces ``max_open_positions`` and ``max_portfolio_heat_pct``; a plan
        that would breach the heat budget is downsized to fit, or zeroed.
        Returns every plan, with ``units=0`` for those not allocated.
        """
        heat_budget = self.equity * self.cfg.max_portfolio_heat_pct
        used, out = used_heat, []  # positions already open count against both limits
        for plan in plans:
            if plan.units == 0:
                out.append(plan)
                continue
            if taken >= self.cfg.max_open_positions:
                out.append(self._resize(plan, 0, "max_open_positions"))
                continue
            remaining = heat_budget - used
            units = plan.units
            if plan.risk_amount > remaining:
                units = max(0, math.floor(remaining / plan.risk_per_share))
            if units == 0:
                out.append(self._resize(plan, 0, "portfolio_heat"))
                continue
            sized = plan if units == plan.units else self._resize(plan, units, "portfolio_heat")
            used += sized.risk_amount
            taken += 1
            out.append(sized)
        logger.info("Allocated %d positions, total risk ₹%.2f (%.2f%% of equity)",
                    taken, used, 100 * used / self.equity)
        return out

    def _resize(self, plan: TradePlan, units: int, reason: str) -> TradePlan:
        return replace(plan, units=units, notional=round(units * plan.entry_price, 2),
                       risk_amount=round(units * plan.risk_per_share, 2),
                       est_cost=round(units * plan.entry_price * self.cfg.est_round_trip_cost_pct, 2),
                       binding_constraint=reason)
