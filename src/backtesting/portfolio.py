from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, Mapping

from .costs import CostModel
from .types import PortfolioSnapshot, PositionState, TickerRealizedGains


class Portfolio:
    """Portfolio state management for backtesting operations.

    Encapsulates cash, positions, and margin tracking.
    Supports both long and short positions with proper cost basis tracking
    and realized gains/losses calculation.

    When a non-zero ``CostModel`` is supplied:
    * Buy / cover fills use ``price * (1 + slippage_bps/1e4)`` as the execution
      price; sell / short-open fills use ``price * (1 - slippage_bps/1e4)``.
    * A per-trade commission of ``commission_bps/1e4 × notional`` is debited from
      cash separately so cost-basis stays at the execution price and realized-gain
      accounting is interpretable.
    * Short-borrow carry accrues via ``accrue_borrow_cost()`` — call once per day.

    All three cost components default to zero, leaving existing test assertions
    that check exact cash values unaffected.
    """

    def __init__(
        self,
        *,
        tickers: list[str],
        initial_cash: float,
        margin_requirement: float,
        cost_model: CostModel | None = None,
    ) -> None:
        self._cost_model: CostModel = cost_model if cost_model is not None else CostModel.zero()
        self._total_slippage: float = 0.0
        self._total_commission: float = 0.0
        self._total_borrow: float = 0.0
        self._portfolio: PortfolioSnapshot = {
            "cash": float(initial_cash),
            "margin_used": 0.0,
            "margin_requirement": float(margin_requirement),
            "positions": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                }
                for ticker in tickers
            },
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
        }

    def get_snapshot(self) -> PortfolioSnapshot:
        positions_copy: Dict[str, PositionState] = {
            t: {
                "long": p["long"],
                "short": p["short"],
                "long_cost_basis": p["long_cost_basis"],
                "short_cost_basis": p["short_cost_basis"],
                "short_margin_used": p["short_margin_used"],
            }
            for t, p in self._portfolio["positions"].items()
        }
        gains_copy: Dict[str, TickerRealizedGains] = {t: {"long": g["long"], "short": g["short"]} for t, g in self._portfolio["realized_gains"].items()}
        return {
            "cash": float(self._portfolio["cash"]),
            "margin_used": float(self._portfolio["margin_used"]),
            "margin_requirement": float(self._portfolio["margin_requirement"]),
            "positions": positions_copy,
            "realized_gains": gains_copy,
        }

    def get_cash(self) -> float:
        return float(self._portfolio["cash"])

    def available_buying_power(self) -> float:
        """Cash available for new long purchases, net of margin committed to open shorts."""
        return self._portfolio["cash"]

    def get_margin_used(self) -> float:
        return float(self._portfolio["margin_used"])

    def get_margin_requirement(self) -> float:
        return float(self._portfolio["margin_requirement"])

    def get_positions(self) -> Mapping[str, PositionState]:
        return MappingProxyType(self._portfolio["positions"])

    def get_realized_gains(self) -> Mapping[str, TickerRealizedGains]:
        return MappingProxyType(self._portfolio["realized_gains"])

    def get_cost_summary(self) -> dict[str, float]:
        """Return cumulative cost breakdown since portfolio creation."""
        return {
            "total_slippage": self._total_slippage,
            "total_commission": self._total_commission,
            "total_borrow": self._total_borrow,
            "total_costs": self._total_slippage + self._total_commission + self._total_borrow,
        }

    def accrue_borrow_cost(self, prices: dict[str, float]) -> float:
        """Accrue daily short-borrow carry on open short positions.

        Debit ``short_shares * price * borrow_bps_annual / 1e4 / 252`` from cash
        for every ticker that has an open short position.  Returns the total cash
        debited this day.  Call once per trading day (before executing new trades
        so the carry is charged on positions actually held that day).
        """
        if self._cost_model.borrow_bps_annual == 0.0:
            return 0.0
        daily_rate = self._cost_model.borrow_bps_annual / 1e4 / 252.0
        total = 0.0
        for ticker, position in self._portfolio["positions"].items():
            short_qty = position["short"]
            if short_qty > 0:
                price = prices.get(ticker, 0.0)
                if price > 0:
                    daily_cost = short_qty * price * daily_rate
                    self._portfolio["cash"] -= daily_cost
                    self._total_borrow += daily_cost
                    total += daily_cost
        return total

    def apply_long_buy(self, ticker: str, quantity: float, price: float) -> float:
        if quantity <= 0:
            return 0
        position = self._portfolio["positions"][ticker]

        slippage = self._cost_model.slippage_bps / 1e4
        comm_rate = self._cost_model.commission_bps / 1e4
        eff_price = price * (1.0 + slippage)
        # All-in per-share cost: execution price + commission on that notional.
        per_share_all_in = eff_price * (1.0 + comm_rate)

        total_outlay = quantity * per_share_all_in
        if total_outlay <= self.available_buying_power():
            cost = quantity * eff_price
            commission = quantity * eff_price * comm_rate
            old_shares = position["long"]
            old_cost_basis = position["long_cost_basis"]
            total_shares = old_shares + quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_shares
                total_new_cost = cost
                position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["long"] = old_shares + quantity
            if math.isclose(position["long"], 0.0, abs_tol=1e-9):
                position["long"] = 0.0
                position["long_cost_basis"] = 0.0
            self._portfolio["cash"] -= cost + commission
            self._total_slippage += quantity * (eff_price - price)
            self._total_commission += commission
            return quantity
        # Partial fill: max shares affordable at the all-in per-share cost.
        max_quantity = self.available_buying_power() / per_share_all_in if per_share_all_in > 0 else 0
        if max_quantity > 0:
            cost = max_quantity * eff_price
            commission = max_quantity * eff_price * comm_rate
            old_shares = position["long"]
            old_cost_basis = position["long_cost_basis"]
            total_shares = old_shares + max_quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_shares
                total_new_cost = cost
                position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["long"] = old_shares + max_quantity
            if math.isclose(position["long"], 0.0, abs_tol=1e-9):
                position["long"] = 0.0
                position["long_cost_basis"] = 0.0
            self._portfolio["cash"] -= cost + commission
            self._total_slippage += max_quantity * (eff_price - price)
            self._total_commission += commission
            return max_quantity
        return 0

    def apply_long_sell(self, ticker: str, quantity: float, price: float) -> float:
        position = self._portfolio["positions"][ticker]
        quantity = min(quantity, position["long"]) if quantity > 0 else 0
        if quantity <= 0:
            return 0

        slippage = self._cost_model.slippage_bps / 1e4
        comm_rate = self._cost_model.commission_bps / 1e4
        eff_price = price * (1.0 - slippage)
        commission = quantity * eff_price * comm_rate

        avg_cost = position["long_cost_basis"] if position["long"] > 0 else 0.0
        # Realized gain uses slippage-adjusted execution price; commission is a
        # separate drag tracked in _total_commission.
        realized_gain = (eff_price - avg_cost) * quantity - commission
        self._portfolio["realized_gains"][ticker]["long"] += realized_gain
        position["long"] -= quantity
        self._portfolio["cash"] += quantity * eff_price - commission
        if math.isclose(position["long"], 0.0, abs_tol=1e-9):
            position["long"] = 0.0
            position["long_cost_basis"] = 0.0
        self._total_slippage += quantity * (price - eff_price)
        self._total_commission += commission
        return quantity

    def apply_short_open(self, ticker: str, quantity: float, price: float) -> float:
        """Open a short position.

        Debits margin_required from cash (after crediting proceeds) so that
        available_buying_power() is always correct for long-buy affordability
        checks without any additional adjustment.
        """
        if quantity <= 0:
            return 0
        position = self._portfolio["positions"][ticker]

        slippage = self._cost_model.slippage_bps / 1e4
        comm_rate = self._cost_model.commission_bps / 1e4
        eff_price = price * (1.0 - slippage)

        proceeds = eff_price * quantity
        margin_ratio = self._portfolio["margin_requirement"]
        margin_required = proceeds * margin_ratio
        commission = proceeds * comm_rate
        available_cash = max(0.0, self._portfolio["cash"])
        if margin_required <= available_cash:
            old_short_shares = position["short"]
            old_cost_basis = position["short_cost_basis"]
            total_shares = old_short_shares + quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_short_shares
                total_new_cost = eff_price * quantity
                position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["short"] = old_short_shares + quantity
            if math.isclose(position["short"], 0.0, abs_tol=1e-9):
                position["short"] = 0.0
                position["short_cost_basis"] = 0.0
            position["short_margin_used"] += margin_required
            self._portfolio["margin_used"] += margin_required
            self._portfolio["cash"] += proceeds
            self._portfolio["cash"] -= margin_required
            self._portfolio["cash"] -= commission
            self._total_slippage += quantity * (price - eff_price)
            self._total_commission += commission
            return quantity
        # Partial fill: max shares such that margin + commission stays within available_cash.
        denom = eff_price * (margin_ratio + comm_rate) if (margin_ratio + comm_rate) > 0 and eff_price > 0 else 0
        max_quantity = available_cash / denom if denom > 0 else 0
        if max_quantity > 0:
            proceeds = eff_price * max_quantity
            margin_required = proceeds * margin_ratio
            commission = proceeds * comm_rate
            old_short_shares = position["short"]
            old_cost_basis = position["short_cost_basis"]
            total_shares = old_short_shares + max_quantity
            if total_shares > 0:
                total_old_cost = old_cost_basis * old_short_shares
                total_new_cost = eff_price * max_quantity
                position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares
            position["short"] = old_short_shares + max_quantity
            if math.isclose(position["short"], 0.0, abs_tol=1e-9):
                position["short"] = 0.0
                position["short_cost_basis"] = 0.0
            position["short_margin_used"] += margin_required
            self._portfolio["margin_used"] += margin_required
            self._portfolio["cash"] += proceeds
            self._portfolio["cash"] -= margin_required
            self._portfolio["cash"] -= commission
            self._total_slippage += max_quantity * (price - eff_price)
            self._total_commission += commission
            return max_quantity
        return 0

    def apply_short_cover(self, ticker: str, quantity: float, price: float) -> float:
        position = self._portfolio["positions"][ticker]
        quantity = min(quantity, position["short"]) if quantity > 0 else 0
        if quantity <= 0:
            return 0

        slippage = self._cost_model.slippage_bps / 1e4
        comm_rate = self._cost_model.commission_bps / 1e4
        eff_price = price * (1.0 + slippage)

        cover_cost = quantity * eff_price
        commission = cover_cost * comm_rate
        avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0.0
        realized_gain = (avg_short_price - eff_price) * quantity - commission
        if position["short"] > 0:
            portion = min(quantity / position["short"], 1.0)
        else:
            portion = 1.0
        margin_to_release = portion * position["short_margin_used"]
        position["short"] -= quantity
        position["short_margin_used"] -= margin_to_release
        self._portfolio["margin_used"] -= margin_to_release
        self._portfolio["cash"] += margin_to_release
        self._portfolio["cash"] -= cover_cost + commission
        self._portfolio["realized_gains"][ticker]["short"] += realized_gain
        if math.isclose(position["short"], 0.0, abs_tol=1e-9):
            position["short"] = 0.0
            position["short_cost_basis"] = 0.0
            position["short_margin_used"] = 0.0
        self._total_slippage += quantity * (eff_price - price)
        self._total_commission += commission
        return quantity
