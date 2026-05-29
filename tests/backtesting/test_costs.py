"""Tests for the CostModel and its integration with Portfolio.

Covers:
* Slippage direction per action type.
* Commission cash debit.
* Daily borrow accrual.
* Affordability / partial-fill under costs.
* Regression: CostModel.zero() gives identical results to pre-cost behaviour.
"""

from __future__ import annotations

import math

import pytest

from src.backtesting.costs import CostModel
from src.backtesting.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _portfolio(initial_cash: float = 100_000.0, cost_model: CostModel | None = None) -> Portfolio:
    return Portfolio(
        tickers=["AAPL", "TSLA"],
        initial_cash=initial_cash,
        margin_requirement=0.5,
        cost_model=cost_model,
    )


# ---------------------------------------------------------------------------
# CostModel construction
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_zero_factory_is_frictionless(self):
        cm = CostModel.zero()
        assert cm.slippage_bps == 0.0
        assert cm.commission_bps == 0.0
        assert cm.borrow_bps_annual == 0.0

    def test_from_args_stores_values(self):
        cm = CostModel.from_args(slippage_bps=5.0, commission_bps=2.0, borrow_bps_annual=50.0)
        assert cm.slippage_bps == 5.0
        assert cm.commission_bps == 2.0
        assert cm.borrow_bps_annual == 50.0

    def test_frozen(self):
        cm = CostModel.zero()
        with pytest.raises(Exception):
            cm.slippage_bps = 10.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Regression: zero-cost model exactly matches prior behaviour
# ---------------------------------------------------------------------------


class TestZeroCostRegression:
    """CostModel.zero() must reproduce exact pre-cost cash arithmetic."""

    def test_buy_full_fill_cash(self):
        p = _portfolio(cost_model=CostModel.zero())
        p.apply_long_buy("AAPL", 100, 150.0)
        assert math.isclose(p.get_cash(), 100_000 - 100 * 150.0)

    def test_sell_cash(self):
        p = _portfolio(cost_model=CostModel.zero())
        p.apply_long_buy("AAPL", 100, 100.0)
        p.apply_long_sell("AAPL", 100, 120.0)
        assert math.isclose(p.get_cash(), 100_000 - 100 * 100.0 + 100 * 120.0)

    def test_short_open_cash(self):
        p = _portfolio(cost_model=CostModel.zero())
        p.apply_short_open("AAPL", 10, 100.0)
        proceeds = 10 * 100.0
        margin = proceeds * 0.5
        assert math.isclose(p.get_cash(), 100_000 + proceeds - margin)

    def test_short_cover_cash(self):
        p = _portfolio(cost_model=CostModel.zero())
        p.apply_short_open("AAPL", 10, 100.0)
        cash_after_open = p.get_cash()
        p.apply_short_cover("AAPL", 10, 90.0)
        # margin released = 10 * 100 * 0.5 = 500; cover cost = 900
        margin = 10 * 100.0 * 0.5
        cover = 10 * 90.0
        assert math.isclose(p.get_cash(), cash_after_open + margin - cover)


# ---------------------------------------------------------------------------
# Slippage direction tests
# ---------------------------------------------------------------------------


class TestSlippageDirection:
    """Slippage should WORSEN fill prices (higher for buys/covers, lower for sells/shorts)."""

    SLIPPAGE_BPS = 100  # 1% — large enough to see clearly

    def test_buy_fills_at_higher_price(self):
        no_cost = _portfolio()
        with_cost = _portfolio(cost_model=CostModel.from_args(slippage_bps=self.SLIPPAGE_BPS))

        no_cost.apply_long_buy("AAPL", 10, 100.0)
        with_cost.apply_long_buy("AAPL", 10, 100.0)

        # With slippage, more cash is consumed.
        assert with_cost.get_cash() < no_cost.get_cash()

    def test_sell_receives_less_cash(self):
        p_no = _portfolio()
        p_yes = _portfolio(cost_model=CostModel.from_args(slippage_bps=self.SLIPPAGE_BPS))
        for p in (p_no, p_yes):
            p.apply_long_buy("AAPL", 10, 100.0)
        cash_before_no = p_no.get_cash()
        cash_before_yes = p_yes.get_cash()

        p_no.apply_long_sell("AAPL", 10, 110.0)
        p_yes.apply_long_sell("AAPL", 10, 110.0)

        proceeds_no = p_no.get_cash() - cash_before_no
        proceeds_yes = p_yes.get_cash() - cash_before_yes
        assert proceeds_yes < proceeds_no

    def test_short_open_receives_less_cash(self):
        p_no = _portfolio()
        p_yes = _portfolio(cost_model=CostModel.from_args(slippage_bps=self.SLIPPAGE_BPS))

        cash_before_no = p_no.get_cash()
        cash_before_yes = p_yes.get_cash()
        p_no.apply_short_open("AAPL", 10, 100.0)
        p_yes.apply_short_open("AAPL", 10, 100.0)

        # Net cash received from short open is lower with slippage (worse short price).
        delta_no = p_no.get_cash() - cash_before_no
        delta_yes = p_yes.get_cash() - cash_before_yes
        assert delta_yes < delta_no

    def test_cover_costs_more_cash(self):
        p_no = _portfolio()
        p_yes = _portfolio(cost_model=CostModel.from_args(slippage_bps=self.SLIPPAGE_BPS))
        for p in (p_no, p_yes):
            p.apply_short_open("AAPL", 10, 100.0)

        cash_before_no = p_no.get_cash()
        cash_before_yes = p_yes.get_cash()
        p_no.apply_short_cover("AAPL", 10, 95.0)
        p_yes.apply_short_cover("AAPL", 10, 95.0)

        delta_no = p_no.get_cash() - cash_before_no
        delta_yes = p_yes.get_cash() - cash_before_yes
        # With slippage cover costs more, so net cash increase is smaller.
        assert delta_yes < delta_no

    def test_slippage_tracked_in_cost_summary(self):
        p = _portfolio(cost_model=CostModel.from_args(slippage_bps=self.SLIPPAGE_BPS))
        p.apply_long_buy("AAPL", 10, 100.0)
        p.apply_long_sell("AAPL", 10, 110.0)
        summary = p.get_cost_summary()
        assert summary["total_slippage"] > 0.0
        assert summary["total_commission"] == 0.0


# ---------------------------------------------------------------------------
# Commission tests
# ---------------------------------------------------------------------------


class TestCommission:
    COMM_BPS = 20  # 0.2%

    def test_commission_debited_on_buy(self):
        p_no = _portfolio()
        p_yes = _portfolio(cost_model=CostModel.from_args(commission_bps=self.COMM_BPS))
        p_no.apply_long_buy("AAPL", 10, 100.0)
        p_yes.apply_long_buy("AAPL", 10, 100.0)
        assert p_yes.get_cash() < p_no.get_cash()

    def test_commission_debited_on_sell(self):
        p_no = _portfolio()
        p_yes = _portfolio(cost_model=CostModel.from_args(commission_bps=self.COMM_BPS))
        for p in (p_no, p_yes):
            p.apply_long_buy("AAPL", 10, 100.0)
        p_no.apply_long_sell("AAPL", 10, 100.0)
        p_yes.apply_long_sell("AAPL", 10, 100.0)
        assert p_yes.get_cash() < p_no.get_cash()

    def test_commission_amount_correct(self):
        """Commission = commission_bps / 1e4 × qty × eff_price (eff_price = price when no slippage)."""
        cm = CostModel.from_args(commission_bps=self.COMM_BPS)
        p = _portfolio(cost_model=cm)
        p.apply_long_buy("AAPL", 10, 100.0)
        expected_commission = 10 * 100.0 * (self.COMM_BPS / 1e4)
        expected_cash = 100_000 - 10 * 100.0 - expected_commission
        assert math.isclose(p.get_cash(), expected_cash, rel_tol=1e-9)

    def test_commission_tracked_in_cost_summary(self):
        p = _portfolio(cost_model=CostModel.from_args(commission_bps=self.COMM_BPS))
        p.apply_long_buy("AAPL", 10, 100.0)
        p.apply_long_sell("AAPL", 10, 100.0)
        summary = p.get_cost_summary()
        assert summary["total_commission"] > 0.0
        assert summary["total_slippage"] == 0.0


# ---------------------------------------------------------------------------
# Borrow accrual tests
# ---------------------------------------------------------------------------


class TestBorrowAccrual:
    def test_no_accrual_without_short_positions(self):
        p = _portfolio(cost_model=CostModel.from_args(borrow_bps_annual=500))
        cash_before = p.get_cash()
        accrued = p.accrue_borrow_cost({"AAPL": 100.0, "TSLA": 200.0})
        assert accrued == 0.0
        assert p.get_cash() == cash_before

    def test_accrues_on_open_short(self):
        cm = CostModel.from_args(borrow_bps_annual=252 * 100)  # 100bps/day for easy maths
        p = _portfolio(cost_model=cm)
        p.apply_short_open("AAPL", 10, 100.0)
        cash_before = p.get_cash()
        accrued = p.accrue_borrow_cost({"AAPL": 100.0})
        # daily_rate = 252*100/1e4/252 = 0.01 (1% per day)
        expected = 10 * 100.0 * 0.01
        assert math.isclose(accrued, expected, rel_tol=1e-9)
        assert math.isclose(p.get_cash(), cash_before - expected, rel_tol=1e-9)

    def test_borrow_tracked_in_cost_summary(self):
        cm = CostModel.from_args(borrow_bps_annual=252 * 100)
        p = _portfolio(cost_model=cm)
        p.apply_short_open("AAPL", 5, 200.0)
        p.accrue_borrow_cost({"AAPL": 200.0})
        summary = p.get_cost_summary()
        assert summary["total_borrow"] > 0.0
        assert summary["total_costs"] == summary["total_borrow"]

    def test_zero_borrow_rate_is_no_op(self):
        p = _portfolio(cost_model=CostModel.zero())
        p.apply_short_open("AAPL", 10, 100.0)
        cash_before = p.get_cash()
        accrued = p.accrue_borrow_cost({"AAPL": 100.0})
        assert accrued == 0.0
        assert p.get_cash() == cash_before


# ---------------------------------------------------------------------------
# Affordability / partial-fill under costs
# ---------------------------------------------------------------------------


class TestAffordabilityWithCosts:
    def test_near_capacity_buy_does_not_overdraw_cash(self):
        """A buy that is just within buying power + costs must leave cash >= 0."""
        initial = 1010.0
        cm = CostModel.from_args(slippage_bps=50, commission_bps=20)
        p = Portfolio(
            tickers=["AAPL"],
            initial_cash=initial,
            margin_requirement=0.0,
            cost_model=cm,
        )
        # Try to buy 10 shares at $100 (notional = $1000, all-in ~ $1011)
        p.apply_long_buy("AAPL", 10, 100.0)
        assert p.get_cash() >= 0.0

    def test_partial_fill_does_not_overdraw_cash(self):
        """Partial-fill path under high costs must never produce negative cash."""
        initial = 500.0
        cm = CostModel.from_args(slippage_bps=200, commission_bps=100)  # extreme costs
        p = Portfolio(
            tickers=["AAPL"],
            initial_cash=initial,
            margin_requirement=0.0,
            cost_model=cm,
        )
        qty = p.apply_long_buy("AAPL", 100, 100.0)  # only partial fill possible
        assert p.get_cash() >= -1e-6  # allow tiny float epsilon
        assert qty < 100  # partial fill occurred

    def test_total_cost_summary_is_non_negative(self):
        cm = CostModel.from_args(slippage_bps=10, commission_bps=5)
        p = _portfolio(cost_model=cm)
        p.apply_long_buy("AAPL", 5, 100.0)
        p.apply_long_sell("AAPL", 5, 110.0)
        s = p.get_cost_summary()
        assert s["total_slippage"] >= 0
        assert s["total_commission"] >= 0
        assert math.isclose(s["total_costs"], s["total_slippage"] + s["total_commission"] + s["total_borrow"])
