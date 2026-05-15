"""Tests for R29: backtest portfolio short margin is not double-subtracted."""

import pytest

from src.backtesting.portfolio import Portfolio


def _portfolio(cash: float = 10_000.0, margin: float = 0.5) -> Portfolio:
    tickers = ["AAPL", "MSFT"]
    return Portfolio(initial_cash=cash, margin_requirement=margin, tickers=tickers)


class TestShortMarginNotDoubleSubtracted:
    def test_second_short_sized_off_cash_not_cash_minus_margin_used(self):
        """
        Open two sequential shorts. The second short's available capacity must be
        cash-after-first-short, not cash-after-first-short minus margin_used.

        Pre-fix: available_cash = cash - margin_used (double-subtracts because cash
        was already reduced by margin_required when the first short was opened).
        Post-fix: available_cash = cash (reflects real buying power).
        """
        p = _portfolio(cash=10_000.0, margin=0.5)

        # Short 10 AAPL @ $100 → proceeds=$1000, margin=$500.
        # After: cash = 10_000 + 1000 - 500 = 10_500; margin_used = 500.
        p.apply_short_open("AAPL", quantity=10, price=100.0)

        cash_after_first_short = p.get_cash()
        margin_after_first_short = p.get_margin_used()
        assert cash_after_first_short == pytest.approx(10_500.0)
        assert margin_after_first_short == pytest.approx(500.0)

        # With the double-subtract bug, available_cash = 10_500 - 500 = 10_000.
        # After the fix, available_cash = 10_500.
        # At margin_ratio=0.5 and price=$100, max short qty = available_cash / (price * ratio).
        # Bug: 10_000 / (100 * 0.5) = 200 shares (same as initial — as if first short never happened)
        # Fix: 10_500 / (100 * 0.5) = 210 shares
        # We test that a 201-share short request fills 201 shares (only possible if available_cash > 10_000).
        filled = p.apply_short_open("MSFT", quantity=201, price=100.0)
        assert filled == pytest.approx(201.0), "Second short should fill 201 shares; with the double-count bug available_cash was understated to exactly 10_000, and qty=201 would be capped at 200."

    def test_available_cash_equals_actual_cash_after_short(self):
        """After opening a short, cash already reflects the margin haircut."""
        p = _portfolio(cash=5_000.0, margin=0.5)
        p.apply_short_open("AAPL", quantity=5, price=100.0)

        # proceeds=500, margin=250 → cash = 5_000 + 500 - 250 = 5_250
        assert p.get_cash() == pytest.approx(5_250.0)
        assert p.get_margin_used() == pytest.approx(250.0)
