"""Tests for R36: calculate_owner_earnings_value normalizes capex sign."""

import pytest

from src.agents.valuation import calculate_owner_earnings_value


class TestOwnerEarningsCapexSign:
    def test_negative_capex_equals_positive_capex(self):
        """
        capex from yfinance is conventionally negative (cash-flow outflow sign).
        calculate_owner_earnings_value must produce the same result for capex=-1000
        as for capex=+1000 (textbook positive convention).
        """
        kwargs = dict(
            net_income=5_000_000.0,
            depreciation=1_000_000.0,
            working_capital_change=200_000.0,
            growth_rate=0.05,
        )
        val_positive_capex = calculate_owner_earnings_value(capex=1_000_000.0, **kwargs)
        val_negative_capex = calculate_owner_earnings_value(capex=-1_000_000.0, **kwargs)

        assert val_positive_capex == pytest.approx(val_negative_capex), f"capex=-1000 produced {val_negative_capex:.2f} but capex=+1000 produced {val_positive_capex:.2f}. The function must normalize capex sign (both conventions should yield the same owner earnings)."

    def test_negative_capex_yields_correct_owner_earnings(self):
        """
        With capex=-1000 (yfinance convention):
        owner_earnings = NI + D - abs(capex) - WCΔ
                       = 5000 + 1000 - 1000 - 200 = 4800.
        Before the fix: owner_earnings = 5000 + 1000 - (-1000) - 200 = 6800 (2× too high).
        """
        val = calculate_owner_earnings_value(
            net_income=5_000.0,
            depreciation=1_000.0,
            capex=-1_000.0,
            working_capital_change=200.0,
        )
        # val is a DCF sum, not owner_earnings directly, but must be positive and reasonable.
        # Key: val with capex=-1000 must be the same as capex=+1000.
        val_positive = calculate_owner_earnings_value(
            net_income=5_000.0,
            depreciation=1_000.0,
            capex=1_000.0,
            working_capital_change=200.0,
        )
        assert val == pytest.approx(val_positive)

    def test_zero_owner_earnings_returns_zero(self):
        """If owner earnings is <= 0 after capex, return 0 (not inflated by wrong sign)."""
        val = calculate_owner_earnings_value(
            net_income=1_000.0,
            depreciation=100.0,
            capex=-5_000.0,  # large capex outflow → negative owner earnings
            working_capital_change=0.0,
        )
        assert val == 0, "When abs(capex) > NI + D, owner earnings should be <= 0, returning 0."

    def test_none_components_return_zero(self):
        val = calculate_owner_earnings_value(
            net_income=None,
            depreciation=1_000.0,
            capex=-500.0,
            working_capital_change=100.0,
        )
        assert val == 0
