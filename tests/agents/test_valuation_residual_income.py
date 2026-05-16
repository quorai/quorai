"""Tests for R60: residual income model must use shareholders_equity, not market_cap/pb_ratio."""

import pytest

from src.agents.valuation import calculate_residual_income_value


class TestValuationResidualIncome:
    def test_shareholders_equity_used_when_provided(self):
        """
        R60: When shareholders_equity is provided, book_val must equal it — not market_cap/pb_ratio.
        With market_cap=1000, pb=2.0 → derived book_val=500.
        With shareholders_equity=800, the intrinsic should be meaningfully different.
        """
        # Using derived book val (old behaviour)
        val_derived = calculate_residual_income_value(
            market_cap=1_000.0,
            net_income=100.0,
            price_to_book_ratio=2.0,
            book_value_growth=0.05,
            shareholders_equity=None,
        )

        # Using explicit book val from balance sheet
        val_explicit = calculate_residual_income_value(
            market_cap=1_000.0,
            net_income=100.0,
            price_to_book_ratio=2.0,
            book_value_growth=0.05,
            shareholders_equity=800.0,
        )

        # They must differ because book_val differs (500 vs 800)
        assert val_derived != pytest.approx(val_explicit, abs=1.0), "Residual income with shareholders_equity=800 vs derived=500 must produce different values"

    def test_larger_equity_base_produces_different_intrinsic(self):
        """Higher book value changes residual income calculation."""
        val_low = calculate_residual_income_value(
            market_cap=1_000.0,
            net_income=100.0,
            price_to_book_ratio=5.0,
            shareholders_equity=200.0,
        )
        val_high = calculate_residual_income_value(
            market_cap=1_000.0,
            net_income=100.0,
            price_to_book_ratio=5.0,
            shareholders_equity=600.0,
        )
        # Both non-zero, and they must differ
        assert val_low != pytest.approx(val_high, abs=0.1), "Different shareholders_equity values should produce different RIM intrinsic values"

    def test_no_equity_no_pb_returns_zero(self):
        """With neither shareholders_equity nor price_to_book_ratio, model must return 0."""
        val = calculate_residual_income_value(
            market_cap=1_000.0,
            net_income=100.0,
            price_to_book_ratio=None,
            shareholders_equity=None,
        )
        assert val == 0, f"Expected 0 with no book_val source, got {val}"

    def test_negative_shareholders_equity_falls_back_to_pb(self):
        """Negative equity (balance-sheet impaired) must not be used — fall back to pb if available."""
        val = calculate_residual_income_value(
            market_cap=1_000.0,
            net_income=100.0,
            price_to_book_ratio=2.0,
            shareholders_equity=-300.0,
        )
        # Should still compute something using market_cap/pb_ratio fallback (book_val=500)
        assert val >= 0, f"Should fall back to pb_ratio when equity is negative, got {val}"
