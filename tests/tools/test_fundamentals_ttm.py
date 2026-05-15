"""Tests for TTM partial-quarter handling in _yfinance_fundamentals._fetch_ttm."""

import logging
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _make_income_df(n_quarters: int, revenue_per_quarter: float = 1000.0) -> pd.DataFrame:
    """Create a quarterly income statement DataFrame in yfinance format.

    yfinance DataFrames: rows = field labels, columns = quarter Timestamps.
    """
    dates = [pd.Timestamp(f"2024-{3 * (4 - i):02d}-01") for i in range(n_quarters)]
    # Build with dates as columns and "Total Revenue" as the only row label
    df = pd.DataFrame({d: {"Total Revenue": revenue_per_quarter} for d in dates})
    return df


def _make_yf_ticker(n_income_quarters: int = 4) -> MagicMock:
    ticker = MagicMock()
    ticker.quarterly_income_stmt = _make_income_df(n_income_quarters)
    ticker.quarterly_balance_sheet = pd.DataFrame()
    ticker.quarterly_cashflow = pd.DataFrame()
    return ticker


class TestTTMFourQuarters:
    def test_four_quarters_revenue_is_sum(self):
        """With all 4 quarters, revenue = 4 × per_quarter value."""
        from src.tools._yfinance_fundamentals import _fetch_ttm

        yf_ticker = _make_yf_ticker(n_income_quarters=4)
        bundles = _fetch_ttm(yf_ticker, "AAPL", "2024-12-31", limit=1)

        assert len(bundles) == 1
        assert bundles[0].income.get("revenue") == pytest.approx(4000.0)


class TestTTMPartialQuarters:
    def test_three_quarters_revenue_is_none(self, caplog):
        """With only 3 quarters available, revenue is None (not a 3-quarter partial sum)."""
        from src.tools._yfinance_fundamentals import _fetch_ttm

        yf_ticker = _make_yf_ticker(n_income_quarters=3)

        with caplog.at_level(logging.WARNING, logger="src.tools._yfinance_fundamentals"):
            bundles = _fetch_ttm(yf_ticker, "AAPL", "2024-12-31", limit=1)

        assert len(bundles) == 1
        assert bundles[0].income.get("revenue") is None

    def test_three_quarters_emits_warning(self, caplog):
        """When fewer than 4 quarters are available, a WARNING with the count is logged."""
        from src.tools._yfinance_fundamentals import _fetch_ttm

        yf_ticker = _make_yf_ticker(n_income_quarters=3)

        with caplog.at_level(logging.WARNING, logger="src.tools._yfinance_fundamentals"):
            _fetch_ttm(yf_ticker, "AAPL", "2024-12-31", limit=1)

        assert any("3/4" in r.message for r in caplog.records), "Expected a WARNING mentioning '3/4 quarters'"

    def test_one_quarter_revenue_is_none(self, caplog):
        """Even a single quarter returns None for revenue (not a 1-quarter sum)."""
        from src.tools._yfinance_fundamentals import _fetch_ttm

        yf_ticker = _make_yf_ticker(n_income_quarters=1)

        with caplog.at_level(logging.WARNING, logger="src.tools._yfinance_fundamentals"):
            bundles = _fetch_ttm(yf_ticker, "AAPL", "2024-12-31", limit=1)

        if bundles:
            assert bundles[0].income.get("revenue") is None
