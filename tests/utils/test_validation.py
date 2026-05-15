"""Tests for ticker validation in src/utils/validation.py."""

import pytest

from src.utils.validation import validate_ticker


class TestValidTickers:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "BRK.B", "RDS-A", "A", "GOOGL"])
    def test_valid_tickers_pass(self, ticker):
        assert validate_ticker(ticker) == ticker

    def test_returns_ticker_unchanged(self):
        assert validate_ticker("NVDA") == "NVDA"


class TestInvalidTickers:
    @pytest.mark.parametrize(
        "ticker",
        [
            "aapl",  # lowercase
            "TOOLONG1234",  # > 10 chars
            "AAPL MSFT",  # space
            "AAPL\nMSFT",  # newline injection
            "AAPL; DROP",  # special chars
            "",  # empty
            "AAPL!",  # exclamation
        ],
    )
    def test_invalid_tickers_raise(self, ticker):
        with pytest.raises(ValueError, match="Invalid ticker"):
            validate_ticker(ticker)
