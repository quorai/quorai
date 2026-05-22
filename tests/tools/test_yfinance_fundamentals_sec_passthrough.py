"""Tests for the SEC store passthrough in fetch_statements / fetch_market_cap."""

from unittest.mock import MagicMock, patch

from src.tools._yfinance_fundamentals import StatementBundle, fetch_statements


def _fake_bundle():
    return StatementBundle(
        period_end="2023-12-31",
        income={"revenue": 50_000.0},
        balance={"total_assets": 300_000.0},
        cashflow={"net_cash_from_operating_activities": 10_000.0},
        shares_outstanding=None,
    )


class TestSecPassthrough:
    def test_sec_hit_returns_without_calling_yfinance(self):
        """When SEC store returns a non-None list, yfinance is never touched."""
        expected = [_fake_bundle()]
        mock_store = MagicMock()
        mock_store.get_statements.return_value = expected

        # get_sec_store is imported lazily inside the function; patch at the source module.
        with patch("src.data.sec_store.get_sec_store", return_value=mock_store):
            with patch("src.tools._yfinance_fundamentals.yf") as mock_yf:
                result = fetch_statements("AAPL", "annual", "2023-12-31", 4)

        assert result is expected
        mock_yf.Ticker.assert_not_called()

    def test_sec_none_falls_through_to_yfinance(self):
        """When SEC store returns None (ticker not seeded), yfinance path is attempted."""
        mock_store = MagicMock()
        mock_store.get_statements.return_value = None

        with patch("src.data.sec_store.get_sec_store", return_value=mock_store):
            with patch("src.tools._yfinance_fundamentals._get_ticker") as mock_get_ticker:
                mock_get_ticker.return_value = None  # _get_ticker returning None → returns []
                result = fetch_statements("UNKNOWN", "annual", "2023-12-31", 4)

        # _get_ticker returned None → empty list from yfinance path
        assert result == []

    def test_sec_empty_list_does_not_fall_through(self):
        """When SEC store returns [] (seeded but no data), yfinance is NOT called."""
        mock_store = MagicMock()
        mock_store.get_statements.return_value = []

        with patch("src.data.sec_store.get_sec_store", return_value=mock_store):
            with patch("src.tools._yfinance_fundamentals.yf") as mock_yf:
                result = fetch_statements("AAPL", "annual", "2020-01-01", 4)

        assert result == []
        mock_yf.Ticker.assert_not_called()
