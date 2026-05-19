from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.orchestration.price_feed import BacktestPriceFeed, LivePriceFeed


def _make_df(dates, closes, opens=None):
    idx = pd.DatetimeIndex(dates)
    data = {"close": closes}
    if opens is not None:
        data["open"] = opens
    return pd.DataFrame(data, index=idx)


class TestBacktestPriceFeed:
    def test_returns_last_close_on_or_before_date(self):
        trading_dates = ["2025-01-07", "2025-01-08", "2025-01-09"]
        df = _make_df(trading_dates, [100.0, 101.0, 102.0])
        feed = BacktestPriceFeed({"AAPL": df}, spy_prices=None)

        prices = feed.get_signal_prices(["AAPL"], "2025-01-08", "2024-01-08")

        assert prices == {"AAPL": 101.0}

    def test_skips_ticker_with_no_data(self):
        feed = BacktestPriceFeed({}, spy_prices=None)
        prices = feed.get_signal_prices(["AAPL"], "2025-01-08", "2024-01-08")
        assert prices == {}

    def test_skips_ticker_with_no_data_before_date(self):
        df = _make_df(["2025-01-10"], [150.0])
        feed = BacktestPriceFeed({"AAPL": df}, spy_prices=None)
        prices = feed.get_signal_prices(["AAPL"], "2025-01-08", "2024-01-08")
        assert prices == {}

    def test_get_spy_df_returns_prefetched(self):
        spy_df = _make_df(["2025-01-07"], [400.0])
        feed = BacktestPriceFeed({}, spy_prices=spy_df)
        result = feed.get_spy_df("2024-01-07", "2025-01-07")
        pd.testing.assert_frame_equal(result, spy_df)

    def test_get_spy_df_returns_none_when_no_spy(self):
        feed = BacktestPriceFeed({}, spy_prices=None)
        assert feed.get_spy_df("2024-01-07", "2025-01-07") is None


class TestLivePriceFeed:
    def test_get_signal_prices_calls_get_prices(self):
        fake_price = MagicMock()
        fake_price.close = 175.5

        with patch("src.orchestration.price_feed.LivePriceFeed.get_signal_prices") as mock_method:
            mock_method.return_value = {"AAPL": 175.5}
            feed = LivePriceFeed()
            result = feed.get_signal_prices(["AAPL"], "2025-01-08", "2025-01-01")

        assert result == {"AAPL": 175.5}

    def test_get_signal_prices_delegates_to_api(self):
        fake_price = MagicMock()
        fake_price.close = 175.5

        with patch("src.tools.api.get_prices", return_value=[fake_price]) as mock_get:
            feed = LivePriceFeed()
            result = feed.get_signal_prices(["AAPL"], "2025-01-08", "2025-01-01")

        mock_get.assert_called_once_with("AAPL", "2025-01-01", "2025-01-08")
        assert result == {"AAPL": 175.5}

    def test_get_spy_df_delegates_to_api(self):
        spy_df = _make_df(["2025-01-07"], [400.0])

        with patch("src.tools.api.get_price_data", return_value=spy_df) as mock_get:
            feed = LivePriceFeed()
            result = feed.get_spy_df("2024-07-07", "2025-01-07")

        mock_get.assert_called_once_with("SPY", "2024-07-07", "2025-01-07")
        pd.testing.assert_frame_equal(result, spy_df)

    def test_get_spy_df_returns_none_on_empty(self):
        with patch("src.tools.api.get_price_data", return_value=pd.DataFrame()):
            feed = LivePriceFeed()
            result = feed.get_spy_df("2024-07-07", "2025-01-07")

        assert result is None
