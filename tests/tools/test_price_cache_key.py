"""Tests for ticker-level price cache key (prevents range-based fragmentation)."""

from unittest.mock import patch

from src.data.cache import Cache


def _seed_cache(cache: Cache, ticker: str, rows: list[dict]) -> None:
    cache.set_prices(f"prices_{ticker}", rows)


def _make_price_row(date: str, close: float = 100.0) -> dict:
    return {"open": close, "close": close, "high": close, "low": close, "volume": 1000, "time": f"{date}T00:00:00"}


def test_cache_key_is_ticker_only():
    """Price cache key must be prices_{ticker}, not a range-based key."""
    cache = Cache(db_path=":memory:")
    rows = [_make_price_row("2024-01-15")]
    cache.set_prices("prices_AAPL", rows)

    assert cache.get_prices("prices_AAPL") is not None
    assert cache.get_prices("AAPL_2024-01-01_2024-06-30") is None


def test_sub_range_returns_from_cache():
    """A narrower request is sliced from cache without re-downloading."""
    cache = Cache(db_path=":memory:")
    # Seed daily prices Jan 1 – Jun 30 (end_date of the sub-range request is within this)
    jan_jun = [_make_price_row(f"2024-0{m}-{d:02d}") for m in range(1, 7) for d in [1, 15]]
    _seed_cache(cache, "AAPL", jan_jun)

    # Request Jan 1 – Feb 15: max cached date in range is 2024-02-15 == end_date → cache hit
    with (
        patch("src.tools.api.get_backtest_store") as mock_store,
        patch("src.tools.api._cache", cache),
    ):
        mock_store.return_value.slice_prices.return_value = None

        import yfinance as yf

        with patch.object(yf, "download", side_effect=AssertionError("yfinance must not be called")):
            from src.tools.api import get_prices

            result = get_prices("AAPL", "2024-01-01", "2024-02-15")

    assert len(result) > 0
    for p in result:
        assert "2024-01-01" <= p.time[:10] <= "2024-02-15"


def test_wider_range_re_downloads_missing_tail():
    """A request with end_date > 5 days past max cached data falls through to yfinance."""
    cache = Cache(db_path=":memory:")
    # Cache has data only through 2024-01-15; request goes to 2024-03-31 (75 days later)
    jan = [_make_price_row("2024-01-15")]
    _seed_cache(cache, "AAPL", jan)

    fetched_from_yf = [False]

    def fake_download(*args, **kwargs):
        import pandas as pd

        fetched_from_yf[0] = True
        return pd.DataFrame()  # Empty — just to confirm yfinance was called

    with (
        patch("src.tools.api.get_backtest_store") as mock_store,
        patch("src.tools.api._cache", cache),
        patch("yfinance.download", side_effect=fake_download),
    ):
        mock_store.return_value.slice_prices.return_value = None

        from src.tools.api import get_prices

        get_prices("AAPL", "2024-01-01", "2024-03-31")

    assert fetched_from_yf[0], "yfinance must be called when cached data is >5 days stale vs end_date"


def test_get_prices_uses_auto_adjust():
    """yfinance download must always be called with auto_adjust=True."""
    import pandas as pd

    captured_kwargs: dict = {}

    def fake_download(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return pd.DataFrame()

    with (
        patch("src.tools.api.get_backtest_store") as mock_store,
        patch("src.tools.api._cache", Cache(db_path=":memory:")),
        patch("yfinance.download", side_effect=fake_download),
    ):
        mock_store.return_value.slice_prices.return_value = None
        from src.tools.api import get_prices

        get_prices("AAPL", "2024-01-01", "2024-01-31")

    assert captured_kwargs.get("auto_adjust") is True, "auto_adjust must be True to get split-adjusted prices"
