"""Tests for the SQLite-backed Cache implementation."""

import threading

import pytest

from src.data.cache import Cache


@pytest.fixture()
def cache():
    """Fresh in-memory cache for each test."""
    return Cache(db_path=":memory:")


class TestInit:
    def test_empty_on_init(self, cache):
        assert cache.get_prices("AAPL") is None
        assert cache.get_financial_metrics("AAPL") is None
        assert cache.get_line_items("AAPL") is None
        assert cache.get_insider_trades("AAPL") is None
        assert cache.get_company_news("AAPL") is None
        assert cache.get_market_cap("AAPL_2024-01-01") is None


class TestRoundTrip:
    def test_prices_roundtrip(self, cache):
        data = [{"time": "2024-01-01T00:00:00", "close": 150.0, "open": 149.0, "high": 151.0, "low": 148.0, "volume": 1000}]
        cache.set_prices("AAPL_2024-01-01_2024-01-31", data)
        result = cache.get_prices("AAPL_2024-01-01_2024-01-31")
        assert result == data

    def test_financial_metrics_roundtrip(self, cache):
        data = [{"report_period": "2024-Q1", "revenue": 90000}]
        cache.set_financial_metrics("AAPL_annual_2024-01-01_4", data)
        assert cache.get_financial_metrics("AAPL_annual_2024-01-01_4") == data

    def test_line_items_roundtrip(self, cache):
        data = [{"report_period": "2024-Q1", "net_income": 20000}]
        cache.set_line_items("AAPL_annual_2024-01-01_4_net_income", data)
        assert cache.get_line_items("AAPL_annual_2024-01-01_4_net_income") == data

    def test_insider_trades_roundtrip(self, cache):
        data = [{"filing_date": "2024-01-15", "transaction_shares": 500.0}]
        cache.set_insider_trades("AAPL_none_2024-01-31_1000", data)
        assert cache.get_insider_trades("AAPL_none_2024-01-31_1000") == data

    def test_company_news_roundtrip(self, cache):
        data = [{"date": "2024-01-01T12:00:00", "title": "Earnings Beat"}]
        cache.set_company_news("AAPL_none_2024-01-31_100", data)
        assert cache.get_company_news("AAPL_none_2024-01-31_100") == data

    def test_market_cap_roundtrip(self, cache):
        cache.set_market_cap("AAPL_2024-01-01", 3_000_000_000.0)
        assert cache.get_market_cap("AAPL_2024-01-01") == pytest.approx(3_000_000_000.0)


class TestUpsertDeduplication:
    def test_prices_dedup_by_time(self, cache):
        key = "AAPL_k1"
        cache.set_prices(key, [{"time": "2024-01-01", "close": 150.0}])
        cache.set_prices(key, [{"time": "2024-01-01", "close": 999.0}, {"time": "2024-01-02", "close": 155.0}])
        result = cache.get_prices(key)
        assert len(result) == 2
        assert result[0]["close"] == 150.0  # original preserved

    def test_financial_metrics_dedup_by_report_period(self, cache):
        key = "AAPL_m1"
        cache.set_financial_metrics(key, [{"report_period": "2024-Q1", "revenue": 1000}])
        cache.set_financial_metrics(key, [{"report_period": "2024-Q1", "revenue": 9999}, {"report_period": "2024-Q2", "revenue": 1100}])
        result = cache.get_financial_metrics(key)
        assert len(result) == 2
        assert result[0]["revenue"] == 1000

    def test_market_cap_overwrites(self, cache):
        cache.set_market_cap("AAPL_2024", 1_000_000.0)
        cache.set_market_cap("AAPL_2024", 2_000_000.0)
        assert cache.get_market_cap("AAPL_2024") == pytest.approx(2_000_000.0)


class TestOnDiskConcurrentWrites:
    """Guard against the connection-leak / FD-exhaustion bug (GH: unable to open database file).

    The fix: _connect() returns the same persistent connection object on every call instead of
    opening a new OS file handle each time.  These tests verify both the structural guarantee
    (connection reuse) and the symptom it prevents (OperationalError under concurrent write load).
    """

    def test_connection_is_reused(self, tmp_path):
        c = Cache(db_path=tmp_path / "reuse.db")
        # _connect() must return the exact same object every time — one FD, not N.
        assert c._connect() is c._connect()

    def test_concurrent_writes_do_not_raise(self, tmp_path):
        """Simulate analyst × ticker nested parallelism against a shared on-disk Cache."""
        c = Cache(db_path=tmp_path / "concurrent.db")
        errors: list[Exception] = []

        def write_batch(analyst_idx: int) -> None:
            for ticker_idx in range(4):
                try:
                    key = f"T{analyst_idx}_{ticker_idx}_2024-01-01"
                    c.set_market_cap(key, float(analyst_idx * 1000 + ticker_idx))
                    c.set_prices(
                        f"prices_{key}",
                        [{"time": "2024-01-01T00:00:00", "close": 100.0 + ticker_idx}],
                    )
                except Exception as exc:
                    errors.append(exc)

        # 8 analyst threads × 4 ticker ops each — mirrors the default parallelism ceiling
        threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes raised: {errors}"

        # Verify data landed correctly for a spot-check key
        assert c.get_market_cap("T0_0_2024-01-01") == pytest.approx(0.0)
        assert c.get_market_cap("T7_3_2024-01-01") == pytest.approx(7003.0)


class TestConcurrentReads:
    def test_concurrent_reads_are_safe(self, cache):
        cache.set_prices("AAPL_concurrent", [{"time": "2024-01-01", "close": 150.0}])
        errors = []

        def reader():
            try:
                result = cache.get_prices("AAPL_concurrent")
                assert result is not None
                assert result[0]["close"] == 150.0
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
