"""Tests for the filing-lag gate in BacktestStore.

Financial statements are public only after the SEC filing deadline:
- Quarterly (10-Q): 45 days after period end
- Annual (10-K): 90 days after period end

A backtest on 2024-04-01 must NOT see Q1-2024 (period end 2024-03-31)
because the filing isn't due until mid-May.
"""

from src.data.backtest_store import BacktestStore, _earliest_available, _TickerStore
from src.data.models import FinancialMetrics, LineItem


def _make_metric(report_period: str, period: str) -> FinancialMetrics:
    return FinancialMetrics(
        ticker="AAPL",
        report_period=report_period,
        period=period,
        currency="USD",
        **{k: None for k in FinancialMetrics.model_fields if k not in ("ticker", "report_period", "period", "currency")},
    )


def _make_line_item(report_period: str, period: str) -> LineItem:
    return LineItem(ticker="AAPL", report_period=report_period, period=period, currency="USD")


def _store_with_metrics(metrics: list[FinancialMetrics], period: str) -> BacktestStore:
    store = BacktestStore()
    ticker_store = _TickerStore()
    ticker_store.financial_metrics[period] = metrics
    store.install({"AAPL": ticker_store}, ("2020-01-01", "2025-01-01"))
    return store


def _store_with_line_items(items: list[LineItem], period: str) -> BacktestStore:
    store = BacktestStore()
    ticker_store = _TickerStore()
    ticker_store.line_items[period] = items
    store.install({"AAPL": ticker_store}, ("2020-01-01", "2025-01-01"))
    return store


class TestEarliestAvailable:
    def test_quarterly_lag_is_45_days(self):
        result = _earliest_available("2024-03-31", "quarterly")
        assert result == "2024-05-15"

    def test_annual_lag_is_90_days(self):
        result = _earliest_available("2023-12-31", "annual")
        assert result == "2024-03-30"

    def test_ttm_uses_quarterly_lag(self):
        result = _earliest_available("2024-03-31", "ttm")
        assert result == "2024-05-15"


class TestSliceFinancialMetrics:
    def test_recent_quarter_excluded_before_filing_lag(self):
        # Q1-2024 ended 2024-03-31; filing deadline ~2024-05-15
        q1 = _make_metric("2024-03-31", "quarterly")
        store = _store_with_metrics([q1], "quarterly")
        result = store.slice_financial_metrics("AAPL", "quarterly", end_date="2024-04-01", limit=10)
        assert result == []

    def test_recent_quarter_included_after_filing_lag(self):
        q1 = _make_metric("2024-03-31", "quarterly")
        store = _store_with_metrics([q1], "quarterly")
        result = store.slice_financial_metrics("AAPL", "quarterly", end_date="2024-05-15", limit=10)
        assert len(result) == 1

    def test_previous_quarter_always_visible(self):
        q4 = _make_metric("2023-12-31", "quarterly")
        store = _store_with_metrics([q4], "quarterly")
        # Q4-2023 filing deadline ~2024-02-14; checking well after
        result = store.slice_financial_metrics("AAPL", "quarterly", end_date="2024-04-01", limit=10)
        assert len(result) == 1

    def test_annual_uses_90_day_lag(self):
        # FY2023 ended 2023-12-31; filing deadline ~2024-03-30
        fy23 = _make_metric("2023-12-31", "annual")
        store = _store_with_metrics([fy23], "annual")
        assert store.slice_financial_metrics("AAPL", "annual", end_date="2024-03-29", limit=10) == []
        assert len(store.slice_financial_metrics("AAPL", "annual", end_date="2024-03-30", limit=10)) == 1


class TestSliceLineItems:
    def test_recent_quarter_excluded_before_lag(self):
        q1 = _make_line_item("2024-03-31", "quarterly")
        store = _store_with_line_items([q1], "quarterly")
        result = store.slice_line_items("AAPL", "quarterly", end_date="2024-04-01", limit=10)
        assert result == []

    def test_recent_quarter_included_after_lag(self):
        q1 = _make_line_item("2024-03-31", "quarterly")
        store = _store_with_line_items([q1], "quarterly")
        result = store.slice_line_items("AAPL", "quarterly", end_date="2024-05-15", limit=10)
        assert len(result) == 1
