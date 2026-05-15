"""Tests for yfinance API field mapping: prices, financial metrics, line items, insider trades, company news, market cap."""

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from src.tools.api import (
    _build_financial_metrics,
    get_company_news,
    get_insider_trades,
    get_market_cap,
    get_prices,
    search_line_items,
)


def _mock_response(status_code: int, json_data) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    return r


def _make_yf_ticker(income_data: dict | None = None, balance_data: dict | None = None, cashflow_data: dict | None = None, info: dict | None = None) -> MagicMock:
    """Build a MagicMock yfinance Ticker with DataFrames for the given data."""
    col = pd.Timestamp("2023-12-31")

    def _df(data: dict | None) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        return pd.DataFrame({col: data})

    t = MagicMock()
    t.income_stmt = _df(income_data)
    t.quarterly_income_stmt = _df(income_data)
    t.balance_sheet = _df(balance_data)
    t.quarterly_balance_sheet = _df(balance_data)
    t.cashflow = _df(cashflow_data)
    t.quarterly_cashflow = _df(cashflow_data)
    t.info = info or {}
    # fast_info: None for both attributes so tests fall through to .info
    t.fast_info = MagicMock()
    t.fast_info.market_cap = None
    t.fast_info.shares = None
    t.get_shares_full.return_value = None
    t.history.return_value = pd.DataFrame()
    return t


class TestGetPrices:
    def test_get_prices_returns_price_list(self):
        """Cache hit returns correctly typed Price objects."""
        cached = [
            {"open": 100.0, "close": 105.0, "high": 106.0, "low": 99.0, "volume": 1_000_000, "time": "2024-01-02T00:00:00"},
            {"open": 105.0, "close": 110.0, "high": 111.0, "low": 104.0, "volume": 1_100_000, "time": "2024-01-03T00:00:00"},
        ]
        with patch("src.tools.api._cache.get_prices", return_value=cached):
            result = get_prices("AAPL", "2024-01-01", "2024-01-31")

        assert len(result) == 2
        assert all(isinstance(p, Price) for p in result)
        assert result[0].open == pytest.approx(100.0)
        assert result[0].close == pytest.approx(105.0)
        assert result[0].volume == 1_000_000
        assert result[1].close == pytest.approx(110.0)


class TestBuildFinancialMetrics:
    def _income(self, **kw) -> dict:
        base = {
            "period_end": "2023-12-31",
            "revenue": 100_000.0,
            "gross_profit": 60_000.0,
            "operating_income": 30_000.0,
            "net_income_loss_attributable_common_shareholders": 20_000.0,
            "basic_earnings_per_share": 2.50,
            "basic_shares_outstanding": 8_000.0,
            "ebitda": None,
            "depreciation_depletion_amortization": 5_000.0,
            "research_development": 10_000.0,
            "selling_general_administrative": 20_000.0,
            "interest_expense": 1_000.0,
            "cost_of_revenue": 40_000.0,
        }
        return {**base, **kw}

    def _balance(self, **kw) -> dict:
        base = {
            "period_end": "2023-12-31",
            "total_assets": 200_000.0,
            "total_liabilities": 80_000.0,
            "total_current_assets": 50_000.0,
            "total_current_liabilities": 20_000.0,
            "total_equity_attributable_to_parent": 120_000.0,
            "total_equity": 120_000.0,
            "long_term_debt_and_capital_lease_obligations": 30_000.0,
            "debt_current": 5_000.0,
        }
        return {**base, **kw}

    def test_build_financial_metrics_fields(self):
        m = _build_financial_metrics("AAPL", self._income(), self._balance(), "annual")

        assert isinstance(m, FinancialMetrics)
        assert m.ticker == "AAPL"
        assert m.report_period == "2023-12-31"
        assert m.earnings_per_share == pytest.approx(2.50)
        assert m.gross_margin == pytest.approx(0.60)
        assert m.operating_margin == pytest.approx(0.30)
        assert m.net_margin == pytest.approx(0.20)
        assert m.return_on_equity == pytest.approx(20_000 / 120_000)
        assert m.return_on_assets == pytest.approx(20_000 / 200_000)
        assert m.debt_to_equity == pytest.approx(35_000 / 120_000)
        assert m.current_ratio == pytest.approx(50_000 / 20_000)
        assert m.book_value_per_share == pytest.approx(120_000 / 8_000)
        # ROIC = net_income / (total_debt + total_equity)
        assert m.return_on_invested_capital == pytest.approx(20_000 / (35_000 + 120_000))

    def test_zero_revenue_margins_are_none(self):
        """Zero revenue → all margin ratios are None, not a ZeroDivisionError."""
        m = _build_financial_metrics("AAPL", self._income(revenue=0), self._balance(), "annual")

        assert m.gross_margin is None
        assert m.operating_margin is None
        assert m.net_margin is None


class TestSearchLineItems:
    def test_search_line_items_returns_typed_list(self):
        """yfinance income_stmt maps to LineItem objects with correct field values."""
        yf_ticker = _make_yf_ticker(
            income_data={
                "Total Revenue": 100_000.0,
                "Net Income Common Stockholders": 20_000.0,
            }
        )

        with (
            patch("src.tools.api._cache.get_line_items", return_value=None),
            patch("src.tools.api._cache.set_line_items"),
            patch("src.tools._yfinance_fundamentals.yf") as mock_yf,
        ):
            mock_yf.Ticker.return_value = yf_ticker
            result = search_line_items("AAPL", ["revenue", "net_income"], "2024-05-01", period="annual", limit=1)

        assert len(result) == 1
        item = result[0]
        assert isinstance(item, LineItem)
        assert item.ticker == "AAPL"
        assert item.revenue == pytest.approx(100_000)
        assert item.net_income == pytest.approx(20_000)

    def test_search_line_items_goodwill_and_intangibles(self):
        """goodwill_and_intangible_assets is computed as goodwill + intangible_assets_net."""
        yf_ticker = _make_yf_ticker(
            income_data={"Total Revenue": 1.0},  # needed so period is detected
            balance_data={
                "Goodwill": 50_000.0,
                "Other Intangible Assets": 10_000.0,
            },
        )

        with (
            patch("src.tools.api._cache.get_line_items", return_value=None),
            patch("src.tools.api._cache.set_line_items"),
            patch("src.tools._yfinance_fundamentals.yf") as mock_yf,
        ):
            mock_yf.Ticker.return_value = yf_ticker
            result = search_line_items("AAPL", ["goodwill_and_intangible_assets"], "2024-05-01", period="annual", limit=1)

        assert len(result) == 1
        assert result[0].goodwill_and_intangible_assets == pytest.approx(60_000)

    def test_search_line_items_return_on_invested_capital(self):
        """return_on_invested_capital is served via the metric route."""
        yf_ticker = _make_yf_ticker(
            income_data={"Net Income Common Stockholders": 20_000.0},
            balance_data={
                "Total Debt": 30_000.0,
                "Stockholders Equity": 120_000.0,
            },
        )

        with (
            patch("src.tools.api._cache.get_line_items", return_value=None),
            patch("src.tools.api._cache.set_line_items"),
            patch("src.tools._yfinance_fundamentals.yf") as mock_yf,
        ):
            mock_yf.Ticker.return_value = yf_ticker
            result = search_line_items("AAPL", ["return_on_invested_capital"], "2024-05-01", period="annual", limit=1)

        assert len(result) == 1
        # ROIC = 20_000 / (30_000 + 120_000)
        assert result[0].return_on_invested_capital == pytest.approx(20_000 / 150_000)


class TestGetInsiderTrades:
    def test_get_insider_trades_maps_fields(self):
        """Finnhub insider-transactions response maps to InsiderTrade with correct fields."""
        payload = {
            "data": [
                {
                    "name": "Jane Smith",
                    "transactionDate": "2024-01-10",
                    "filingDate": "2024-01-12",
                    "change": 500,
                    "share": 5_000,
                    "value": 50_000,
                }
            ]
        }
        mock_resp = _mock_response(200, payload)

        with (
            patch("src.tools.api._cache.get_insider_trades", return_value=None),
            patch("src.tools.api._cache.set_insider_trades"),
            patch("src.tools.api._make_api_request", return_value=mock_resp),
        ):
            result = get_insider_trades("AAPL", "2024-02-01", api_key="fake")

        assert len(result) == 1
        t = result[0]
        assert isinstance(t, InsiderTrade)
        assert t.ticker == "AAPL"
        assert t.name == "Jane Smith"
        assert t.transaction_date == "2024-01-10"
        assert t.filing_date == "2024-01-12"
        assert t.transaction_shares == pytest.approx(500.0)
        assert t.shares_owned_after_transaction == 5_000
        assert t.transaction_value == pytest.approx(50_000.0)
        assert t.transaction_price_per_share == pytest.approx(100.0)


class TestGetCompanyNews:
    def test_get_company_news_maps_fields(self):
        """Finnhub company-news response maps to CompanyNews with correct fields."""
        ts = int(datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp())
        payload = [
            {
                "headline": "Apple Q1 Earnings Beat",
                "source": "Reuters",
                "url": "https://reuters.com/article/1",
                "datetime": ts,
            }
        ]
        mock_resp = _mock_response(200, payload)

        with (
            patch("src.tools.api._cache.get_company_news", return_value=None),
            patch("src.tools.api._cache.set_company_news"),
            patch("src.tools.api._make_api_request", return_value=mock_resp),
        ):
            result = get_company_news("AAPL", "2024-02-01", api_key="fake")

        assert len(result) == 1
        n = result[0]
        assert isinstance(n, CompanyNews)
        assert n.ticker == "AAPL"
        assert n.title == "Apple Q1 Earnings Beat"
        assert n.source == "Reuters"
        assert n.url == "https://reuters.com/article/1"
        assert "2024-01-15" in n.date


class TestGetMarketCap:
    def test_get_market_cap_from_info(self):
        """For a near-today date, market cap is read from yfinance info['marketCap']."""
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        yf_ticker = _make_yf_ticker(info={"marketCap": 3_000_000_000_000})

        with (
            patch("src.tools.api._cache.get_market_cap", return_value=None),
            patch("src.tools.api._cache.set_market_cap"),
            patch("src.tools._yfinance_fundamentals.yf") as mock_yf,
        ):
            mock_yf.Ticker.return_value = yf_ticker
            result = get_market_cap("AAPL", end_date)

        assert result == pytest.approx(3_000_000_000_000)

    def test_get_market_cap_returns_none_when_yfinance_unavailable(self):
        """When yfinance returns no info, market cap is None and cache miss is recorded."""
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        yf_ticker = _make_yf_ticker(info={})

        with (
            patch("src.tools.api._cache.get_market_cap", return_value=None),
            patch("src.tools.api._market_cap_none_cache", set()),
            patch("src.tools._yfinance_fundamentals.yf") as mock_yf,
        ):
            mock_yf.Ticker.return_value = yf_ticker
            result = get_market_cap("AAPL", end_date)

        assert result is None
