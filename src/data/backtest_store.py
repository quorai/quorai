"""In-memory pre-fetch store for backtesting.

Populated once by BacktestEngine._prefetch_data and consulted by api.py functions
before the SQLite cache + HTTP path. Live-mode code never calls install(), so the
store stays inactive and all api.py calls fall through to their normal paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price

# Conservative SEC filing lags: earliest date a report is likely public.
# 10-Q: due within 40-45 days after quarter end (using 45 for non-accelerated filers).
# 10-K / annual: due within 60-90 days (using 90 to be safe).
_QUARTERLY_LAG_DAYS = 45
_ANNUAL_LAG_DAYS = 90


def _earliest_available(report_period: str, period: str) -> str:
    """Return the earliest YYYY-MM-DD on which this report is likely public."""
    lag = _ANNUAL_LAG_DAYS if period == "annual" else _QUARTERLY_LAG_DAYS
    return (datetime.strptime(report_period, "%Y-%m-%d") + timedelta(days=lag)).strftime("%Y-%m-%d")


@dataclass
class _TickerStore:
    prices: list[Price] = field(default_factory=list)  # ascending by time
    # period ("ttm" | "annual" | "quarterly") -> newest-first list
    financial_metrics: dict[str, list[FinancialMetrics]] = field(default_factory=dict)
    line_items: dict[str, list[LineItem]] = field(default_factory=dict)
    insider_trades: list[InsiderTrade] = field(default_factory=list)
    company_news: list[CompanyNews] = field(default_factory=list)
    yfinance_news: list[CompanyNews] = field(default_factory=list)
    sec_news: list[CompanyNews] = field(default_factory=list)
    shares_outstanding: float | None = None


class BacktestStore:
    """Process-global singleton. Inactive until install() is called."""

    def __init__(self) -> None:
        self._data: dict[str, _TickerStore] = {}
        self._window: tuple[str, str] | None = None  # (prefetch_start, prefetch_end)
        self._active: bool = False

    def install(self, data: dict[str, _TickerStore], window: tuple[str, str]) -> None:
        self._data = data
        self._window = window
        self._active = True

    def uninstall(self) -> None:
        self._active = False
        self._data = {}
        self._window = None

    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def slice_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price] | None:
        if not self._active or ticker not in self._data:
            return None
        prices = self._data[ticker].prices
        return [p for p in prices if start_date <= p.time[:10] <= end_date]

    # ------------------------------------------------------------------
    # Financial metrics
    # ------------------------------------------------------------------

    def slice_financial_metrics(
        self,
        ticker: str,
        period: str,
        end_date: str,
        limit: int,
    ) -> list[FinancialMetrics] | None:
        if not self._active or ticker not in self._data:
            return None
        by_period = self._data[ticker].financial_metrics
        if period not in by_period:
            return None  # period not pre-fetched; fall through to HTTP
        filtered = [m for m in by_period[period] if _earliest_available(m.report_period, m.period) <= end_date]
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Line items
    # ------------------------------------------------------------------

    def slice_line_items(
        self,
        ticker: str,
        period: str,
        end_date: str,
        limit: int,
    ) -> list[LineItem] | None:
        """Return pre-fetched line items filtered to end_date.

        The store holds a superset of all fields; agents only access the attributes
        they requested, so extra populated fields are harmless (LineItem allows
        extra fields via model_config = {"extra": "allow"}).
        """
        if not self._active or ticker not in self._data:
            return None
        by_period = self._data[ticker].line_items
        if period not in by_period:
            return None
        filtered = [li for li in by_period[period] if _earliest_available(li.report_period, li.period) <= end_date]
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Insider trades
    # ------------------------------------------------------------------

    def slice_insider_trades(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int,
    ) -> list[InsiderTrade] | None:
        if not self._active or ticker not in self._data:
            return None
        result = []
        for t in self._data[ticker].insider_trades:
            fd = t.filing_date or t.transaction_date or ""
            if fd > end_date:
                continue
            if start_date and fd and fd < start_date:
                continue
            result.append(t)
        return result[:limit]

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def slice_company_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int,
    ) -> list[CompanyNews] | None:
        if not self._active or ticker not in self._data:
            return None
        return _filter_news(self._data[ticker].company_news, start_date, end_date, limit)

    def slice_yfinance_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int,
    ) -> list[CompanyNews] | None:
        if not self._active or ticker not in self._data:
            return None
        return _filter_news(self._data[ticker].yfinance_news, start_date, end_date, limit)

    def slice_sec_news(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str,
        limit: int,
    ) -> list[CompanyNews] | None:
        if not self._active or ticker not in self._data:
            return None
        return _filter_news(self._data[ticker].sec_news, start_date, end_date, limit)

    # ------------------------------------------------------------------
    # Market cap (analytical: shares_outstanding × close price)
    # ------------------------------------------------------------------

    def market_cap(self, ticker: str, end_date: str) -> float | None:
        if not self._active or ticker not in self._data:
            return None
        store = self._data[ticker]
        shares = store.shares_outstanding
        if not shares:
            return None
        # prices is sorted ascending; find the last price on or before end_date
        candidates = [p for p in store.prices if p.time[:10] <= end_date]
        if not candidates:
            return None
        return shares * candidates[-1].close


def _filter_news(
    news: list[CompanyNews],
    start_date: str | None,
    end_date: str,
    limit: int,
) -> list[CompanyNews]:
    result = []
    for n in news:
        d = n.date[:10] if n.date else ""
        if not d:
            result.append(n)
            continue
        if d > end_date:
            continue
        if start_date and d < start_date:
            continue
        result.append(n)
    return result[:limit]


_store = BacktestStore()


def get_backtest_store() -> BacktestStore:
    return _store
