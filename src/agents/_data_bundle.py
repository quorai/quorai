"""Shared data-fetch helper for personality agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    search_line_items,
)


@dataclass
class AgentDataBundle:
    ticker: str
    line_items: list[LineItem]
    market_cap: float | None
    financial_metrics: list[FinancialMetrics] = field(default_factory=list)
    insider_trades: list[InsiderTrade] = field(default_factory=list)
    company_news: list[CompanyNews] = field(default_factory=list)
    prices: list[Price] = field(default_factory=list)

    @classmethod
    def fetch(
        cls,
        ticker: str,
        end_date: str,
        *,
        start_date: str | None = None,
        line_item_names: list[str] | None = None,
        line_item_period: str = "annual",
        line_item_limit: int = 5,
        metrics_period: str | None = None,
        metrics_limit: int = 5,
        include_market_cap: bool = True,
        insider_limit: int | None = None,
        news_limit: int | None = None,
        include_prices: bool = False,
        api_key: str | None = None,
    ) -> "AgentDataBundle":
        """Fetch all data needed by a personality agent in one call.

        Pass ``metrics_period`` to fetch financial metrics, ``insider_limit`` /
        ``news_limit`` to fetch insider trades / news, and ``include_prices=True``
        (with ``start_date``) to fetch price history.
        """
        line_items = (
            search_line_items(
                ticker,
                line_item_names,
                end_date,
                period=line_item_period,
                limit=line_item_limit,
                api_key=api_key,
            )
            if line_item_names
            else []
        )

        market_cap = get_market_cap(ticker, end_date, api_key=api_key) if include_market_cap else None

        financial_metrics = get_financial_metrics(ticker, end_date, period=metrics_period, limit=metrics_limit, api_key=api_key) if metrics_period is not None else []

        insider_trades = get_insider_trades(ticker, end_date, start_date=start_date, limit=insider_limit, api_key=api_key) if insider_limit is not None else []

        company_news = get_company_news(ticker, end_date, start_date=start_date, limit=news_limit, api_key=api_key) if news_limit is not None else []

        prices = get_prices(ticker, start_date, end_date, api_key=api_key) if include_prices and start_date else []

        return cls(
            ticker=ticker,
            line_items=line_items,
            market_cap=market_cap,
            financial_metrics=financial_metrics,
            insider_trades=insider_trades,
            company_news=company_news,
            prices=prices,
        )
