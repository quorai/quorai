from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class PriceFeed(Protocol):
    def get_signal_prices(self, tickers: list[str], date: str, lookback_start: str) -> dict[str, float]: ...
    def get_spy_df(self, lookback_start: str, date: str) -> pd.DataFrame | None: ...


class BacktestPriceFeed:
    """Resolves prices from pre-fetched DataFrames for backtesting."""

    def __init__(
        self,
        prefetched_prices: dict[str, pd.DataFrame],
        spy_prices: pd.DataFrame | None,
    ) -> None:
        self._prices = prefetched_prices
        self._spy_prices = spy_prices

    def get_signal_prices(self, tickers: list[str], date: str, lookback_start: str) -> dict[str, float]:
        signal_prices: dict[str, float] = {}
        for ticker in tickers:
            df = self._prices.get(ticker)
            if df is None or df.empty:
                logger.warning("No price data for %s on %s, skipping ticker", ticker, date)
                continue
            sliced = df[df.index <= pd.Timestamp(date)]
            if sliced.empty:
                logger.warning("No price data for %s on %s, skipping ticker", ticker, date)
                continue
            signal_prices[ticker] = float(sliced.iloc[-1]["close"])
        return signal_prices

    def get_spy_df(self, lookback_start: str, date: str) -> pd.DataFrame | None:
        return self._spy_prices


class LivePriceFeed:
    """Resolves prices from live market data APIs."""

    def get_signal_prices(self, tickers: list[str], date: str, lookback_start: str) -> dict[str, float]:
        from src.tools.api import get_prices

        signal_prices: dict[str, float] = {}
        for ticker in tickers:
            prices = get_prices(ticker, lookback_start, date)
            if prices:
                signal_prices[ticker] = prices[-1].close
        return signal_prices

    def get_spy_df(self, lookback_start: str, date: str) -> pd.DataFrame | None:
        from src.tools.api import get_price_data

        fetched = get_price_data("SPY", lookback_start, date)
        return fetched if not fetched.empty else None
