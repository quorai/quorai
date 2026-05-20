from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkCalculator:
    """Computes benchmark buy-and-hold returns from pre-loaded price DataFrames.

    Call `load(df, ticker)` for each ticker before querying. Defaults to "SPY"
    so existing call-sites that omit the ticker argument continue to work.
    """

    def __init__(self) -> None:
        self._dfs: dict[str, pd.DataFrame] = {}

    def load(self, df: pd.DataFrame, ticker: str = "SPY") -> None:
        """Store a pre-fetched price DataFrame under the given ticker key."""
        self._dfs[ticker] = df

    def get_daily_returns(self, ticker: str, start_date: str, end_date: str) -> "pd.Series | None":
        """Return a Series of daily pct-change returns for ticker in [start_date, end_date].

        Indexed by date (same dtype as the DataFrame index). Returns None if data is missing.
        """
        try:
            df = self._dfs.get(ticker)
            if df is None or df.empty:
                return None
            mask = (df.index >= start_date) & (df.index <= end_date)
            window = df.loc[mask, "close"].dropna()
            if len(window) < 2:
                return None
            return window.pct_change().dropna()
        except Exception:
            logger.warning("Failed to compute daily returns for %s (%s→%s)", ticker, start_date, end_date)
            return None

    def get_basket_daily_returns(self, tickers: list[str], start_date: str, end_date: str) -> "pd.Series | None":
        """Return daily returns for a buy-and-hold equal-weight basket of tickers.

        Each ticker is normalized by its first close so the basket is buy-and-hold weighted.
        Returns None if any ticker's data is missing or insufficient.
        """
        import numpy as np

        normalized: list[pd.Series] = []
        for ticker in tickers:
            df = self._dfs.get(ticker)
            if df is None or df.empty:
                return None
            mask = (df.index >= start_date) & (df.index <= end_date)
            window = df.loc[mask, "close"].dropna()
            if len(window) < 2:
                return None
            first = float(window.iloc[0])
            if first == 0 or np.isnan(first):
                return None
            normalized.append(window / first)
        basket = pd.concat(normalized, axis=1, join="inner").mean(axis=1)
        daily = basket.pct_change().dropna()
        if len(daily) < 1:
            return None
        return daily

    def get_return_pct(self, ticker: str, start_date: str, end_date: str) -> float | None:
        """Compute simple buy-and-hold return % for ticker from start_date to end_date.

        Uses the pre-loaded DataFrame for that ticker; returns None if data is unavailable.
        Return is (last_close / first_close - 1) * 100.
        """
        try:
            df = self._dfs.get(ticker)
            if df is None or df.empty:
                return None
            # Filter to the requested date window
            mask = (df.index >= start_date) & (df.index <= end_date)
            window = df.loc[mask]
            if window.empty:
                return None
            first_close = window.iloc[0]["close"]
            last_close = window.iloc[-1]["close"]
            if first_close is None or pd.isna(first_close):
                return None
            if last_close is None or pd.isna(last_close):
                last_valid = window["close"].dropna()
                if last_valid.empty:
                    return None
                last_close = float(last_valid.iloc[-1])
            return (float(last_close) / float(first_close) - 1.0) * 100.0
        except Exception:
            logger.warning("Failed to compute benchmark return for %s (%s→%s)", ticker, start_date, end_date)
            return None
