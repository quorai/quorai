from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkCalculator:
    """Computes benchmark buy-and-hold returns from a pre-loaded price DataFrame.

    Call `load(df)` once with the full date-range prices before iterating.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame = pd.DataFrame()

    def load(self, df: pd.DataFrame) -> None:
        """Store the pre-fetched price data for later lookups."""
        self._df = df

    def get_return_pct(self, ticker: str, start_date: str, end_date: str) -> float | None:
        """Compute simple buy-and-hold return % for ticker from start_date to end_date.

        Uses the pre-loaded DataFrame; falls back to None if data is unavailable.
        Return is (last_close / first_close - 1) * 100.
        """
        try:
            df = self._df
            if df.empty:
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
