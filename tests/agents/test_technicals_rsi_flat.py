"""Tests for R62: RSI must return 50 (neutral) on flat series and 100 on rising-only series."""

import numpy as np
import pandas as pd

from src.agents.technicals import calculate_rsi


def _flat_df(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"close": np.full(n, price)})


def _rising_df(n: int = 50, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"close": np.arange(start, start + n * step, step)})


def _falling_df(n: int = 50, start: float = 150.0, step: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"close": np.arange(start, start - n * step, -step)})


class TestTechnicalsRsiFlat:
    def test_flat_series_returns_50(self):
        """
        R62: Flat price series → avg_gain=0, avg_loss=0 → NaN before fix → 0 via safe_float.
        After fix: returns 50.0 (neutral) instead of NaN/0.
        """
        rsi = calculate_rsi(_flat_df(), period=14)
        # After the rolling window fills (row >= period), all values must be 50
        filled = rsi.dropna()
        assert len(filled) > 0
        assert (filled == 50.0).all(), f"Flat series must produce RSI=50 (neutral). Got:\n{filled.tail()}"

    def test_rising_series_returns_100(self):
        """R62: Strictly rising series → avg_loss=0, avg_gain>0 → RSI must be 100."""
        rsi = calculate_rsi(_rising_df(), period=14)
        filled = rsi.dropna()
        assert len(filled) > 0
        assert (filled == 100.0).all(), f"Strictly rising series must produce RSI=100. Got:\n{filled.tail()}"

    def test_falling_series_returns_0(self):
        """Strictly falling series → avg_gain=0, avg_loss>0 → RSI=0 (original formula)."""
        rsi = calculate_rsi(_falling_df(), period=14)
        filled = rsi.dropna()
        assert len(filled) > 0
        assert (filled == 0.0).all(), f"Strictly falling series must produce RSI=0. Got:\n{filled.tail()}"

    def test_normal_series_not_affected(self):
        """Regression: typical noisy series should produce RSI in (0, 100)."""
        rng = np.random.default_rng(42)
        closes = 100.0 + rng.standard_normal(100).cumsum()
        df = pd.DataFrame({"close": closes})
        rsi = calculate_rsi(df, period=14)
        filled = rsi.dropna()
        assert ((filled > 0) & (filled < 100)).all(), "Normal price series must produce RSI strictly between 0 and 100"
