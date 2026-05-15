"""Tests that classify_regime handles NaN prices and short history gracefully."""

import numpy as np
import pandas as pd

from src.regime import MarketRegime, classify_regime


def _make_spy(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


def _last_date(df: pd.DataFrame) -> str:
    return str(df.index[-1].date())


def test_nan_in_close_returns_neutral():
    closes = [float("nan")] * 30 + list(np.linspace(100, 120, 30))
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.NEUTRAL


def test_single_nan_in_otherwise_valid_series_returns_neutral():
    closes = list(np.linspace(100, 120, 70))
    closes[35] = float("nan")  # inject a gap mid-series
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.NEUTRAL


def test_too_short_history_returns_neutral():
    # VOL_LONG_WINDOW default is 60; 30 bars is not enough
    closes = list(np.linspace(100, 120, 30))
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result == MarketRegime.NEUTRAL


def test_valid_series_is_not_neutral():
    # Steadily rising prices — should produce a non-neutral classification
    closes = list(np.linspace(100, 130, 80))
    spy = _make_spy(closes)
    result = classify_regime(spy, _last_date(spy))
    assert result != MarketRegime.NEUTRAL
