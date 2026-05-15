"""Tests for R34: technicals strategy functions return neutral on thin price series."""

import numpy as np
import pandas as pd


def _make_prices_df(n_rows: int) -> pd.DataFrame:
    """Create a minimal prices DataFrame with n_rows rows."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    close = 100.0 + np.arange(n_rows, dtype=float) * 0.1
    volume = np.full(n_rows, 1_000_000.0)
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume, "open": close}, index=dates)


_THIN = _make_prices_df(20)  # well below _MIN_PRICES (126)


class TestTrendSignalsThinSeries:
    def test_thin_series_returns_neutral(self):
        from src.agents.technicals import calculate_trend_signals

        result = calculate_trend_signals(_THIN)
        assert result["signal"] == "neutral", f"Expected neutral on thin series, got {result['signal']}"
        assert result["confidence"] == 0

    def test_thin_series_no_exception(self):
        from src.agents.technicals import calculate_trend_signals

        calculate_trend_signals(_THIN)  # must not raise


class TestMeanReversionSignalsThinSeries:
    def test_thin_series_returns_neutral(self):
        from src.agents.technicals import calculate_mean_reversion_signals

        result = calculate_mean_reversion_signals(_THIN)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0

    def test_thin_series_no_exception(self):
        from src.agents.technicals import calculate_mean_reversion_signals

        calculate_mean_reversion_signals(_THIN)


class TestMomentumSignalsThinSeries:
    def test_thin_series_returns_neutral(self):
        from src.agents.technicals import calculate_momentum_signals

        result = calculate_momentum_signals(_THIN)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0

    def test_thin_series_no_exception(self):
        from src.agents.technicals import calculate_momentum_signals

        calculate_momentum_signals(_THIN)


class TestVolatilitySignalsThinSeries:
    def test_thin_series_returns_neutral(self):
        from src.agents.technicals import calculate_volatility_signals

        result = calculate_volatility_signals(_THIN)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0

    def test_thin_series_no_exception(self):
        from src.agents.technicals import calculate_volatility_signals

        calculate_volatility_signals(_THIN)


class TestStatArbSignalsThinSeries:
    def test_thin_series_returns_neutral(self):
        from src.agents.technicals import calculate_stat_arb_signals

        result = calculate_stat_arb_signals(_THIN)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0

    def test_thin_series_no_exception(self):
        from src.agents.technicals import calculate_stat_arb_signals

        calculate_stat_arb_signals(_THIN)
