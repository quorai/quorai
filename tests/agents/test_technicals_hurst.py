"""Tests for R61: Hurst exponent must use std() not sqrt(std())."""

import numpy as np

from src.agents.technicals import calculate_hurst_exponent


def _random_walk_prices(n: int = 1000, seed: int = 0) -> np.ndarray:
    """Standard random walk (cumsum of i.i.d. returns); true H = 0.5."""
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.standard_normal(n))


def _noisy_trend_prices(n: int = 1000, trend: float = 0.5, seed: int = 0) -> np.ndarray:
    """
    Linear trend dominated series: P[t] ≈ trend * t + noise.
    std(diff at lag) scales closer to lag^1 than lag^0.5, so H > 0.5.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    noise = rng.standard_normal(n) * 0.1
    return trend * t + noise.cumsum()


class TestTechnicalsHurst:
    def test_random_walk_hurst_near_half(self):
        """
        R61: std(P[t+lag] - P[t]) = σ * sqrt(lag) for a random walk → slope = 0.5.
        Before fix: sqrt(std) halves the log-slope → H ≈ 0.25.
        After fix: slope is correctly 0.5.
        """
        prices = _random_walk_prices(n=1000, seed=42)
        h = calculate_hurst_exponent(prices, max_lag=30)
        assert 0.4 < h < 0.7, f"Random walk must have H near 0.5. Got H={h:.3f}. A value ≈ 0.25 indicates the sqrt bug is still present."

    def test_trending_series_hurst_above_random_walk(self):
        """
        R61: Trend-dominated series has std(diffs at lag) scaling closer to lag^1
        (linear) than lag^0.5 (random walk), so H(trend) > H(random walk).
        """
        h_trend = calculate_hurst_exponent(_noisy_trend_prices(n=1000), max_lag=30)
        h_rw = calculate_hurst_exponent(_random_walk_prices(n=1000), max_lag=30)
        assert h_trend > h_rw, f"Trend-dominated prices (H={h_trend:.3f}) must have higher Hurst than random walk (H={h_rw:.3f})"

    def test_hurst_returns_finite_float(self):
        """Regression: calculate_hurst_exponent must never return NaN or Inf."""
        for seed in range(5):
            prices = _random_walk_prices(n=200, seed=seed)
            h = calculate_hurst_exponent(prices, max_lag=20)
            assert np.isfinite(h), f"Hurst exponent must be finite, got {h} (seed={seed})"
