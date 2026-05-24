"""Tests for risk manager position sizing logic: volatility-adjusted limits and correlation penalties."""

from unittest.mock import patch

import pytest

from src.agents.risk_manager import (
    calculate_correlation_multiplier,
    calculate_volatility_adjusted_limit,
    risk_management_agent,
)
from src.data.models import Price


def _make_prices(n: int = 5, base: float = 100.0, step: float = 0.01) -> list[Price]:
    """Deterministic price series with minimal variation for predictable volatility."""
    return [
        Price(
            open=base + i * step,
            close=base + i * step,
            high=base + i * step + 0.5,
            low=base + i * step - 0.5,
            volume=1_000_000,
            time=f"2024-01-{i + 2:02d}T00:00:00",
        )
        for i in range(n)
    ]


def _make_state(tickers: list[str], cash: float, positions: dict | None = None) -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": tickers,
            "portfolio": {
                "cash": cash,
                "positions": positions or {},
            },
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


class TestCalculateVolatilityAdjustedLimit:
    def test_typical_vol_returns_near_20pct(self):
        """~20% annualized vol stays within the 15-25% band."""
        limit = calculate_volatility_adjusted_limit(0.20)
        assert 0.15 <= limit <= 0.25

    def test_high_vol_ticker_gets_lower_limit(self):
        high_limit = calculate_volatility_adjusted_limit(0.60)
        low_limit = calculate_volatility_adjusted_limit(0.10)
        assert high_limit < low_limit

    def test_low_vol_gets_above_baseline(self):
        """Stable stocks (<15% vol) are allowed higher than 20% baseline."""
        assert calculate_volatility_adjusted_limit(0.10) > 0.20

    def test_very_high_vol_still_bounded(self):
        """Even extreme volatility produces a positive, capped result."""
        limit = calculate_volatility_adjusted_limit(0.90)
        assert 0.0 < limit <= 0.20

    def test_vol_multiplier_matches_docstring_bands(self):
        """
        Regression for F3: piecewise-linear, continuous at all boundaries.
        - Low (<0.15): 25%
        - Medium (0.15–0.30): linear 20% → 15%
        - High (0.30–0.50): linear 15% → 10%
        - Very high (≥0.50): 10%
        """
        base = 0.20
        # Low vol: flat at 25%
        assert calculate_volatility_adjusted_limit(0.10, base_limit=base) == pytest.approx(0.25, abs=1e-6)

        # Medium band endpoints
        assert calculate_volatility_adjusted_limit(0.15, base_limit=base) == pytest.approx(0.20, abs=1e-4)
        assert calculate_volatility_adjusted_limit(0.2999, base_limit=base) == pytest.approx(0.15, abs=0.005)

        # Boundary at vol=0.30: high band starts at 15%
        assert calculate_volatility_adjusted_limit(0.30, base_limit=base) == pytest.approx(0.15, abs=1e-4)

        # High band endpoints — continuous at 0.50
        assert calculate_volatility_adjusted_limit(0.4999, base_limit=base) == pytest.approx(0.10, abs=0.005)
        assert calculate_volatility_adjusted_limit(0.50, base_limit=base) == pytest.approx(0.10, abs=1e-4)

        # Very high: flat at 10%
        assert calculate_volatility_adjusted_limit(0.80, base_limit=base) == pytest.approx(0.10, abs=1e-4)

        # Strictly monotonically non-increasing across bands
        limits = [calculate_volatility_adjusted_limit(v, base_limit=base) for v in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]]
        for i in range(len(limits) - 1):
            assert limits[i] >= limits[i + 1], f"Non-monotonic at index {i}: {limits[i]} < {limits[i + 1]}"


class TestCalculateCorrelationMultiplier:
    def test_correlation_penalty_applied(self):
        """Highly correlated portfolio gets a lower multiplier than uncorrelated."""
        assert calculate_correlation_multiplier(0.85) < calculate_correlation_multiplier(0.05)

    def test_very_high_correlation_hard_reduction(self):
        assert calculate_correlation_multiplier(0.85) == pytest.approx(0.70)

    def test_very_low_correlation_bonus(self):
        assert calculate_correlation_multiplier(0.10) == pytest.approx(1.10)

    def test_moderate_correlation_neutral(self):
        assert calculate_correlation_multiplier(0.50) == pytest.approx(1.00)


class TestRiskManagementAgent:
    def test_zero_cash_caps_limit_to_zero(self):
        """Portfolio with zero cash → remaining_position_limit is 0 regardless of vol."""
        state = _make_state(["AAPL"], cash=0.0)
        with patch("src.agents.risk_manager.get_prices", return_value=_make_prices()):
            result = risk_management_agent(state)

        limit = result["data"]["analyst_signals"]["risk_management_agent"]["AAPL"]["remaining_position_limit"]
        assert limit == pytest.approx(0.0)

    def test_existing_position_reduces_remaining_limit(self):
        """A long position already held reduces the remaining headroom vs a clean slate."""
        # Baseline: 100k cash, no existing position
        state_clean = _make_state(["AAPL"], cash=100_000.0)
        with patch("src.agents.risk_manager.get_prices", return_value=_make_prices()):
            result_clean = risk_management_agent(state_clean)
        limit_clean = result_clean["data"]["analyst_signals"]["risk_management_agent"]["AAPL"]["remaining_position_limit"]

        # With 50 shares already held, cash reduced accordingly
        state_held = _make_state(
            ["AAPL"],
            cash=50_000.0,
            positions={"AAPL": {"long": 50, "short": 0}},
        )
        with patch("src.agents.risk_manager.get_prices", return_value=_make_prices()):
            result_held = risk_management_agent(state_held)
        limit_held = result_held["data"]["analyst_signals"]["risk_management_agent"]["AAPL"]["remaining_position_limit"]

        assert limit_held < limit_clean
