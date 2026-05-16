"""Tests for risk profile integration in calculate_volatility_adjusted_limit."""

import pytest

from src.agents.risk_manager import calculate_volatility_adjusted_limit


def test_default_base_backward_compat():
    """No base_limit kwarg → old behaviour, base = 0.20."""
    result_default = calculate_volatility_adjusted_limit(0.10)
    result_explicit = calculate_volatility_adjusted_limit(0.10, base_limit=0.20)
    assert result_default == pytest.approx(result_explicit)


def test_speculative_base_scales_proportionally():
    """speculative base_limit=0.50 should return 2.5x the balanced (0.20) result."""
    vol = 0.10
    balanced = calculate_volatility_adjusted_limit(vol, base_limit=0.20)
    speculative = calculate_volatility_adjusted_limit(vol, base_limit=0.50)
    assert speculative == pytest.approx(balanced * 2.5)


def test_conservative_base_halves_output():
    vol = 0.20
    balanced = calculate_volatility_adjusted_limit(vol, base_limit=0.20)
    conservative = calculate_volatility_adjusted_limit(vol, base_limit=0.10)
    assert conservative == pytest.approx(balanced * 0.5)
