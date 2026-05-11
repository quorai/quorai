"""Sanity checks for the ANALYST_CONFIG registry."""

import pytest

pytest.importorskip("langchain_core", reason="langchain_core not available in this Python environment")

from src.utils.analysts import ANALYST_CONFIG  # noqa: E402

_KNOWN_GROUPS = {
    "deep_value",
    "quality_compounders",
    "growth_and_catalyst",
    "macro_and_cycle",
    "quant_systematic",
    "sentiment_and_analytical",
}


def test_expected_analyst_count():
    assert len(ANALYST_CONFIG) == 25, f"Expected 25 analysts, got {len(ANALYST_CONFIG)}"


def test_all_agent_funcs_are_callable():
    for key, config in ANALYST_CONFIG.items():
        func = config["agent_func"]
        assert callable(func), f"agent_func for '{key}' is not callable"


def test_order_values_are_unique():
    orders = [config["order"] for config in ANALYST_CONFIG.values()]
    assert len(orders) == len(set(orders)), f"Duplicate order values: {sorted(orders)}"


def test_every_analyst_has_strategy_group():
    for key, config in ANALYST_CONFIG.items():
        assert "strategy_group" in config, f"Missing strategy_group for '{key}'"
        assert config["strategy_group"] in _KNOWN_GROUPS, f"Unknown strategy_group '{config['strategy_group']}' for '{key}'"
