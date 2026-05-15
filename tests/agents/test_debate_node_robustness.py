"""Tests for debate_node robustness against malformed LLM outputs."""

import pytest

pytest.importorskip("langchain_core", reason="langchain_core not available")

from src.agents.debate_node import _aggregate_to_groups


def _make_signals(**entries) -> dict[str, dict]:
    """Build analyst_signals from {agent_key: (signal, confidence)} pairs."""
    out = {}
    for agent_key, (signal, confidence) in entries.items():
        out[f"{agent_key}_agent"] = {"AAPL": {"signal": signal, "confidence": confidence, "reasoning": ""}}
    return out


class TestUnknownSignalValue:
    def test_unknown_signal_does_not_raise(self):
        signals = _make_signals(
            ben_graham=("buy", 80.0),  # "buy" is not a valid stance
            michael_burry=("bullish", 70.0),
        )
        # Should not raise KeyError
        result = _aggregate_to_groups(signals, ["AAPL"])
        assert "deep_value" in result["AAPL"]

    def test_unknown_signal_contributes_zero_weight(self):
        # "hold" has no entry in _STANCE_SCORE — treated as neutral (0)
        # ben_graham: bullish 80, michael_burry: hold (unknown→0) 80
        # weighted_stance = (1*80 + 0*80) / 160 = 0.5 → bullish (>0.25)
        signals = _make_signals(
            ben_graham=("bullish", 80.0),
            michael_burry=("hold", 80.0),
        )
        result = _aggregate_to_groups(signals, ["AAPL"])
        assert result["AAPL"]["deep_value"]["signal"] == "bullish"

    def test_all_unknown_signals_yield_neutral(self):
        # All unknown signals → stance == 0 → neutral
        signals = _make_signals(
            ben_graham=("buy", 80.0),
            michael_burry=("sell", 70.0),
        )
        result = _aggregate_to_groups(signals, ["AAPL"])
        assert result["AAPL"]["deep_value"]["signal"] == "neutral"


class TestConfidenceClamping:
    def test_over_scale_confidence_is_clamped(self):
        # confidence=500 should behave identically to confidence=100 (max on 0-100 scale)
        signals_over = _make_signals(ben_graham=("bullish", 500.0))
        signals_cap = _make_signals(ben_graham=("bullish", 100.0))
        result_over = _aggregate_to_groups(signals_over, ["AAPL"])
        result_cap = _aggregate_to_groups(signals_cap, ["AAPL"])
        assert result_over["AAPL"]["deep_value"]["signal"] == result_cap["AAPL"]["deep_value"]["signal"]
        assert result_over["AAPL"]["deep_value"]["confidence"] == pytest.approx(result_cap["AAPL"]["deep_value"]["confidence"], abs=0.1)

    def test_negative_confidence_is_clamped_to_zero(self):
        # negative confidence → clipped to 0 → no contribution to stance
        signals = _make_signals(ben_graham=("bullish", -50.0))
        result = _aggregate_to_groups(signals, ["AAPL"])
        # zero contribution from negative confidence → neutral
        assert result["AAPL"]["deep_value"]["signal"] == "neutral"

    def test_nan_confidence_treated_as_zero(self):
        signals = _make_signals(ben_graham=("bullish", float("nan")))
        result = _aggregate_to_groups(signals, ["AAPL"])
        # NaN confidence → 0 contribution → neutral stance
        assert result["AAPL"]["deep_value"]["signal"] == "neutral"
