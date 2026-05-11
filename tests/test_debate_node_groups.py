"""Unit tests for the group-aggregation helper in debate_node — no LLM calls."""

import pytest

pytest.importorskip("langchain_core", reason="langchain_core not available in this Python environment")

from src.agents.debate_node import _aggregate_to_groups  # noqa: E402


def _make_signals(entries: list[tuple[str, str, float, str]]) -> dict[str, dict]:
    """Build a minimal analyst_signals dict from (agent_key, signal, confidence, reasoning) tuples."""
    out = {}
    for agent_key, signal, confidence, reasoning in entries:
        out[f"{agent_key}_agent"] = {"AAPL": {"signal": signal, "confidence": confidence, "reasoning": reasoning}}
    return out


def test_unanimous_bullish_group():
    signals = _make_signals(
        [
            ("ben_graham", "bullish", 80.0, "NCAV > market cap"),
            ("michael_burry", "bullish", 75.0, "deep discount to intrinsic value"),
            ("mohnish_pabrai", "bullish", 70.0, "FCF yield 12%"),
            ("joel_greenblatt", "bullish", 85.0, "top magic formula rank"),
        ]
    )
    result = _aggregate_to_groups(signals, ["AAPL"])
    dv = result["AAPL"]["deep_value"]
    assert dv["signal"] == "bullish"
    assert dv["dissent"] == 0
    assert dv["confidence"] == pytest.approx(77.5, abs=1)


def test_majority_sets_group_signal_with_dissent():
    signals = _make_signals(
        [
            ("ben_graham", "bullish", 80.0, "cheap on NCAV"),
            ("michael_burry", "bullish", 75.0, "hated stock"),
            ("mohnish_pabrai", "bearish", 60.0, "FCF yield insufficient"),
            ("joel_greenblatt", "neutral", 50.0, "mixed signals"),
        ]
    )
    result = _aggregate_to_groups(signals, ["AAPL"])
    dv = result["AAPL"]["deep_value"]
    # Weighted: (80+75-60+0) / (80+75+60+50) = 95/265 ≈ 0.358 → bullish
    assert dv["signal"] == "bullish"
    assert dv["dissent"] == 1  # pabrai disagrees


def test_bearish_dominance():
    signals = _make_signals(
        [
            ("ben_graham", "bearish", 90.0, "price above Graham Number"),
            ("michael_burry", "bearish", 85.0, "overleveraged"),
            ("mohnish_pabrai", "bearish", 80.0, "no FCF yield"),
            ("joel_greenblatt", "bearish", 75.0, "low ROIC"),
        ]
    )
    result = _aggregate_to_groups(signals, ["AAPL"])
    dv = result["AAPL"]["deep_value"]
    assert dv["signal"] == "bearish"
    assert dv["dissent"] == 0


def test_neutral_group_near_zero_stance():
    # Two bullish, two bearish of equal confidence → weighted stance ≈ 0 → neutral
    signals = _make_signals(
        [
            ("ben_graham", "bullish", 60.0, "reason A"),
            ("michael_burry", "bullish", 60.0, "reason B"),
            ("mohnish_pabrai", "bearish", 60.0, "reason C"),
            ("joel_greenblatt", "bearish", 60.0, "reason D"),
        ]
    )
    result = _aggregate_to_groups(signals, ["AAPL"])
    dv = result["AAPL"]["deep_value"]
    assert dv["signal"] == "neutral"


def test_unknown_agent_ignored():
    signals = _make_signals(
        [
            ("ben_graham", "bullish", 80.0, "cheap"),
            ("unknown_persona", "bearish", 99.0, "should be ignored"),
        ]
    )
    result = _aggregate_to_groups(signals, ["AAPL"])
    # unknown_persona has no group mapping — should not appear in result
    dv = result["AAPL"]["deep_value"]
    assert dv["signal"] == "bullish"
    # No group for unknown persona
    assert all("unknown" not in g for g in result["AAPL"])


def test_key_args_capped_at_two():
    signals = _make_signals(
        [
            ("ben_graham", "bullish", 80.0, "reason one"),
            ("michael_burry", "bullish", 75.0, "reason two"),
            ("mohnish_pabrai", "bullish", 70.0, "reason three"),
            ("joel_greenblatt", "bullish", 85.0, "reason four"),
        ]
    )
    result = _aggregate_to_groups(signals, ["AAPL"])
    assert len(result["AAPL"]["deep_value"]["key_args"]) <= 2


def test_risk_management_agent_excluded():
    signals = _make_signals(
        [
            ("ben_graham", "bullish", 80.0, "cheap"),
        ]
    )
    signals["risk_management_agent"] = {"AAPL": {"signal": "bullish", "confidence": 100.0, "reasoning": "ignore me"}}
    result = _aggregate_to_groups(signals, ["AAPL"])
    # risk_management_agent must not create a group entry
    assert "risk_management" not in result["AAPL"]
