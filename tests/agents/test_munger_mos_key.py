"""Tests for R52: Munger facts builder used a non-existent 'mos_to_reasonable' key."""

from src.agents.charlie_munger import make_munger_facts_bundle


def _analysis(mos: float) -> dict:
    return {
        "signal": "bullish",
        "score": 7,
        "moat_analysis": {"score": 8},
        "management_analysis": {"score": 7},
        "predictability_analysis": {"score": 6},
        "valuation_analysis": {
            "score": 5,
            "margin_of_safety_vs_fair_value": mos,
            "fcf_yield": 0.06,
        },
    }


class TestMungerMosKey:
    def test_positive_mos_yields_mos_positive_true(self):
        """
        R52: With margin_of_safety_vs_fair_value=0.25 (25% upside), mos_positive must be True.
        Before the fix: val.get('mos_to_reasonable') always returned None → mos_positive=False.
        """
        result = make_munger_facts_bundle(_analysis(mos=0.25))
        flags = result.get("flags") or result
        mos_positive = flags.get("mos_positive")
        assert mos_positive is True, f"margin_of_safety=0.25 (undervalued) must produce mos_positive=True. Got: {mos_positive}"

    def test_negative_mos_yields_mos_positive_false(self):
        """With negative MoS (overvalued), mos_positive must be False."""
        result = make_munger_facts_bundle(_analysis(mos=-0.15))
        flags = result.get("flags") or result
        mos_positive = flags.get("mos_positive")
        assert mos_positive is False, f"margin_of_safety=-0.15 (overvalued) must produce mos_positive=False. Got: {mos_positive}"

    def test_zero_mos_yields_mos_positive_false(self):
        """MoS of exactly 0 is not positive."""
        result = make_munger_facts_bundle(_analysis(mos=0.0))
        flags = result.get("flags") or result
        assert flags.get("mos_positive") is False
