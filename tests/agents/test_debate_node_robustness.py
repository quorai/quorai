"""Tests for debate_node robustness against malformed LLM outputs."""

import pytest

pytest.importorskip("langchain_core", reason="langchain_core not available")

from src.agents.debate_node import _aggregate_to_groups, _compute_panel_stats


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


class TestSignalPollutionFilter:
    def test_polluted_signal_neutralized_in_aggregation(self):
        """A fundamentals_analyst N/A-heavy bearish signal must not tip the group bearish.

        Setup: two agents in the same group (sentiment_and_analytical).
          - fundamentals_analyst_agent: bearish conf=75, dict reasoning with N/A in every sub-field.
          - valuation_analyst_agent: bullish conf=75, clean reasoning.
        Expected: group signal = bullish, because the polluted agent contributed zero weight.
        """
        analyst_signals = {
            "fundamentals_analyst_agent": {
                "AAPL": {
                    "signal": "bearish",
                    "confidence": 75.0,
                    "reasoning": {
                        "profitability_signal": {"signal": "bearish", "details": "ROE: N/A, Net Margin: N/A"},
                        "growth_signal": {"signal": "bearish", "details": "Revenue Growth: N/A"},
                        "financial_health_signal": {"signal": "neutral", "details": "Current Ratio: N/A"},
                        "price_ratios_signal": {"signal": "bearish", "details": "P/E: N/A, P/B: N/A"},
                    },
                }
            },
            "valuation_analyst_agent": {
                "AAPL": {
                    "signal": "bullish",
                    "confidence": 75.0,
                    "reasoning": "DCF value $320, market price $290, 10% discount to intrinsic value. Strong free cash flow.",
                }
            },
        }
        result = _aggregate_to_groups(analyst_signals, ["AAPL"])
        sa = result["AAPL"].get("sentiment_and_analytical")
        assert sa is not None, f"Expected sentiment_and_analytical group, got: {list(result['AAPL'].keys())}"
        assert sa["signal"] == "bullish", f"Polluted fundamentals signal should be zeroed, leaving only the bullish valuation_analyst. Got: {sa}"

    def test_clean_directional_signal_not_neutralized(self):
        """An analyst with clean reasoning must NOT be zeroed, even if using words like 'limited'."""
        analyst_signals = {
            "fundamentals_analyst_agent": {
                "AAPL": {
                    "signal": "bullish",
                    "confidence": 80.0,
                    "reasoning": "ROE 22%, Net Margin 18%, strong growth despite limited macro tailwinds.",
                }
            }
        }
        result = _aggregate_to_groups(analyst_signals, ["AAPL"])
        sa = result["AAPL"].get("sentiment_and_analytical")
        assert sa is not None
        assert sa["signal"] == "bullish", f"Clean bullish signal should not be filtered. Got: {sa}"

    def test_ben_graham_missing_data_zeroed(self):
        """ben_graham cannot-compute bearish-100 must contribute zero weight."""
        analyst_signals = {
            "ben_graham_agent": {
                "AAPL": {
                    "signal": "bearish",
                    "confidence": 100.0,
                    "reasoning": "- Cannot compute Graham Number (EPS or Book Value missing).\n- Cannot compute current ratio.",
                }
            },
            "mohnish_pabrai_agent": {
                "AAPL": {
                    "signal": "bullish",
                    "confidence": 60.0,
                    "reasoning": "FCF yield 12%, deep discount to book value.",
                }
            },
        }
        result = _aggregate_to_groups(analyst_signals, ["AAPL"])
        dv = result["AAPL"]["deep_value"]
        assert dv["signal"] == "bullish", f"ben_graham missing-data bearish-100 should be zeroed. Got: {dv}"


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


class TestDirectionalAggregation:
    """Tests for the directional-only weighted stance + participation floor."""

    def test_heavy_neutral_mass_does_not_flip_bearish_to_neutral_when_participation_met(self):
        """5 bearish at conf=80, 10 neutrals at conf=60: participation = 400/1000 = 0.4 ≥ 1/3 → bearish."""
        # All in sentiment_and_analytical (valuation + fundamentals both map there)
        # Use agents that map to the same group: fundamentals_analyst (bearish) + valuation_analyst (neutral×many)
        # We only have 2 real agents for this group so test a group with more members.
        # Instead use deep_value: ben_graham + mohnish_pabrai (bearish), michael_burry + warren_buffett (neutral).
        signals = {
            "ben_graham_agent": {"AAPL": {"signal": "bearish", "confidence": 80.0, "reasoning": "undervalued"}},
            "mohnish_pabrai_agent": {"AAPL": {"signal": "bearish", "confidence": 80.0, "reasoning": "cheap"}},
            "michael_burry_agent": {"AAPL": {"signal": "neutral", "confidence": 60.0, "reasoning": "wait"}},
            "warren_buffett_agent": {"AAPL": {"signal": "neutral", "confidence": 60.0, "reasoning": "fair"}},
        }
        result = _aggregate_to_groups(signals, ["AAPL"])
        dv = result["AAPL"]["deep_value"]
        # dir_weight = 160, total_weight = 280, participation = 160/280 ≈ 0.57 ≥ 1/3
        # weighted_stance = -160/160 = -1.0 → bearish
        assert dv["signal"] == "bearish", f"Expected bearish, got {dv}"

    def test_thin_directional_participation_yields_neutral(self):
        """1 bearish at conf=60, 10 neutrals at conf=80: participation = 60/860 ≈ 0.07 < 1/3 → neutral."""
        signals = {
            "ben_graham_agent": {"AAPL": {"signal": "bearish", "confidence": 60.0, "reasoning": "cheap"}},
            "mohnish_pabrai_agent": {"AAPL": {"signal": "neutral", "confidence": 80.0, "reasoning": "fair"}},
            "michael_burry_agent": {"AAPL": {"signal": "neutral", "confidence": 80.0, "reasoning": "wait"}},
            "warren_buffett_agent": {"AAPL": {"signal": "neutral", "confidence": 80.0, "reasoning": "hold"}},
        }
        result = _aggregate_to_groups(signals, ["AAPL"])
        dv = result["AAPL"]["deep_value"]
        # dir_weight = 60, total_weight = 60+80+80+80 = 300, participation ≈ 0.2 < 1/3 → neutral
        assert dv["signal"] == "neutral", f"Expected neutral due to thin participation, got {dv}"


class TestPanelStats:
    """Tests for _compute_panel_stats panel-wide tilt computation."""

    def _make_full_panel(self, entries: list[tuple[str, str, float]]) -> dict[str, dict]:
        """Build analyst_signals from [(agent_name, signal, confidence)] triples."""
        out = {}
        for agent_name, signal, confidence in entries:
            out[f"{agent_name}_agent"] = {"AAPL": {"signal": signal, "confidence": confidence, "reasoning": "clean reasoning"}}
        return out

    def test_net_bearish_panel_has_negative_tilt(self):
        """8 bearish, 2 bullish, 15 neutral → tilt strongly negative (bearish panel)."""
        # Use real analyst names that map to valid groups.
        entries = [
            # 2 bullish
            ("warren_buffett", "bullish", 70.0),
            ("charlie_munger", "bullish", 70.0),
            # 8 bearish
            ("ben_graham", "bearish", 70.0),
            ("mohnish_pabrai", "bearish", 70.0),
            ("michael_burry", "bearish", 70.0),
            ("ray_dalio", "bearish", 70.0),
            ("nassim_taleb", "bearish", 70.0),
            ("bill_ackman", "bearish", 70.0),
            ("howard_marks", "bearish", 70.0),
            ("joel_greenblatt", "bearish", 70.0),
            # 3 neutral (fewer to keep test concise; tilt formula only uses directional)
            ("jim_simons", "neutral", 70.0),
            ("ed_seykota", "neutral", 70.0),
            ("cathie_wood", "neutral", 70.0),
        ]
        signals = self._make_full_panel(entries)
        stats = _compute_panel_stats(signals, ["AAPL"])
        aapl = stats["AAPL"]
        assert aapl["bearish"] == 8
        assert aapl["bullish"] == 2
        assert aapl["neutral"] == 3
        # tilt = (bull_conf - bear_conf) / (bull_conf + bear_conf)
        # = (2×70 - 8×70) / (2×70 + 8×70) = (140-560)/(700) = -0.6
        assert aapl["tilt"] == pytest.approx(-0.6, abs=0.01), f"Expected tilt ≈ -0.6, got {aapl['tilt']}"

    def test_balanced_panel_tilt_near_zero(self):
        """Equal bullish and bearish with equal confidence → tilt == 0.0."""
        entries = [
            ("warren_buffett", "bullish", 80.0),
            ("ben_graham", "bearish", 80.0),
            ("jim_simons", "neutral", 80.0),
        ]
        signals = self._make_full_panel(entries)
        stats = _compute_panel_stats(signals, ["AAPL"])
        assert stats["AAPL"]["tilt"] == pytest.approx(0.0, abs=0.001)

    def test_all_neutral_panel_tilt_zero(self):
        """No directional votes → tilt == 0.0 (safe default)."""
        entries = [
            ("warren_buffett", "neutral", 80.0),
            ("ben_graham", "neutral", 60.0),
        ]
        signals = self._make_full_panel(entries)
        stats = _compute_panel_stats(signals, ["AAPL"])
        assert stats["AAPL"]["tilt"] == 0.0

    def test_empty_signals_returns_zero_stats(self):
        """No relevant analysts → all-zero counts and tilt=0."""
        stats = _compute_panel_stats({}, ["AAPL"])
        assert stats["AAPL"] == {"bullish": 0, "bearish": 0, "neutral": 0, "n": 0, "tilt": 0.0}

    def test_data_quality_discount_reduces_bearish_tilt(self):
        """A bearish analyst with one missing-data phrase gets dq=0.5 → partial weight in tilt."""
        entries_clean = [("warren_buffett", "bullish", 80.0), ("ben_graham", "bearish", 80.0)]
        # Add a discounted bearish: tilt should be more negative than the balanced case but less than -1.0.
        discounted_signal = {
            "michael_burry_agent": {
                "AAPL": {
                    "signal": "bearish",
                    "confidence": 80.0,
                    "reasoning": "Missing critical data. Bearish outlook.",
                }
            }
        }
        full_signals = self._make_full_panel(entries_clean)
        full_signals.update(discounted_signal)
        stats = _compute_panel_stats(full_signals, ["AAPL"])
        tilt = stats["AAPL"]["tilt"]
        # bull_conf=80, bear_conf=80 + 80×0.5=40 → bull=80, bear=120 → tilt=(80-120)/(200)=-0.2
        assert tilt == pytest.approx(-0.2, abs=0.01), f"Expected tilt ≈ -0.2, got {tilt}"
