"""Tests for src/utils/signal_quality.py data_quality_multiplier and helpers."""

from src.utils.signal_quality import _extract_reasoning_text, data_quality_multiplier


class TestExtractReasoningText:
    def test_string_passthrough(self):
        assert "strong fundamentals" in _extract_reasoning_text("strong fundamentals")

    def test_dict_concatenates_values(self):
        r = {"a": "ROE: N/A", "b": "Net Margin: 15%"}
        text = _extract_reasoning_text(r)
        assert "N/A" in text
        assert "Net Margin" in text

    def test_nested_dict_descends(self):
        r = {"sub": {"details": "Revenue Growth: N/A, Earnings Growth: N/A"}}
        text = _extract_reasoning_text(r)
        assert "N/A" in text

    def test_list_joins_elements(self):
        r = ["bullish outlook", "N/A for FCF"]
        text = _extract_reasoning_text(r)
        assert "bullish outlook" in text
        assert "N/A" in text

    def test_none_returns_empty(self):
        assert _extract_reasoning_text(None) == ""

    def test_number_coerced_to_string(self):
        assert _extract_reasoning_text(42) == "42"


class TestDataQualityMultiplier:
    def test_clean_reasoning_returns_one(self):
        sig = {"signal": "bullish", "confidence": 80, "reasoning": "ROE 28%, FCF yield +15%, strong balance sheet"}
        assert data_quality_multiplier(sig) == 1.0

    def test_empty_reasoning_returns_one(self):
        sig = {"signal": "bearish", "confidence": 70, "reasoning": ""}
        assert data_quality_multiplier(sig) == 1.0

    def test_missing_reasoning_key_returns_one(self):
        sig = {"signal": "bearish", "confidence": 70}
        assert data_quality_multiplier(sig) == 1.0

    def test_non_dict_sig_data_returns_one(self):
        assert data_quality_multiplier("not a dict") == 1.0  # type: ignore[arg-type]
        assert data_quality_multiplier(None) == 1.0  # type: ignore[arg-type]

    def test_two_phrases_neutralize_regardless_of_signal(self):
        # "cannot compute" and "missing data" → count=2 → 0.0
        sig = {"signal": "bearish", "confidence": 75, "reasoning": "Cannot compute Graham Number. Missing data for balance sheet."}
        assert data_quality_multiplier(sig) == 0.0

    def test_two_na_occurrences_neutralize(self):
        sig = {"signal": "bearish", "confidence": 75, "reasoning": "ROE: N/A, Net Margin: N/A, Op Margin: 12%"}
        assert data_quality_multiplier(sig) == 0.0

    def test_single_phrase_with_directional_signal_discounts(self):
        # one "missing critical" phrase + bearish → 0.5 (partial discount, not full zero)
        sig = {"signal": "bearish", "confidence": 10, "reasoning": "Missing critical financial data for analysis."}
        assert data_quality_multiplier(sig) == 0.5

    def test_single_phrase_with_neutral_signal_passes(self):
        # one phrase + neutral → signal is already stance-zero; let it through at 1.0
        sig = {"signal": "neutral", "confidence": 50, "reasoning": "Insufficient data for valuation."}
        assert data_quality_multiplier(sig) == 1.0

    def test_dict_reasoning_extracts_nested_details(self):
        """Exact shape from fundamentals_analyst_agent — N/A lives in 'details' sub-values."""
        reasoning = {
            "profitability_signal": {"signal": "bearish", "details": "ROE: N/A, Net Margin: N/A, Op Margin: N/A"},
            "growth_signal": {"signal": "bearish", "details": "Revenue Growth: N/A, Earnings Growth: N/A"},
            "financial_health_signal": {"signal": "neutral", "details": "Current Ratio: N/A, D/E: N/A"},
            "price_ratios_signal": {"signal": "bearish", "details": "P/E: N/A, P/B: N/A, P/S: N/A"},
        }
        sig = {"signal": "bearish", "confidence": 75, "reasoning": reasoning}
        # Multiple "N/A" occurrences found in nested details → 0.0
        assert data_quality_multiplier(sig) == 0.0

    def test_case_insensitive_matching(self):
        # "N/A" uppercased vs phrase list entry "n/a" — must match
        sig = {"signal": "bearish", "confidence": 60, "reasoning": "ROE: N/A and Net Margin: N/A."}
        assert data_quality_multiplier(sig) == 0.0

    def test_partial_word_not_matched(self):
        # "data" alone does not match "missing data" as a phrase
        sig = {"signal": "bearish", "confidence": 60, "reasoning": "Fundamental data is the core of value investing."}
        assert data_quality_multiplier(sig) == 1.0

    def test_clean_reasoning_with_word_not_in_phrase_list(self):
        # "limits" alone should not trigger; only "incomplete data" as a full phrase would
        sig = {"signal": "bullish", "confidence": 70, "reasoning": "Growth limits competition; strong moat."}
        assert data_quality_multiplier(sig) == 1.0

    def test_ben_graham_cannot_compute_bearish_100(self):
        """Replicates the exact failure from cycle-2026-05-20 ben_graham_agent entry."""
        reasoning = "- Cannot compute Graham Number (EPS or Book Value missing/<=0).\n- Cannot compute current ratio.\n- Company did not pay dividends."
        sig = {"signal": "bearish", "confidence": 100, "reasoning": reasoning}
        assert data_quality_multiplier(sig) == 0.0
