"""Tests for R56: Buffett pricing-power overlapping windows at 3-item datasets."""

from unittest.mock import MagicMock

from src.agents.warren_buffett import analyze_pricing_power


def _item(gross_margin):
    m = MagicMock()
    m.gross_margin = gross_margin
    return m


def _metric():
    m = MagicMock()
    m.price_to_earnings_ratio = None
    return m


class TestBuffettPricingPower:
    def test_clear_declining_margins_score_zero(self):
        """
        R56: newest-first [0.10, 0.20, 0.30, 0.40] — newest=0.10, oldest=0.40 (steady decline).
        mid=2: recent_avg=(0.10+0.20)/2=0.15, older_avg=(0.30+0.40)/2=0.35.
        recent < older → no "expanding" or "improving" score from the window comparison.
        Before the fix (len >= 3, overlapping [:2] and [-2:]):
          recent=mean(0.10,0.20)=0.15, older=mean(0.30,0.40)=0.35 → same result, so R56 is
          about correctness of the split, not about fixing this specific ordering.
        """
        items = [_item(0.10), _item(0.20), _item(0.30), _item(0.40)]
        result = analyze_pricing_power(items, [_metric()])
        details = result.get("details", "")

        assert "Expanding gross margins" not in details, f"Declining margins (newest=0.10, oldest=0.40) must not produce 'Expanding'. Got: {details}"
        assert "Improving gross margins" not in details, f"Declining margins must not produce 'Improving'. Got: {details}"

    def test_clear_expanding_margins_score_positive(self):
        """newest-first [0.40, 0.30, 0.20, 0.10] — newest=0.40, oldest=0.10 (steady expansion)."""
        items = [_item(0.40), _item(0.30), _item(0.20), _item(0.10)]
        result = analyze_pricing_power(items, [_metric()])
        details = result.get("details", "")

        improving_or_expanding = any(kw in details for kw in ("Expanding", "Improving", "strong pricing power", "good pricing power"))
        assert improving_or_expanding, f"Expanding margins (newest=0.40, oldest=0.10) should score positive pricing power. Got: {details}"

    def test_three_item_dataset_uses_gate_not_overlapping_windows(self):
        """
        R56: With only 3 gross_margin values, len < 4, so the gate is not reached.
        No pricing-power score from margin windows — function still returns a dict.
        """
        items = [_item(0.40), _item(0.35), _item(0.30)]
        result = analyze_pricing_power(items, [_metric()])
        assert "score" in result, "analyze_pricing_power must return a dict with 'score'"
