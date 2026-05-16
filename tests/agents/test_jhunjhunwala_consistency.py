"""Tests for R51: jhunjhunwala growth-consistency was inverted (newest-first array)."""

from unittest.mock import MagicMock

from src.agents.rakesh_jhunjhunwala import analyze_growth


def _item(revenue, net_income=100.0):
    m = MagicMock()
    m.revenue = revenue
    m.net_income = net_income
    return m


def _revenue_items(revenues_newest_first):
    """Build financial_line_items with revenues in newest-first order."""
    return [_item(r) for r in revenues_newest_first]


class TestJhunjhunwalaConsistency:
    def test_steady_growth_scores_consistent(self):
        """
        R51: revenues newest-first [110, 105, 100, 95, 90] = steadily growing company.
        Before the fix: the code counted growth years as 'declining_years' and
        consistency_ratio = 1 - growth_ratio → rewarded decliners.
        After the fix: growing_years / total → consistency_ratio = 4/4 = 1.0 → score +1.
        """
        items = _revenue_items([110, 105, 100, 95, 90])
        result = analyze_growth(items)

        assert result["score"] >= 1, f"Steady-growth revenues (newest-first: 110→90) should score consistent-growth (+1). Got score={result['score']}, details: {result['details']}"
        assert "Consistent" in result["details"], f"Expected 'Consistent' in reasoning for a steadily growing company. Got: {result['details']}"

    def test_steady_decline_scores_inconsistent(self):
        """
        R51: revenues newest-first [90, 95, 100, 105, 110] = steadily declining company.
        After the fix: growing_years = 0 → consistency_ratio = 0.0 → no growth-consistency score.
        """
        items = _revenue_items([90, 95, 100, 105, 110])
        result = analyze_growth(items)

        assert "Consistent growth" not in result["details"], f"Declining revenues (newest-first: 90→110) must NOT claim 'Consistent growth'. Details: {result['details']}"

    def test_mixed_revenues_partial_consistency(self):
        """Verify the ratio is computed correctly for a 3-year mixed dataset."""
        # newest-first: [110, 95, 100] → yr1 pair: 110>95 (growth), yr2 pair: 95<100 (decline)
        # growing_years=1, total_pairs=2 → ratio=0.5 → below 0.8 threshold → no score
        items = _revenue_items([110, 95, 100])
        result = analyze_growth(items)

        assert "Consistent growth" not in result["details"], f"50% growth consistency should not produce 'Consistent growth'. Got: {result['details']}"
