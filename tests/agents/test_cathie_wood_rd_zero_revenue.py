"""Tests for R37 (regression): Cathie Wood R&D ratio division is guarded against zero revenue."""

from unittest.mock import MagicMock


def _make_financial_item(**kwargs):
    item = MagicMock()
    item.revenue = kwargs.get("revenue", 1_000_000.0)
    item.research_and_development = kwargs.get("rd", 100_000.0)
    item.gross_margin = kwargs.get("gross_margin", 0.5)
    item.operating_margin = kwargs.get("op_margin", 0.2)
    item.capital_expenditure = kwargs.get("capex", -50_000.0)
    item.free_cash_flow = kwargs.get("fcf", 200_000.0)
    item.dividends_and_other_cash_distributions = kwargs.get("dividends", 0.0)
    return item


class TestCathieWoodRDZeroRevenue:
    def test_zero_revenue_in_disruptive_analysis_does_not_crash(self):
        """
        analyze_disruptive_potential with revenue=0 must not raise ZeroDivisionError.
        The revenues list comprehension filters `if item.revenue`, so revenue=0 items
        are excluded — this test confirms that guard is in place.
        """
        from src.agents.cathie_wood import analyze_disruptive_potential

        items = [_make_financial_item(revenue=0) for _ in range(4)]
        metrics = []
        result = analyze_disruptive_potential(metrics, items)
        assert "score" in result

    def test_zero_revenue_in_innovation_analysis_does_not_crash(self):
        """analyze_innovation_growth with revenue=0 must not crash."""
        from src.agents.cathie_wood import analyze_innovation_growth

        items = [_make_financial_item(revenue=0) for _ in range(4)]
        metrics = [MagicMock()]
        result = analyze_innovation_growth(metrics, items)
        assert "score" in result

    def test_normal_revenue_computes_rd_intensity(self):
        from src.agents.cathie_wood import analyze_disruptive_potential

        items = [_make_financial_item(revenue=1_000_000, rd=200_000) for _ in range(4)]
        metrics = []
        result = analyze_disruptive_potential(metrics, items)
        assert "score" in result
        assert result["score"] >= 0
