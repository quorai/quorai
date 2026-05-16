"""Tests for R53: Cathie Wood payout ratio admits negative FCF as reinvestment signal."""

from unittest.mock import MagicMock

from src.agents.cathie_wood import analyze_innovation_growth


def _item(fcf=None, revenue=None, capex=None, dividend=None, op_margin=None):
    m = MagicMock()
    m.free_cash_flow_per_share = None
    m.free_cash_flow = fcf
    m.revenue = revenue
    m.capital_expenditure = capex
    m.dividends_and_other_cash_distributions = dividend
    m.operating_margin = op_margin
    m.research_and_development = None
    return m


def _metric(pe=None):
    m = MagicMock()
    m.price_to_earnings_ratio = pe
    return m


class TestCathieWoodPayout:
    def test_negative_fcf_with_dividends_not_scored_as_reinvestment(self):
        """
        R53: dividends=5, fcf=-100 → payout_ratio = 5/-100 = -0.05 (negative).
        Before the fix: -0.05 < 0.2 → score +2 ("Strong focus on reinvestment").
        After the fix: fcf_vals[0] must be > 0 — negative FCF falls through to 'insufficient'.
        """
        items = [
            _item(fcf=-100.0, dividend=5.0, revenue=1000.0),
            _item(fcf=-80.0, revenue=900.0),
        ]
        result = analyze_innovation_growth(metrics=[_metric()], financial_line_items=items)

        assert "Strong focus on reinvestment over dividends" not in result.get("details", ""), f"Negative FCF with dividends must NOT produce 'Strong focus on reinvestment'. Details: {result.get('details')}"

    def test_positive_fcf_with_low_dividends_scores_reinvestment(self):
        """Positive FCF with very small dividends should still score reinvestment focus."""
        items = [
            _item(fcf=1000.0, dividend=50.0, revenue=5000.0),
            _item(fcf=900.0, revenue=4500.0),
        ]
        result = analyze_innovation_growth(metrics=[_metric()], financial_line_items=items)

        details = result.get("details", "")
        scored_reinvestment = "Strong focus on reinvestment over dividends" in details or "Moderate focus on reinvestment over dividends" in details
        assert scored_reinvestment, f"Positive FCF with payout_ratio=0.05 should score reinvestment focus. Details: {details}"
