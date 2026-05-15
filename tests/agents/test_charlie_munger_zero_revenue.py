"""Tests for R33 (regression): Charlie Munger cash_to_revenue division is guarded against zero revenue."""

from unittest.mock import MagicMock


def _make_financial_item(**kwargs):
    item = MagicMock()
    item.revenue = kwargs.get("revenue", 1_000_000.0)
    item.net_income = kwargs.get("net_income", 100_000.0)
    item.free_cash_flow = kwargs.get("free_cash_flow", 150_000.0)
    item.cash_and_equivalents = kwargs.get("cash", 500_000.0)
    item.total_debt = kwargs.get("total_debt", 200_000.0)
    item.shareholders_equity = kwargs.get("equity", 800_000.0)
    item.outstanding_shares = kwargs.get("shares", 1_000_000.0)
    return item


class TestCharlieMungerZeroRevenue:
    def test_zero_revenue_does_not_crash(self):
        """
        When revenue_values[0] == 0, the cash_to_revenue computation at charlie_munger.py
        must not raise ZeroDivisionError. The if-guard `revenue_values[0] and revenue_values[0] > 0`
        catches this — this test prevents accidental removal of that guard.
        """
        from src.agents.charlie_munger import analyze_management_quality

        items = [_make_financial_item(revenue=0) for _ in range(4)]
        # Should not raise
        result = analyze_management_quality(items, insider_trades=[])
        assert "score" in result

    def test_normal_revenue_computes_normally(self):
        from src.agents.charlie_munger import analyze_management_quality

        items = [_make_financial_item(revenue=1_000_000, cash=200_000) for _ in range(4)]
        result = analyze_management_quality(items, insider_trades=[])
        assert "score" in result
        assert result["score"] >= 0
