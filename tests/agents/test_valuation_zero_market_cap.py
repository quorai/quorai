"""Tests for R32: market_cap=0 guard in Cathie Wood and Bill Ackman valuation functions."""

from unittest.mock import MagicMock


def _make_line_item(**kwargs):
    item = MagicMock()
    item.free_cash_flow = kwargs.get("free_cash_flow", 1_000_000.0)
    return item


class TestCathieWoodValuationZeroMarketCap:
    def test_zero_market_cap_returns_score_zero(self):
        """analyze_cathie_wood_valuation with market_cap=0 must not raise ZeroDivisionError."""
        from src.agents.cathie_wood import analyze_cathie_wood_valuation

        result = analyze_cathie_wood_valuation([_make_line_item()], market_cap=0)
        assert result["score"] == 0

    def test_none_market_cap_returns_score_zero(self):
        from src.agents.cathie_wood import analyze_cathie_wood_valuation

        result = analyze_cathie_wood_valuation([_make_line_item()], market_cap=None)
        assert result["score"] == 0

    def test_negative_market_cap_returns_score_zero(self):
        from src.agents.cathie_wood import analyze_cathie_wood_valuation

        result = analyze_cathie_wood_valuation([_make_line_item()], market_cap=-1000)
        assert result["score"] == 0

    def test_positive_market_cap_computes_normally(self):
        from src.agents.cathie_wood import analyze_cathie_wood_valuation

        result = analyze_cathie_wood_valuation([_make_line_item(free_cash_flow=2_000_000)], market_cap=1_000_000)
        assert "score" in result
        assert "intrinsic_value" in result


class TestBillAckmanValuationZeroMarketCap:
    def test_zero_market_cap_returns_score_zero(self):
        """analyze_valuation (Ackman) with market_cap=0 must not raise ZeroDivisionError."""
        from src.agents.bill_ackman import analyze_valuation

        item = _make_line_item(free_cash_flow=500_000.0)
        result = analyze_valuation([item], market_cap=0)
        assert result["score"] == 0

    def test_none_market_cap_returns_score_zero(self):
        from src.agents.bill_ackman import analyze_valuation

        result = analyze_valuation([_make_line_item()], market_cap=None)
        assert result["score"] == 0

    def test_positive_market_cap_computes_normally(self):
        from src.agents.bill_ackman import analyze_valuation

        result = analyze_valuation([_make_line_item(free_cash_flow=1_000_000)], market_cap=5_000_000)
        assert "score" in result
        assert "intrinsic_value" in result
