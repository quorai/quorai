"""Tests for R43: fundamentals health_score treats zero as a valid numeric value, not missing."""

from unittest.mock import MagicMock, patch


def _metrics(current_ratio=None, debt_to_equity=None, fcf_per_share=None, eps=None, revenue_growth=None, earnings_growth=None, book_value_growth=None, pe_ratio=None, pb_ratio=None, ps_ratio=None):
    m = MagicMock()
    m.return_on_equity = 0.15
    m.net_margin = 0.10
    m.operating_margin = 0.12
    m.revenue_growth = revenue_growth
    m.earnings_growth = earnings_growth
    m.book_value_growth = book_value_growth
    m.current_ratio = current_ratio
    m.debt_to_equity = debt_to_equity
    m.free_cash_flow_per_share = fcf_per_share
    m.earnings_per_share = eps
    m.price_to_earnings_ratio = pe_ratio
    m.price_to_book_ratio = pb_ratio
    m.price_to_sales_ratio = ps_ratio
    return m


def _run_fundamentals(metrics_obj):
    from src.agents.fundamentals import fundamentals_analyst_agent

    state = {
        "messages": [],
        "data": {
            "tickers": ["AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }

    with patch("src.agents.fundamentals.get_financial_metrics", return_value=[metrics_obj]):
        result = fundamentals_analyst_agent(state)

    return result["data"]["analyst_signals"]["fundamentals_analyst_agent"]["AAPL"]


class TestFundamentalsZeroIsValid:
    def test_zero_debt_to_equity_scores_bullish_health(self):
        """
        debt_to_equity=0.0 means no debt at all — this is strongly bullish for financial health.
        Before the fix: `if 0.0 and 0.0 < 0.5` evaluated False (truthy check dropped the score).
        After the fix: `if 0.0 is not None and 0.0 < 0.5` is True → health_score += 1.
        """
        m = _metrics(current_ratio=2.0, debt_to_equity=0.0)
        result = _run_fundamentals(m)

        health_signal = result["reasoning"]["financial_health_signal"]["signal"]
        assert health_signal in ("bullish", "neutral"), f"Zero debt should not cause 'bearish' health signal. Got: {health_signal}. With current_ratio=2.0 (bullish) and debt_to_equity=0.0 (bullish), health_score should be >= 1."

    def test_health_signal_does_not_treat_zero_as_missing(self):
        """D/E=0 and current_ratio=2.0 → health_score=2 → 'bullish', not 'bearish'."""
        m = _metrics(current_ratio=2.0, debt_to_equity=0.0)
        result = _run_fundamentals(m)

        health_signal = result["reasoning"]["financial_health_signal"]["signal"]
        assert health_signal == "bullish", f"D/E=0 (score +1) + current_ratio=2.0 (score +1) should give bullish. Got: {health_signal}"

    def test_details_show_zero_not_na_for_zero_values(self):
        """Details string must show '0.00' for zero numeric values, not 'N/A'."""
        m = _metrics(current_ratio=0.0, debt_to_equity=0.0)
        result = _run_fundamentals(m)

        details = result["reasoning"]["financial_health_signal"]["details"]
        assert "D/E: N/A" not in details, f"D/E=0.0 should not display as N/A. Details: {details}"
        assert "D/E: 0.00" in details, f"D/E=0.0 should display as '0.00'. Details: {details}"
