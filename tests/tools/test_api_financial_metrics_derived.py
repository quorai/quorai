"""Tests for R40: get_financial_metrics computes revenue/earnings/book_value growth rates."""

import pytest


def _make_bundle(revenue, net_income, book_value, period_end):
    from src.tools._yfinance_fundamentals import StatementBundle

    return StatementBundle(
        period_end=period_end,
        income={
            "revenue": revenue,
            "net_income_loss_attributable_common_shareholders": net_income,
            "period_end": period_end,
        },
        balance={
            "total_equity_attributable_to_parent": book_value,
            "period_end": period_end,
        },
        cashflow={},
    )


class TestFinancialMetricsGrowthRates:
    def test_revenue_growth_computed_from_consecutive_bundles(self, monkeypatch):
        """revenue_growth = (curr - prev) / abs(prev) from consecutive periods."""
        from src.tools import api as api_mod

        bundles = [
            _make_bundle(revenue=110, net_income=11, book_value=55, period_end="2024-12-31"),
            _make_bundle(revenue=100, net_income=10, book_value=50, period_end="2023-12-31"),
        ]
        monkeypatch.setattr(api_mod, "_cache", type("C", (), {"get_financial_metrics": lambda *a: None, "set_financial_metrics": lambda *a: None})())
        monkeypatch.setattr(api_mod, "get_backtest_store", lambda: type("S", (), {"slice_financial_metrics": lambda *a: None})())

        from src.tools import _yfinance_fundamentals as yfmod

        monkeypatch.setattr(yfmod, "fetch_statements", lambda *a, **k: bundles)
        # Reload the get_financial_metrics function (it imports fetch_statements lazily inside)
        monkeypatch.setattr("src.tools._yfinance_fundamentals.fetch_statements", lambda *a, **k: bundles)

        from src.tools.api import get_financial_metrics

        # end_date must be >= report_period + 90 days annual lag so the 2024-12-31
        # report is considered "published" and survives the _earliest_available filter.
        metrics = get_financial_metrics("AAPL", "2025-06-30", period="annual", limit=10)

        assert len(metrics) >= 1
        m = metrics[0]
        assert m.revenue_growth == pytest.approx(0.10), f"Expected 10% revenue growth, got {m.revenue_growth}"
        assert m.earnings_growth == pytest.approx(0.10), f"Expected 10% earnings growth, got {m.earnings_growth}"
        assert m.book_value_growth == pytest.approx(0.10), f"Expected 10% book-value growth, got {m.book_value_growth}"

    def test_growth_none_when_only_one_period(self, monkeypatch):
        """With a single bundle, growth rates must remain None (no prior period)."""
        from src.tools import api as api_mod

        bundles = [_make_bundle(revenue=100, net_income=10, book_value=50, period_end="2024-12-31")]
        monkeypatch.setattr(api_mod, "_cache", type("C", (), {"get_financial_metrics": lambda *a: None, "set_financial_metrics": lambda *a: None})())
        monkeypatch.setattr(api_mod, "get_backtest_store", lambda: type("S", (), {"slice_financial_metrics": lambda *a: None})())
        monkeypatch.setattr("src.tools._yfinance_fundamentals.fetch_statements", lambda *a, **k: bundles)

        from src.tools.api import get_financial_metrics

        # end_date past the 90-day annual lag so the 2024-12-31 report is visible.
        metrics = get_financial_metrics("AAPL", "2025-06-30", period="annual", limit=10)

        assert len(metrics) == 1
        assert metrics[0].revenue_growth is None
        assert metrics[0].earnings_growth is None
        assert metrics[0].book_value_growth is None

    def test_negative_prior_revenue_handled(self, monkeypatch):
        """Growth rate computation uses abs(prev) as divisor — no sign flip."""
        from src.tools import api as api_mod

        bundles = [
            _make_bundle(revenue=10, net_income=5, book_value=20, period_end="2024-12-31"),
            _make_bundle(revenue=-100, net_income=-10, book_value=10, period_end="2023-12-31"),
        ]
        monkeypatch.setattr(api_mod, "_cache", type("C", (), {"get_financial_metrics": lambda *a: None, "set_financial_metrics": lambda *a: None})())
        monkeypatch.setattr(api_mod, "get_backtest_store", lambda: type("S", (), {"slice_financial_metrics": lambda *a: None})())
        monkeypatch.setattr("src.tools._yfinance_fundamentals.fetch_statements", lambda *a, **k: bundles)

        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics("AAPL", "2025-06-30", period="annual", limit=10)

        assert len(metrics) >= 1
        # revenue_growth = (10 - (-100)) / abs(-100) = 110 / 100 = 1.10
        assert metrics[0].revenue_growth == pytest.approx(1.10)
