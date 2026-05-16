"""Tests for R59 (growth capping) and R63 (ROIC positive guard) in tools/api.py."""

import pytest

from src.tools._yfinance_fundamentals import StatementBundle


def _bundle(income: dict, balance: dict, period_end: str = "2024-12-31") -> StatementBundle:
    return StatementBundle(
        period_end=period_end,
        income={**income, "period_end": period_end},
        balance={**balance, "period_end": period_end},
        cashflow={},
    )


def _patch(monkeypatch, bundles):
    from src.tools import api as api_mod

    monkeypatch.setattr(api_mod, "_cache", type("C", (), {"get_financial_metrics": lambda *a: None, "set_financial_metrics": lambda *a: None})())
    monkeypatch.setattr(api_mod, "get_backtest_store", lambda: type("S", (), {"slice_financial_metrics": lambda *a: None})())
    monkeypatch.setattr("src.tools._yfinance_fundamentals.fetch_statements", lambda *a, **k: bundles)


class TestApiGrowthCaps:
    def test_earnings_growth_capped_at_500_percent(self, monkeypatch):
        """
        R59: prev_ni=500K, curr_ni=50M → raw=(50M-0.5M)/0.5M=99 (9900%); must be capped at 5.0.
        """
        bundles = [
            _bundle(
                income={"net_income_loss_attributable_common_shareholders": 50_000_000.0, "revenue": 100_000_000.0},
                balance={"total_equity_attributable_to_parent": 200_000_000.0},
            ),
            _bundle(
                income={"net_income_loss_attributable_common_shareholders": 500_000.0, "revenue": 90_000_000.0},
                balance={"total_equity_attributable_to_parent": 190_000_000.0},
                period_end="2023-12-31",
            ),
        ]
        _patch(monkeypatch, bundles)

        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics("TEST", "2025-06-30", period="annual", limit=10)
        assert metrics
        eg = metrics[0].earnings_growth
        assert eg is not None
        assert eg <= 5.0, f"earnings_growth must be capped at 5.0 (500%), got {eg}"

    def test_revenue_growth_capped_at_500_percent(self, monkeypatch):
        """R59: Same cap applies to revenue_growth."""
        bundles = [
            _bundle(
                income={"revenue": 50_000_000.0, "net_income_loss_attributable_common_shareholders": 5_000_000.0},
                balance={"total_equity_attributable_to_parent": 200_000_000.0},
            ),
            _bundle(
                income={"revenue": 500_000.0, "net_income_loss_attributable_common_shareholders": 500_000.0},
                balance={"total_equity_attributable_to_parent": 190_000_000.0},
                period_end="2023-12-31",
            ),
        ]
        _patch(monkeypatch, bundles)

        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics("TEST", "2025-06-30", period="annual", limit=10)
        assert metrics
        rg = metrics[0].revenue_growth
        assert rg is not None
        assert rg <= 5.0, f"revenue_growth must be capped at 5.0, got {rg}"

    def test_normal_growth_not_clipped(self, monkeypatch):
        """R59: Typical 10% growth must pass through unchanged."""
        bundles = [
            _bundle(
                income={"net_income_loss_attributable_common_shareholders": 11_000_000.0, "revenue": 110_000_000.0},
                balance={"total_equity_attributable_to_parent": 55_000_000.0},
            ),
            _bundle(
                income={"net_income_loss_attributable_common_shareholders": 10_000_000.0, "revenue": 100_000_000.0},
                balance={"total_equity_attributable_to_parent": 50_000_000.0},
                period_end="2023-12-31",
            ),
        ]
        _patch(monkeypatch, bundles)

        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics("TEST", "2025-06-30", period="annual", limit=10)
        assert metrics
        assert metrics[0].earnings_growth == pytest.approx(0.10, abs=1e-6)

    def test_roic_none_when_invested_capital_negative(self, monkeypatch):
        """
        R63: total_equity=-200M, total_debt=100M → invested_capital=-100M < 0 → roic must be None.
        Before fix: negative denominator inverts ROIC sign silently.
        """
        bundles = [
            _bundle(
                income={"net_income_loss_attributable_common_shareholders": 10_000_000.0, "revenue": 100_000_000.0},
                balance={"total_equity_attributable_to_parent": -200_000_000.0, "total_debt": 100_000_000.0},
            ),
        ]
        _patch(monkeypatch, bundles)

        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics("TEST", "2025-06-30", period="annual", limit=10)
        assert metrics
        assert metrics[0].return_on_invested_capital is None, f"ROIC must be None when invested_capital < 0, got {metrics[0].return_on_invested_capital}"

    def test_roic_positive_when_invested_capital_positive(self, monkeypatch):
        """R63: Positive equity and debt → ROIC should be computed normally."""
        bundles = [
            _bundle(
                income={"net_income_loss_attributable_common_shareholders": 10_000_000.0, "revenue": 100_000_000.0},
                balance={"total_equity_attributable_to_parent": 100_000_000.0, "total_debt": 50_000_000.0},
            ),
        ]
        _patch(monkeypatch, bundles)

        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics("TEST", "2025-06-30", period="annual", limit=10)
        assert metrics
        roic = metrics[0].return_on_invested_capital
        assert roic is not None
        assert roic == pytest.approx(10_000_000.0 / 150_000_000.0, abs=1e-6)
