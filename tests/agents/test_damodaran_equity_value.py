"""Tests for R57: Damodaran DCF must subtract net debt to convert firm value to equity value."""

from unittest.mock import MagicMock

from src.agents.aswath_damodaran import calculate_intrinsic_value_dcf


def _make_metric(fcf=100.0, revenue=200.0):
    m = MagicMock()
    m.free_cash_flow = fcf
    m.revenue = revenue
    m.price_to_earnings_ratio = 20.0
    return m


def _make_line_item(shares=100.0, total_debt=0.0, cash=0.0):
    li = MagicMock()
    li.outstanding_shares = shares
    li.total_debt = total_debt
    li.cash_and_equivalents = cash
    return li


class TestDamodaranEquityValue:
    def test_net_debt_subtracted_from_firm_value(self):
        """
        R57: enterprise value = pv_sum + tv; equity_value = enterprise_value - net_debt.
        With total_debt=1000 and cash=200, equity_value must equal firm_value - 800.
        """
        metrics = [_make_metric(fcf=100.0, revenue=200.0), _make_metric(fcf=90.0, revenue=180.0)]
        risk = {"cost_of_equity": 0.09}

        li_no_debt = _make_line_item(shares=100.0, total_debt=0.0, cash=0.0)
        li_with_debt = _make_line_item(shares=100.0, total_debt=1000.0, cash=200.0)

        result_no_debt = calculate_intrinsic_value_dcf(metrics, [li_no_debt], risk)
        result_with_debt = calculate_intrinsic_value_dcf(metrics, [li_with_debt], risk)

        assert result_no_debt["intrinsic_value"] is not None
        assert result_with_debt["intrinsic_value"] is not None

        diff = result_no_debt["intrinsic_value"] - result_with_debt["intrinsic_value"]
        assert abs(diff - 800.0) < 1e-6, f"Equity value with net_debt=800 should be firm_value - 800. Difference was {diff:.2f}, expected 800."

    def test_high_debt_produces_lower_equity_value(self):
        """More debt → lower equity value for same underlying business."""
        metrics = [_make_metric(fcf=200.0, revenue=400.0), _make_metric(fcf=180.0, revenue=360.0)]
        risk = {"cost_of_equity": 0.09}

        low_debt = _make_line_item(shares=100.0, total_debt=500.0, cash=100.0)
        high_debt = _make_line_item(shares=100.0, total_debt=2000.0, cash=100.0)

        low = calculate_intrinsic_value_dcf(metrics, [low_debt], risk)
        high = calculate_intrinsic_value_dcf(metrics, [high_debt], risk)

        assert low["intrinsic_value"] > high["intrinsic_value"], "Higher net debt must produce lower equity intrinsic value."

    def test_net_debt_in_assumptions(self):
        """Return dict must expose net_debt in assumptions."""
        metrics = [_make_metric(fcf=100.0, revenue=200.0), _make_metric(fcf=90.0, revenue=180.0)]
        risk = {"cost_of_equity": 0.09}
        li = _make_line_item(shares=100.0, total_debt=600.0, cash=100.0)

        result = calculate_intrinsic_value_dcf(metrics, [li], risk)

        assert "net_debt" in result["assumptions"], "assumptions must contain net_debt key"
        assert abs(result["assumptions"]["net_debt"] - 500.0) < 1e-6
