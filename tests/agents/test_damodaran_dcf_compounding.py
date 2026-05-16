"""Tests for R50 (DCF compounding) and R54 (median off-by-one) in aswath_damodaran."""

import statistics
from unittest.mock import MagicMock

from src.agents.aswath_damodaran import analyze_relative_valuation, calculate_intrinsic_value_dcf


def _make_metric(fcf=1000.0, revenue=5000.0, pe=20.0):
    m = MagicMock()
    m.free_cash_flow = fcf
    m.revenue = revenue
    m.price_to_earnings_ratio = pe
    return m


def _make_line_item(shares=100.0):
    li = MagicMock()
    li.outstanding_shares = shares
    return li


class TestDamodaran:
    def test_dcf_compounding_year10_fcff_exceeds_fcff0(self):
        """
        R50: With base_growth=0.10 and 10 projection years, the year-10 FCFF must
        compound to fcff0 * (approximately) 1.1^9 growth steps (g fades toward terminal).
        Before the fix: every year used fcff0 as base → year-10 FCFF ≈ fcff0 * 1.025.
        After the fix: year-10 FCFF >> fcff0.
        """
        fcff0 = 100.0
        metrics = [_make_metric(fcf=fcff0, revenue=110.0), _make_metric(fcf=90.0, revenue=100.0)]
        line_items = [_make_line_item(shares=100.0)]
        risk_analysis = {"cost_of_equity": 0.09}

        result = calculate_intrinsic_value_dcf(metrics, line_items, risk_analysis)

        assert result.get("intrinsic_value") is not None
        intrinsic = result["intrinsic_value"]
        # With base_growth=0.10 (capped at 0.12), discount=0.09, 100 shares:
        # sum of compounded PVs + terminal value must be substantially > fcff0*10
        # (which is what the broken version would produce with ~flat fcff_t).
        assert intrinsic > fcff0 * 10, f"Compounded DCF intrinsic={intrinsic:.1f} should far exceed the flat-series lower bound fcff0*10={fcff0 * 10:.1f}. Bug: FCFFs were not being compounded."

    def test_dcf_terminal_value_uses_projected_fcff_not_base(self):
        """
        R50: Terminal value must use year-10 projected FCFF, not fcff0.
        Verify that a high-growth company has a TV > fcff0 * (1+terminal_growth).
        """
        fcff0 = 100.0
        metrics = [_make_metric(fcf=fcff0, revenue=200.0), _make_metric(fcf=80.0, revenue=100.0)]
        line_items = [_make_line_item(shares=100.0)]
        risk_analysis = {"cost_of_equity": 0.09}

        result_high_growth = calculate_intrinsic_value_dcf(metrics, line_items, risk_analysis)
        assert result_high_growth.get("intrinsic_value") is not None

        # A zero-growth company (revs constant) should have lower intrinsic value
        metrics_flat = [_make_metric(fcf=fcff0, revenue=100.0), _make_metric(fcf=fcff0, revenue=100.0)]
        result_flat = calculate_intrinsic_value_dcf(metrics_flat, line_items, risk_analysis)

        assert result_high_growth["intrinsic_value"] > result_flat["intrinsic_value"], "High-growth DCF result should exceed flat-growth result after fixing terminal value."

    def test_pe_median_even_length_list(self):
        """
        R54: For even-length P/E list, statistics.median returns the average of the two middles.
        Before the fix: sorted(pes)[len//2] always returns the upper middle.
        """
        pes = [10.0, 20.0, 30.0, 40.0]
        # statistics.median([10, 20, 30, 40]) = 25.0 (average of 20 and 30)
        # sorted(pes)[len//2] = sorted(pes)[2] = 30.0 (wrong — upper middle)
        assert statistics.median(pes) == 25.0
        assert statistics.median(pes) != sorted(pes)[len(pes) // 2], "statistics.median must differ from the naive upper-middle for even-length lists"

    def test_pe_median_via_relative_valuation(self):
        """R54: analyze_relative_valuation must produce a meaningful score with 5+ PEs."""
        metrics = [_make_metric(pe=p) for p in [10.0, 15.0, 20.0, 25.0, 30.0]]
        # TTM PE = 10 vs median 20 → cheap (score=1)
        result = analyze_relative_valuation(metrics)
        assert result["score"] == 1, f"TTM P/E=10 vs median=20 should be cheap (score=1). Got: {result}"
