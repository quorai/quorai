"""Tests for the SEC EDGAR XBRL parser in _sec_fundamentals.py."""

import pytest

from src.tools._sec_fundamentals import parse_company_facts


def _minimal_facts(concept: str, value: float, fp: str = "Q1", fy: int = 2023, period_start: str = "2022-10-01", period_end: str = "2022-12-31", filed: str = "2023-02-02", form: str = "10-Q") -> dict:
    """Build a minimal EDGAR Company Facts JSON for one concept."""
    entry: dict = {
        "val": value,
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
        "end": period_end,
    }
    if period_start:
        entry["start"] = period_start
    return {
        "facts": {
            "us-gaap": {
                concept: {
                    "label": concept,
                    "description": "",
                    "units": {"USD": [entry]},
                }
            }
        }
    }


class TestConceptMapping:
    def test_revenues_maps_to_income_revenue(self):
        facts = _minimal_facts("Revenues", 1_000_000.0)
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        revenue_rows = [r for r in rows if r.fact_key == "revenue" and r.section == "income"]
        assert len(revenue_rows) >= 1
        assert revenue_rows[0].value == 1_000_000.0

    def test_asc606_successor_maps_to_revenue(self):
        """RevenueFromContractWithCustomerExcludingAssessedTax → ('income', 'revenue')."""
        facts = _minimal_facts("RevenueFromContractWithCustomerExcludingAssessedTax", 2_000_000.0)
        rows = parse_company_facts(facts, "NVDA", "0001045810")
        revenue_rows = [r for r in rows if r.fact_key == "revenue"]
        assert len(revenue_rows) >= 1
        assert revenue_rows[0].value == 2_000_000.0

    def test_assets_maps_to_balance_section(self):
        entry = {
            "val": 500_000.0,
            "fy": 2023,
            "fp": "Q1",
            "form": "10-Q",
            "filed": "2023-02-02",
            "end": "2022-12-31",
            # No "start" — instantaneous balance sheet fact
        }
        facts = {"facts": {"us-gaap": {"Assets": {"label": "", "description": "", "units": {"USD": [entry]}}}}}
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        balance_rows = [r for r in rows if r.fact_key == "total_assets" and r.section == "balance"]
        assert len(balance_rows) >= 1

    def test_dei_shares_outstanding_maps_to_dei_section(self):
        entry = {
            "val": 15_943_425_000,
            "fy": 2023,
            "fp": "Q1",
            "form": "10-Q",
            "filed": "2023-02-02",
            "end": "2022-12-31",
        }
        facts = {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "label": "",
                        "description": "",
                        "units": {"shares": [entry]},
                    }
                }
            }
        }
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        dei_rows = [r for r in rows if r.section == "dei" and r.fact_key == "shares_outstanding"]
        assert len(dei_rows) >= 1
        assert dei_rows[0].value == 15_943_425_000.0

    def test_unknown_concept_ignored(self):
        facts = _minimal_facts("SomeUnknownConceptXYZ", 42.0)
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        assert rows == []

    def test_invalid_form_excluded(self):
        """Facts from 8-K / DEF 14A should be ignored."""
        facts = _minimal_facts("Revenues", 9_999.0, form="8-K")
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        assert rows == []


class TestDurationFilter:
    def test_twelve_month_window_stored_as_annual(self):
        facts = _minimal_facts("Revenues", 100.0, fp="FY", fy=2022, period_start="2021-10-01", period_end="2022-09-30", form="10-K")
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        annual = [r for r in rows if r.period == "annual" and r.fact_key == "revenue"]
        assert len(annual) == 1

    def test_three_month_window_stored_as_quarterly(self):
        facts = _minimal_facts("Revenues", 50.0, fp="Q1", fy=2023, period_start="2022-10-01", period_end="2022-12-31")
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        quarterly = [r for r in rows if r.period == "quarterly" and r.fact_key == "revenue"]
        assert len(quarterly) == 1

    def test_nine_month_cumulative_excluded(self):
        """A 9-month (cumulative YTD) window should not match quarterly or annual."""
        facts = _minimal_facts("Revenues", 300.0, fp="Q3", fy=2022, period_start="2021-10-01", period_end="2022-06-30")
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        assert rows == []


class TestQ4Derivation:
    def test_q4_derived_from_fy_minus_q123(self):
        """Q4 = FY - Q1 - Q2 - Q3 for flow items."""
        entries_q = [
            {"val": 10.0, "fy": 2022, "fp": "Q1", "form": "10-Q", "filed": "2022-02-01", "start": "2021-10-01", "end": "2021-12-31"},
            {"val": 11.0, "fy": 2022, "fp": "Q2", "form": "10-Q", "filed": "2022-05-01", "start": "2022-01-01", "end": "2022-03-31"},
            {"val": 12.0, "fy": 2022, "fp": "Q3", "form": "10-Q", "filed": "2022-08-01", "start": "2022-04-01", "end": "2022-06-30"},
        ]
        entry_fy = {"val": 50.0, "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2022-11-01", "start": "2021-10-01", "end": "2022-09-30"}
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "",
                        "description": "",
                        "units": {"USD": entries_q + [entry_fy]},
                    }
                }
            }
        }
        rows = parse_company_facts(facts, "AAPL", "0000320193")
        q4_rows = [r for r in rows if r.period == "quarterly" and r.fact_key == "revenue" and r.period_end == "2022-09-30"]
        assert len(q4_rows) == 1
        assert q4_rows[0].value == pytest.approx(50.0 - 10.0 - 11.0 - 12.0)
