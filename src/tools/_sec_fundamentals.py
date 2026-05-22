"""SEC EDGAR XBRL Company Facts fetcher and parser.

Fetches https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json and maps
US-GAAP / DEI concepts to the internal field names used by StatementBundle.

SEC fair-use policy requires a User-Agent identifying the caller:
  https://www.sec.gov/os/accessing-edgar-data
Set QUORAI_SEC_USER_AGENT="your.email@example.com" or pass --user-agent to the seeder.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
import time

import httpx

logger = logging.getLogger(__name__)

_EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=5.0, pool=5.0)

# Maps XBRL concept name → (section, internal_field_name).
# For concepts where multiple names are possible (GAAP evolution), list all of them —
# the parser takes the first non-None mapped concept for each (fact_key, period_end, filed).
_SEC_CONCEPT_MAP: dict[str, tuple[str, str]] = {
    # ── Income statement ────────────────────────────────────────────────
    "Revenues": ("income", "revenue"),
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("income", "revenue"),
    "SalesRevenueNet": ("income", "revenue"),
    "SalesRevenueGoodsNet": ("income", "revenue"),
    "CostOfRevenue": ("income", "cost_of_revenue"),
    "CostOfGoodsAndServicesSold": ("income", "cost_of_revenue"),
    "GrossProfit": ("income", "gross_profit"),
    "OperatingIncomeLoss": ("income", "operating_income"),
    "NetIncomeLoss": ("income", "net_income_loss_attributable_common_shareholders"),
    "NetIncomeLossAvailableToCommonStockholdersBasic": (
        "income",
        "net_income_loss_attributable_common_shareholders",
    ),
    "EarningsPerShareBasic": ("income", "basic_earnings_per_share"),
    "WeightedAverageNumberOfSharesOutstandingBasic": ("income", "basic_shares_outstanding"),
    "ResearchAndDevelopmentExpense": ("income", "research_development"),
    "SellingGeneralAndAdministrativeExpense": ("income", "selling_general_administrative"),
    "InterestExpense": ("income", "interest_expense"),
    "InterestExpenseDebt": ("income", "interest_expense"),
    "DepreciationDepletionAndAmortization": ("income", "depreciation_depletion_amortization"),
    "DepreciationAndAmortization": ("income", "depreciation_depletion_amortization"),
    # ── Balance sheet ────────────────────────────────────────────────────
    "Assets": ("balance", "total_assets"),
    "Liabilities": ("balance", "total_liabilities"),
    "AssetsCurrent": ("balance", "total_current_assets"),
    "LiabilitiesCurrent": ("balance", "total_current_liabilities"),
    "CashAndCashEquivalentsAtCarryingValue": ("balance", "cash_and_equivalents"),
    "CashCashEquivalentsAndShortTermInvestments": ("balance", "cash_and_equivalents"),
    "StockholdersEquity": ("balance", "total_equity_attributable_to_parent"),
    "CommonStockholdersEquity": ("balance", "total_equity_attributable_to_parent"),
    "LongTermDebt": ("balance", "long_term_debt_and_capital_lease_obligations"),
    "LongTermDebtNoncurrent": ("balance", "long_term_debt_and_capital_lease_obligations"),
    "ShortTermBorrowings": ("balance", "debt_current"),
    "DebtCurrent": ("balance", "debt_current"),
    "LongTermDebtCurrent": ("balance", "debt_current"),
    "DebtLongtermAndShorttermCombinedAmount": ("balance", "total_debt"),
    "Goodwill": ("balance", "goodwill"),
    "IntangibleAssetsNetExcludingGoodwill": ("balance", "intangible_assets_net"),
    "FiniteLivedIntangibleAssetsNet": ("balance", "intangible_assets_net"),
    "RetainedEarningsAccumulatedDeficit": ("balance", "retained_earnings_deficit"),
    "AccountsReceivableNetCurrent": ("balance", "receivables"),
    "ReceivablesNetCurrent": ("balance", "receivables"),
    "InventoryNet": ("balance", "inventories"),
    "InventoryFinishedGoodsAndWorkInProcess": ("balance", "inventories"),
    # ── Cash flow ───────────────────────────────────────────────────────
    "NetCashProvidedByUsedInOperatingActivities": ("cashflow", "net_cash_from_operating_activities"),
    "PaymentsToAcquirePropertyPlantAndEquipment": ("cashflow", "purchase_of_property_plant_and_equipment"),
    "PaymentsOfDividends": ("cashflow", "dividends"),
    "PaymentsOfDividendsCommonStock": ("cashflow", "dividends"),
    "NetCashProvidedByUsedInFinancingActivities": ("cashflow", "other_financing_activities"),
    # ── DEI (shares outstanding for market cap) ─────────────────────────
    "EntityCommonStockSharesOutstanding": ("dei", "shares_outstanding"),
}

# Reverse map: internal fact_key → section (first encountered wins — consistent within map).
_FACT_KEY_TO_SECTION: dict[str, str] = {}
for _concept, (_sec, _fk) in _SEC_CONCEPT_MAP.items():
    _FACT_KEY_TO_SECTION.setdefault(_fk, _sec)

# Balance-sheet and DEI concepts have no `start` date (instantaneous values).
_INSTANTANEOUS_SECTIONS = frozenset(["balance", "dei"])

# Only accept facts from these form types (exclude 8-K, proxy, etc.)
_VALID_FORMS = frozenset(["10-K", "10-K/A", "10-Q", "10-Q/A"])

# Duration windows for flow items (in days)
_QUARTERLY_DAYS = (75, 110)
_ANNUAL_DAYS = (330, 400)


@dataclass
class FactRow:
    """A single parsed EDGAR fact ready for insertion into sec_facts."""

    ticker: str
    section: str
    fact_key: str
    period: str  # 'annual' | 'quarterly'
    period_end: str
    filed: str
    value: float | None
    unit: str = "USD"


def fetch_company_facts(cik: str, user_agent: str, max_retries: int = 3) -> dict | None:
    """Fetch raw Company Facts JSON for a CIK from EDGAR. Returns None on failure."""
    url = _EDGAR_FACTS_URL.format(cik=cik)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code == 404:
                logger.debug("No EDGAR data for CIK %s (404)", cik)
                return None
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("SEC rate-limited (429) on CIK %s; waiting %ds", cik, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning("EDGAR facts CIK %s → HTTP %d", cik, resp.status_code)
                return None
            return resp.json()
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.warning("EDGAR facts fetch failed for CIK %s: %s", cik, exc)
                return None
            time.sleep(1.5 * (attempt + 1))

    return None


def parse_company_facts(facts: dict, ticker: str, cik: str) -> list[FactRow]:
    """Parse a Company Facts JSON response into a list of FactRows.

    Handles:
    - Duration filtering (only accept ~3-month or ~12-month windows for flow items)
    - Concept priority (first non-None mapped concept wins per fact_key/period/period_end/filed)
    - Q4 derivation (FY - Q1 - Q2 - Q3 for flow items)
    - Balance sheet FY entries stored as both annual AND quarterly (Q4 equivalent)
    """
    # Primary accumulator: key = (section, fact_key, period, period_end, filed)
    seen: dict[tuple[str, str, str, str, str], FactRow] = {}

    # For Q4 derivation of flow items: (fact_key, fy) → {fp → (period_end, filed, value)}
    quarterly_flow: dict[tuple[str, int], dict[str, tuple[str, str, float | None]]] = {}
    annual_flow: dict[tuple[str, int], tuple[str, str, float | None]] = {}

    for namespace in ("us-gaap", "dei"):
        ns_data = facts.get("facts", {}).get(namespace, {})
        for concept, concept_data in ns_data.items():
            mapped = _SEC_CONCEPT_MAP.get(concept)
            if mapped is None:
                continue
            section, fact_key = mapped

            for unit_key, unit_entries in concept_data.get("units", {}).items():
                for entry in unit_entries:
                    if entry.get("form", "") not in _VALID_FORMS:
                        continue

                    fp = entry.get("fp") or ""
                    fy_raw = entry.get("fy")
                    fy = int(fy_raw) if fy_raw else 0
                    period_end = (entry.get("end") or "")[:10]
                    period_start = (entry.get("start") or "")[:10]
                    filed = (entry.get("filed") or "")[:10]

                    if not (period_end and filed):
                        continue

                    val_raw = entry.get("val")
                    value = float(val_raw) if val_raw is not None else None

                    if section in _INSTANTANEOUS_SECTIONS:
                        # Balance sheet / DEI — no duration check
                        if fp == "FY":
                            # Year-end balance: store as both annual and quarterly (Q4 equivalent)
                            _add_row(seen, ticker, section, fact_key, "annual", period_end, filed, value, unit_key)
                            _add_row(seen, ticker, section, fact_key, "quarterly", period_end, filed, value, unit_key)
                        elif fp in ("Q1", "Q2", "Q3", "Q4"):
                            _add_row(seen, ticker, section, fact_key, "quarterly", period_end, filed, value, unit_key)
                    else:
                        # Flow statement — require explicit start date and validate duration
                        if not period_start:
                            continue
                        try:
                            start_dt = datetime.date.fromisoformat(period_start)
                            end_dt = datetime.date.fromisoformat(period_end)
                            days = (end_dt - start_dt).days
                        except ValueError:
                            continue

                        if _QUARTERLY_DAYS[0] <= days <= _QUARTERLY_DAYS[1] and fp in ("Q1", "Q2", "Q3", "Q4"):
                            _add_row(seen, ticker, section, fact_key, "quarterly", period_end, filed, value, unit_key)
                            if fy:
                                quarterly_flow.setdefault((fact_key, fy), {})[fp] = (period_end, filed, value)

                        elif _ANNUAL_DAYS[0] <= days <= _ANNUAL_DAYS[1] and fp == "FY":
                            _add_row(seen, ticker, section, fact_key, "annual", period_end, filed, value, unit_key)
                            if fy:
                                annual_flow[(fact_key, fy)] = (period_end, filed, value)

    # Derive Q4 = FY - Q1 - Q2 - Q3 for flow items where Q4 is not directly reported
    for (fact_key, fy), qtrs in quarterly_flow.items():
        if "Q4" in qtrs:
            continue
        ann = annual_flow.get((fact_key, fy))
        if ann is None:
            continue
        q1 = qtrs.get("Q1")
        q2 = qtrs.get("Q2")
        q3 = qtrs.get("Q3")
        if not (q1 and q2 and q3):
            continue
        fy_pe, fy_filed, fy_val = ann
        if fy_val is None or q1[2] is None or q2[2] is None or q3[2] is None:
            continue
        q4_val = fy_val - q1[2] - q2[2] - q3[2]
        section = _FACT_KEY_TO_SECTION.get(fact_key, "income")
        key = (section, fact_key, "quarterly", fy_pe, fy_filed)
        if key not in seen:
            seen[key] = FactRow(
                ticker=ticker,
                section=section,
                fact_key=fact_key,
                period="quarterly",
                period_end=fy_pe,
                filed=fy_filed,
                value=q4_val,
            )

    return list(seen.values())


def _add_row(
    seen: dict,
    ticker: str,
    section: str,
    fact_key: str,
    period: str,
    period_end: str,
    filed: str,
    value: float | None,
    unit: str,
) -> None:
    """Insert into `seen` only if the key hasn't been claimed by an earlier concept."""
    key = (section, fact_key, period, period_end, filed)
    if key not in seen:
        seen[key] = FactRow(
            ticker=ticker,
            section=section,
            fact_key=fact_key,
            period=period,
            period_end=period_end,
            filed=filed,
            value=value,
            unit=unit,
        )
