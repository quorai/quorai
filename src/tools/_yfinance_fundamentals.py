"""yfinance-based fundamentals fetcher replacing Finnhub /stock/financials-reported."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Maps internal field names -> ordered list of yfinance row label alternates.
# yfinance row labels drift across library versions; always try alternates.
_YF_INCOME: dict[str, list[str]] = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income_loss_attributable_common_shareholders": [
        "Net Income Common Stockholders",
        "Net Income",
        "Net Income From Continuing Operations",
    ],
    "basic_earnings_per_share": ["Basic EPS", "Diluted EPS"],
    "basic_shares_outstanding": ["Basic Average Shares", "Diluted Average Shares"],
    "research_development": ["Research And Development"],
    "selling_general_administrative": [
        "Selling General And Administration",
        "Selling General And Administrative",
    ],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "depreciation_depletion_amortization": [
        "Reconciled Depreciation",
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
    ],
}

_YF_BALANCE: dict[str, list[str]] = {
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_current_assets": ["Current Assets"],
    "total_current_liabilities": ["Current Liabilities"],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
    "total_equity_attributable_to_parent": ["Stockholders Equity", "Common Stock Equity"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity"],
    "long_term_debt_and_capital_lease_obligations": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    ],
    "debt_current": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "total_debt": ["Total Debt"],
    "goodwill": ["Goodwill"],
    "intangible_assets_net": ["Other Intangible Assets"],
    "retained_earnings_deficit": ["Retained Earnings"],
    "receivables": ["Accounts Receivable", "Receivables"],
    "inventories": ["Inventory"],
}

_YF_CASHFLOW: dict[str, list[str]] = {
    "net_cash_from_operating_activities": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ],
    "purchase_of_property_plant_and_equipment": ["Capital Expenditure"],
    "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "other_financing_activities": ["Financing Cash Flow"],
    "free_cash_flow": ["Free Cash Flow"],
}


@dataclass
class StatementBundle:
    period_end: str  # YYYY-MM-DD
    income: dict[str, float | None] = field(default_factory=dict)
    balance: dict[str, float | None] = field(default_factory=dict)
    cashflow: dict[str, float | None] = field(default_factory=dict)
    shares_outstanding: float | None = None


def _lookup(df: pd.DataFrame, col, labels: list[str]) -> float | None:
    """Return the first non-NaN float from df at (label, col) for the given labels."""
    for label in labels:
        if label in df.index:
            try:
                val = df.at[label, col]
                if pd.notna(val) and val is not None:
                    return float(val)
            except (KeyError, TypeError, ValueError):
                pass
    return None


def _extract_section(df: pd.DataFrame, col, label_map: dict[str, list[str]]) -> dict[str, float | None]:
    return {field: _lookup(df, col, labels) for field, labels in label_map.items()}


def _filter_cols(df: pd.DataFrame, end_date: str) -> list:
    """Return DataFrame columns (Timestamps) that fall on or before end_date, newest first."""
    if df is None or df.empty:
        return []
    ed = pd.Timestamp(end_date)
    cols = [c for c in df.columns if pd.notna(c) and c <= ed]
    return sorted(cols, reverse=True)


def _get_ticker(ticker: str):
    """Return a yfinance Ticker object, or None on error."""
    try:
        return yf.Ticker(ticker)
    except Exception as exc:
        logger.warning("Could not create yfinance Ticker for %s: %s", ticker, exc)
        return None


def fetch_statements(ticker: str, period: str, end_date: str, limit: int) -> list[StatementBundle]:
    """Return up to `limit` StatementBundles for the given ticker and period.

    period: "annual" | "quarterly" | "ttm"

    TTM is computed by summing trailing-4-quarter income/cashflow flow items and
    using the most recent quarter's balance-sheet stock items.
    """
    yf_ticker = _get_ticker(ticker)
    if yf_ticker is None:
        return []

    try:
        if period == "ttm":
            return _fetch_ttm(yf_ticker, ticker, end_date, limit)
        elif period == "quarterly":
            return _fetch_period(yf_ticker, ticker, end_date, limit, quarterly=True)
        else:
            return _fetch_period(yf_ticker, ticker, end_date, limit, quarterly=False)
    except Exception as exc:
        logger.warning("yfinance statement fetch failed for %s (%s): %s", ticker, period, exc)
        return []


def _fetch_period(yf_ticker, ticker: str, end_date: str, limit: int, *, quarterly: bool) -> list[StatementBundle]:
    if quarterly:
        df_income = yf_ticker.quarterly_income_stmt
        df_balance = yf_ticker.quarterly_balance_sheet
        df_cashflow = yf_ticker.quarterly_cashflow
        period_label = "quarterly"
    else:
        df_income = yf_ticker.income_stmt
        df_balance = yf_ticker.balance_sheet
        df_cashflow = yf_ticker.cashflow
        period_label = "annual"

    cols = _filter_cols(df_income, end_date)[:limit]
    if not cols:
        logger.warning("No %s financials found for %s up to %s", period_label, ticker, end_date)
        return []

    bundles: list[StatementBundle] = []
    for col in cols:
        period_end = col.strftime("%Y-%m-%d")
        income = _extract_section(df_income, col, _YF_INCOME)
        income["period_end"] = period_end

        balance: dict[str, float | None] = {}
        b_cols = _filter_cols(df_balance, end_date)
        if b_cols:
            bc = b_cols[0] if col not in df_balance.columns else col
            balance = _extract_section(df_balance, bc, _YF_BALANCE)
        balance["period_end"] = period_end

        cashflow: dict[str, float | None] = {}
        cf_cols = _filter_cols(df_cashflow, end_date)
        if cf_cols:
            cfc = cf_cols[0] if col not in df_cashflow.columns else col
            cashflow = _extract_section(df_cashflow, cfc, _YF_CASHFLOW)
        cashflow["period_end"] = period_end

        bundles.append(
            StatementBundle(
                period_end=period_end,
                income=income,
                balance=balance,
                cashflow=cashflow,
                shares_outstanding=income.get("basic_shares_outstanding"),
            )
        )

    return bundles


def _fetch_ttm(yf_ticker, ticker: str, end_date: str, limit: int) -> list[StatementBundle]:
    df_income = yf_ticker.quarterly_income_stmt
    df_balance = yf_ticker.quarterly_balance_sheet
    df_cashflow = yf_ticker.quarterly_cashflow

    q_cols = _filter_cols(df_income, end_date)
    if not q_cols:
        logger.warning("No quarterly financials for %s up to %s; cannot compute TTM", ticker, end_date)
        return []

    # Sum the trailing 4 quarters for flow items
    ttm_cols = q_cols[:4]
    most_recent = ttm_cols[0]
    period_end = most_recent.strftime("%Y-%m-%d")

    income: dict[str, float | None] = {}
    for field_name, labels in _YF_INCOME.items():
        total = 0.0
        found_any = False
        for col in ttm_cols:
            if col in df_income.columns:
                val = _lookup(df_income, col, labels)
                if val is not None:
                    total += val
                    found_any = True
        income[field_name] = total if found_any else None
    income["period_end"] = period_end

    # Balance: use the most recent quarter's values (stock items, not summed)
    b_cols = _filter_cols(df_balance, end_date)
    balance: dict[str, float | None] = {}
    if b_cols and not (df_balance is None or df_balance.empty):
        balance = _extract_section(df_balance, b_cols[0], _YF_BALANCE)
    balance["period_end"] = period_end

    # Cashflow: sum flow items across trailing 4 quarters
    cf_cols = _filter_cols(df_cashflow, end_date)[:4] if df_cashflow is not None and not df_cashflow.empty else []
    cashflow: dict[str, float | None] = {}
    for field_name, labels in _YF_CASHFLOW.items():
        total = 0.0
        found_any = False
        for col in cf_cols:
            val = _lookup(df_cashflow, col, labels)
            if val is not None:
                total += val
                found_any = True
        cashflow[field_name] = total if found_any else None
    cashflow["period_end"] = period_end

    ttm_bundle = StatementBundle(
        period_end=period_end,
        income=income,
        balance=balance,
        cashflow=cashflow,
        shares_outstanding=income.get("basic_shares_outstanding"),
    )

    # Append recent annual periods so callers can compute growth (up to limit)
    annual_bundles = _fetch_period(yf_ticker, ticker, end_date, limit - 1, quarterly=False)
    return [ttm_bundle] + annual_bundles


def fetch_market_cap(ticker: str, end_date: str) -> float | None:
    """Return market cap for the given ticker on or near end_date.

    For dates within 7 calendar days of today, uses yfinance info["marketCap"]
    (current market cap — acceptable for live/dry-run mode).
    For historical dates, computes shares_outstanding × close_price.
    """
    yf_ticker = _get_ticker(ticker)
    if yf_ticker is None:
        return None

    try:
        today = datetime.date.today()
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        near_today = abs((today - end_dt).days) <= 7

        if near_today:
            # fast_info uses the chart endpoint (no Crumb auth) — prefer it over .info
            try:
                fi = yf_ticker.fast_info
                mcap = fi.get("market_cap") if isinstance(fi, dict) else getattr(fi, "market_cap", None)
                if mcap:
                    return float(mcap)
            except Exception:
                pass
            # Fallback: .info (Crumb-protected, may 401 intermittently)
            try:
                info = yf_ticker.info or {}
            except Exception:
                info = {}
            mcap = info.get("marketCap")
            if mcap:
                return float(mcap)
            shares = info.get("sharesOutstanding")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if shares and price:
                return float(shares) * float(price)
            return None

        # Historical: get shares from get_shares_full, price from history
        start_dt = (end_dt - datetime.timedelta(days=35)).strftime("%Y-%m-%d")
        try:
            shares_series = yf_ticker.get_shares_full(start=start_dt, end=end_date)
            if shares_series is not None and not shares_series.empty:
                # Timezone-strip and take the most recent entry
                if hasattr(shares_series.index, "tz_convert"):
                    shares_series.index = shares_series.index.tz_convert(None)
                shares = float(shares_series.iloc[-1])
            else:
                shares = None
        except Exception:
            shares = None

        if shares is None:
            try:
                fi = yf_ticker.fast_info
                sh = fi.get("shares") if isinstance(fi, dict) else getattr(fi, "shares", None)
                if sh:
                    shares = float(sh)
            except Exception:
                pass
        if shares is None:
            try:
                info = yf_ticker.info or {}
            except Exception:
                info = {}
            sh = info.get("sharesOutstanding")
            if sh:
                shares = float(sh)

        if not shares:
            return None

        # Get close price on end_date (or last available trading day before it)
        hist = yf_ticker.history(start=start_dt, end=end_date, auto_adjust=True)
        if hist.empty:
            return None
        close_price = float(hist["Close"].iloc[-1])
        return shares * close_price

    except Exception as exc:
        logger.warning("yfinance market cap fetch failed for %s on %s: %s", ticker, end_date, exc)
        return None
