import datetime
import logging
import os
import random
import threading
import time

import pandas as pd
import requests

from src.config import get_settings
from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)


class RateLimitExhaustedError(Exception):
    """Raised when all retry attempts on a 429 response are exhausted."""

    def __init__(self, endpoint: str, attempts: int) -> None:
        super().__init__(f"Rate limit exhausted for {endpoint!r} after {attempts} attempts")
        self.endpoint = endpoint
        self.attempts = attempts


# Global cache instance
_cache = get_cache()
_market_cap_none_cache: set[str] = set()  # tracks ticker_date pairs with no market-cap data

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Maps requested line item names to (section, internal_field) tuples.
# section: "income" | "balance" | "cashflow" | "computed" | "metric"
_LINE_ITEM_MAP: dict[str, tuple[str, str]] = {
    # Income statement
    "revenue": ("income", "revenue"),
    "gross_profit": ("income", "gross_profit"),
    "operating_income": ("income", "operating_income"),
    "net_income": ("income", "net_income_loss_attributable_common_shareholders"),
    "earnings_per_share": ("income", "basic_earnings_per_share"),
    "ebit": ("income", "operating_income"),
    "ebitda": ("income", "ebitda"),
    "research_and_development": ("income", "research_development"),
    "selling_general_and_administrative_expenses": ("income", "selling_general_administrative"),
    "depreciation_and_amortization": ("income", "depreciation_depletion_amortization"),
    "outstanding_shares": ("income", "basic_shares_outstanding"),
    "interest_expense": ("income", "interest_expense"),
    "cost_of_revenue": ("income", "cost_of_revenue"),
    # Balance sheet
    "total_assets": ("balance", "total_assets"),
    "total_liabilities": ("balance", "total_liabilities"),
    "current_assets": ("balance", "total_current_assets"),
    "current_liabilities": ("balance", "total_current_liabilities"),
    "cash_and_equivalents": ("balance", "cash_and_equivalents"),
    "shareholders_equity": ("balance", "total_equity_attributable_to_parent"),
    "total_equity": ("balance", "total_equity"),
    "long_term_debt": ("balance", "long_term_debt_and_capital_lease_obligations"),
    "short_term_debt": ("balance", "debt_current"),
    "goodwill": ("balance", "goodwill"),
    "intangible_assets": ("balance", "intangible_assets_net"),
    "retained_earnings": ("balance", "retained_earnings_deficit"),
    "receivables": ("balance", "receivables"),
    "inventories": ("balance", "inventories"),
    # Cash flow statement
    "operating_cash_flow": ("cashflow", "net_cash_from_operating_activities"),
    "capital_expenditure": ("cashflow", "purchase_of_property_plant_and_equipment"),
    "dividends_and_other_cash_distributions": ("cashflow", "dividends"),
    "issuance_or_purchase_of_equity_shares": ("cashflow", "other_financing_activities"),
    # Computed fields
    "free_cash_flow": ("cashflow", "_computed_free_cash_flow"),
    "total_debt": ("balance", "_computed_total_debt"),
    "working_capital": ("balance", "_computed_working_capital"),
    "book_value_per_share": ("balance", "_computed_book_value_per_share"),
    "goodwill_and_intangible_assets": ("balance", "_computed_goodwill_and_intangibles"),
    # Metric-derived fields (populated from FinancialMetrics after _build_financial_metrics)
    "return_on_invested_capital": ("metric", "return_on_invested_capital"),
    "gross_margin": ("metric", "gross_margin"),
    "operating_margin": ("metric", "operating_margin"),
    "net_margin": ("metric", "net_margin"),
    "debt_to_equity": ("metric", "debt_to_equity"),
}


def _get_api_key(api_key: str | None) -> str:
    """Return the Finnhub API key from argument or environment."""
    return api_key or get_settings().FINNHUB_API_KEY


# Deduplicates concurrent identical requests so only one HTTP call goes out per
# unique (url, params) pair. Remaining threads wait and share the result.
_inflight_lock = threading.Lock()
_inflight: dict[tuple, tuple[threading.Event, list]] = {}
_INFLIGHT_WAIT_TIMEOUT = float(os.environ.get("QUORAI_INFLIGHT_WAIT_TIMEOUT", "60"))


def _execute_request(url: str, params: dict | None = None, max_retries: int = 3) -> requests.Response:
    """HTTP GET with exponential-backoff retry on 429, 5xx, and network errors."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=(5, 30))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt < max_retries:
                delay = 2**attempt + random.uniform(0, 1)
                logger.warning("Request to %r failed (attempt %d/%d): %s. Retrying in %.1fs", url, attempt + 1, max_retries + 1, exc, delay)
                time.sleep(delay)
                continue
            raise

        if response.status_code == 429:
            if attempt < max_retries:
                delay = 2**attempt + random.uniform(0, 1)
                logger.warning("Rate limited (429) on %r (attempt %d/%d). Retrying in %.1fs", url, attempt + 1, max_retries + 1, delay)
                time.sleep(delay)
                continue
            raise RateLimitExhaustedError(url, max_retries + 1)

        if 500 <= response.status_code < 600:
            if attempt < max_retries:
                delay = 2**attempt + random.uniform(0, 1)
                logger.warning("Server error %d on %r (attempt %d/%d). Retrying in %.1fs", response.status_code, url, attempt + 1, max_retries + 1, delay)
                time.sleep(delay)
                continue

        return response

    raise RuntimeError(f"Unexpected end of retry loop for {url!r}")


def _make_api_request(url: str, params: dict | None = None, max_retries: int = 3) -> requests.Response:
    """Make a GET request, coalescing concurrent identical requests into one HTTP call."""
    req_key = (url, frozenset((params or {}).items()))

    with _inflight_lock:
        if req_key in _inflight:
            event, result_holder = _inflight[req_key]
            owner = False
        else:
            event = threading.Event()
            result_holder = []
            _inflight[req_key] = (event, result_holder)
            owner = True

    if not owner:
        if not event.wait(timeout=_INFLIGHT_WAIT_TIMEOUT):
            logger.warning("Inflight wait timed out for %r; making independent request", url)
            return _execute_request(url, params, max_retries)
        outcome = result_holder[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    try:
        response = _execute_request(url, params, max_retries)
        result_holder.append(response)
        return response
    except BaseException as exc:
        result_holder.append(exc)
        raise
    finally:
        with _inflight_lock:
            _inflight.pop(req_key, None)
            event.set()


def _date_to_unix(date_str: str) -> int:
    """Convert YYYY-MM-DD string to UTC Unix timestamp (start of day)."""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch daily OHLCV price data from cache or yfinance."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    try:
        import yfinance as yf

        from src.tools._yfinance_fundamentals import _yf_semaphore

        # yfinance end_date is exclusive, so add one day
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
        with _yf_semaphore:
            df = yf.download(ticker, start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty:
            logger.warning("No price data for %s from yfinance", ticker)
            return []

        prices: list[Price] = []
        for ts, row in df.iterrows():
            prices.append(
                Price(
                    open=float(row["Open"].iloc[0] if hasattr(row["Open"], "iloc") else row["Open"]),
                    close=float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"]),
                    high=float(row["High"].iloc[0] if hasattr(row["High"], "iloc") else row["High"]),
                    low=float(row["Low"].iloc[0] if hasattr(row["Low"], "iloc") else row["Low"]),
                    volume=int(row["Volume"].iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"]),
                    time=ts.strftime("%Y-%m-%dT%H:%M:%S"),
                )
            )

        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices
    except Exception as e:
        logger.warning("Failed to fetch prices for %s from yfinance: %s", ticker, e)
        return []


def _build_financial_metrics(ticker: str, income: dict, balance: dict, timeframe: str) -> FinancialMetrics:
    """Build a FinancialMetrics object from income statement and balance sheet data."""
    revenue = income.get("revenue") or 0
    gross_profit = income.get("gross_profit")
    operating_income = income.get("operating_income")
    net_income = income.get("net_income_loss_attributable_common_shareholders")
    eps = income.get("basic_earnings_per_share")

    total_assets = balance.get("total_assets")
    total_equity = balance.get("total_equity_attributable_to_parent")
    total_current_assets = balance.get("total_current_assets")
    total_current_liabilities = balance.get("total_current_liabilities")

    # Prefer the yfinance-reported "Total Debt" when available; fall back to summing components
    total_debt_direct = balance.get("total_debt")
    long_term_debt = balance.get("long_term_debt_and_capital_lease_obligations") or 0
    short_term_debt = balance.get("debt_current") or 0
    total_debt = total_debt_direct if total_debt_direct is not None else (long_term_debt + short_term_debt)

    gross_margin = (gross_profit / revenue) if (gross_profit is not None and revenue) else None
    operating_margin = (operating_income / revenue) if (operating_income is not None and revenue) else None
    net_margin = (net_income / revenue) if (net_income is not None and revenue) else None
    return_on_equity = (net_income / total_equity) if (net_income is not None and total_equity) else None
    return_on_assets = (net_income / total_assets) if (net_income is not None and total_assets) else None
    debt_to_equity = (total_debt / total_equity) if (total_equity and total_equity != 0) else None
    debt_to_assets = (total_debt / total_assets) if (total_assets and total_assets != 0) else None
    current_ratio = (total_current_assets / total_current_liabilities) if (total_current_assets is not None and total_current_liabilities) else None

    # ROIC = net_income / (total_debt + total_equity)
    invested_capital = (total_debt + total_equity) if (total_equity is not None) else None
    roic = (net_income / invested_capital) if (net_income is not None and invested_capital) else None

    return FinancialMetrics(
        ticker=ticker,
        report_period=income.get("period_end", ""),
        period=timeframe,
        currency="USD",
        market_cap=None,
        enterprise_value=None,
        price_to_earnings_ratio=None,
        price_to_book_ratio=None,
        price_to_sales_ratio=None,
        enterprise_value_to_ebitda_ratio=None,
        enterprise_value_to_revenue_ratio=None,
        free_cash_flow_yield=None,
        peg_ratio=None,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        return_on_equity=return_on_equity,
        return_on_assets=return_on_assets,
        return_on_invested_capital=roic,
        asset_turnover=(revenue / total_assets) if (revenue and total_assets) else None,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=current_ratio,
        quick_ratio=None,
        cash_ratio=None,
        operating_cash_flow_ratio=None,
        debt_to_equity=debt_to_equity,
        debt_to_assets=debt_to_assets,
        interest_coverage=None,
        revenue_growth=None,
        earnings_growth=None,
        book_value_growth=None,
        earnings_per_share_growth=None,
        free_cash_flow_growth=None,
        operating_income_growth=None,
        ebitda_growth=None,
        payout_ratio=None,
        earnings_per_share=eps,
        book_value_per_share=(total_equity / income.get("basic_shares_outstanding")) if (total_equity and income.get("basic_shares_outstanding")) else None,
        free_cash_flow_per_share=None,
    )


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics via yfinance."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    from src.tools._yfinance_fundamentals import fetch_statements

    bundles = fetch_statements(ticker, period, end_date, limit)
    if not bundles:
        return []

    freq = "quarterly" if period == "quarterly" else "annual"
    metrics: list[FinancialMetrics] = []
    for bundle in bundles:
        m = _build_financial_metrics(ticker, bundle.income, bundle.balance, freq)
        metrics.append(m)

    # Compute EPS growth across consecutive periods
    for i in range(len(metrics) - 1):
        curr = metrics[i]
        prev = metrics[i + 1]
        if curr.earnings_per_share is not None and prev.earnings_per_share:
            curr.earnings_per_share_growth = (curr.earnings_per_share - prev.earnings_per_share) / abs(prev.earnings_per_share)

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
    return metrics


def _resolve_computed_field(field: str, income: dict, balance: dict, cashflow: dict) -> float | None:
    """Compute derived line item values."""
    if field == "_computed_free_cash_flow":
        # Use yfinance's pre-computed Free Cash Flow when available
        cf_direct = cashflow.get("free_cash_flow")
        if cf_direct is not None:
            return cf_direct
        op_cf = cashflow.get("net_cash_from_operating_activities")
        capex = cashflow.get("purchase_of_property_plant_and_equipment")
        if op_cf is not None and capex is not None:
            return op_cf - abs(capex)
        return None
    if field == "_computed_total_debt":
        td = balance.get("total_debt")
        if td is not None:
            return td
        lt = balance.get("long_term_debt_and_capital_lease_obligations") or 0
        st = balance.get("debt_current") or 0
        return lt + st
    if field == "_computed_working_capital":
        ca = balance.get("total_current_assets")
        cl = balance.get("total_current_liabilities")
        if ca is not None and cl is not None:
            return ca - cl
        return None
    if field == "_computed_book_value_per_share":
        equity = balance.get("total_equity_attributable_to_parent")
        shares = income.get("basic_shares_outstanding")
        if equity is not None and shares:
            return equity / shares
        return None
    if field == "_computed_goodwill_and_intangibles":
        goodwill = balance.get("goodwill") or 0
        intangibles = balance.get("intangible_assets_net") or 0
        return goodwill + intangibles if (goodwill or intangibles) else None
    return None


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch financial line items via yfinance."""
    fields_key = ",".join(sorted(line_items))
    cache_key = f"{ticker}_{period}_{end_date}_{limit}_{fields_key}"
    if cached_data := _cache.get_line_items(cache_key):
        return [LineItem(**item) for item in cached_data]

    from src.tools._yfinance_fundamentals import fetch_statements

    bundles = fetch_statements(ticker, period, end_date, limit)
    if not bundles:
        return []

    freq = "quarterly" if period == "quarterly" else "annual"
    needs_metrics = any(_LINE_ITEM_MAP.get(li, (None,))[0] == "metric" for li in line_items)
    output: list[LineItem] = []
    for bundle in bundles:
        income = bundle.income
        balance = bundle.balance
        cashflow = bundle.cashflow
        period_end = bundle.period_end
        section_map = {"income": income, "balance": balance, "cashflow": cashflow}

        metrics_obj = _build_financial_metrics(ticker, income, balance, freq) if needs_metrics else None

        item_data: dict = {
            "ticker": ticker,
            "report_period": period_end,
            "period": freq,
            "currency": "USD",
        }

        for li in line_items:
            mapping = _LINE_ITEM_MAP.get(li)
            if mapping is None:
                item_data[li] = None
                continue

            section, field = mapping
            if field.startswith("_computed_"):
                item_data[li] = _resolve_computed_field(field, income, balance, cashflow)
            elif section == "metric":
                item_data[li] = getattr(metrics_obj, field, None)
            else:
                item_data[li] = section_map.get(section, {}).get(field)

        output.append(LineItem(**item_data))

    _cache.set_line_items(cache_key, [item.model_dump() for item in output])
    return output


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or Finnhub /stock/insider-transactions."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**t) for t in cached_data]

    key = _get_api_key(api_key)
    url = f"{FINNHUB_BASE_URL}/stock/insider-transactions"
    params: dict = {
        "symbol": ticker,
        "token": key,
    }
    if start_date:
        params["from"] = start_date
    if end_date:
        params["to"] = end_date

    try:
        response = _make_api_request(url, params=params)
    except RateLimitExhaustedError as exc:
        logger.warning("Skipping insider trades for %s: %s", ticker, exc)
        return []
    if response.status_code != 200:
        logger.warning("Failed to fetch insider trades for %s: %s", ticker, response.status_code)
        return []

    data = response.json()
    transactions = data.get("data", [])

    trades: list[InsiderTrade] = []
    for t in transactions:
        transaction_date = t.get("transactionDate", "")
        if not transaction_date:
            continue

        # Use the SEC filing date as the public-availability cutoff.
        # Form 4 filings must be filed within 2 business days of the transaction.
        # If filingDate is not returned by Finnhub, approximate with transaction_date + 2 days.
        filing_date = t.get("filingDate") or t.get("filedDate")
        if not filing_date:
            try:
                td_dt = datetime.datetime.strptime(transaction_date, "%Y-%m-%d")
                filing_date = (td_dt + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
            except ValueError:
                filing_date = transaction_date

        # Filter out trades whose filing was not yet public as of end_date
        if end_date and filing_date > end_date:
            continue

        change = t.get("change")
        shares = change if change is not None else t.get("share")
        value = t.get("value")

        trades.append(
            InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=t.get("name"),
                title=None,
                is_board_director=None,
                transaction_date=transaction_date,
                transaction_shares=float(shares) if shares is not None else None,
                transaction_price_per_share=(value / shares) if (value and shares) else None,
                transaction_value=float(value) if value is not None else None,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=t.get("share"),
                security_title=None,
                filing_date=filing_date,
            )
        )

    trades = trades[:limit]
    _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
    return trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or Finnhub /company-news."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    key = _get_api_key(api_key)
    url = f"{FINNHUB_BASE_URL}/company-news"

    # Finnhub requires both from and to dates
    effective_start = start_date or (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    params: dict = {
        "symbol": ticker,
        "from": effective_start,
        "to": end_date,
        "token": key,
    }

    try:
        response = _make_api_request(url, params=params)
    except RateLimitExhaustedError as exc:
        logger.warning("Skipping news for %s: %s", ticker, exc)
        return []
    if response.status_code != 200:
        logger.warning("Failed to fetch news for %s: %s", ticker, response.status_code)
        return []

    results = response.json()
    if not isinstance(results, list):
        return []

    all_news: list[CompanyNews] = []
    for r in results:
        ts = r.get("datetime")
        if ts:
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            date_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            date_str = ""

        all_news.append(
            CompanyNews(
                ticker=ticker,
                title=r.get("headline", ""),
                author=None,
                source=r.get("source", ""),
                date=date_str,
                url=r.get("url", ""),
                summary=r.get("summary") or None,
                sentiment=None,  # Finnhub /company-news does not include sentiment
            )
        )

    all_news = all_news[:limit]

    if not all_news:
        return []

    _cache.set_company_news(cache_key, [n.model_dump() for n in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Return market cap for ticker on or near end_date, using yfinance."""
    cache_key = f"{ticker}_{end_date}"
    cached = _cache.get_market_cap(cache_key)
    if cached is not None:
        return cached
    if cache_key in _market_cap_none_cache:
        return None

    from src.tools._yfinance_fundamentals import fetch_market_cap as _yf_market_cap

    result = _yf_market_cap(ticker, end_date)
    if result is not None:
        _cache.set_market_cap(cache_key, result)
    else:
        logger.warning("No market cap data for %s on %s", ticker, end_date)
        _market_cap_none_cache.add(cache_key)
    return result


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
