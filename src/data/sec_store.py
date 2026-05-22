"""Read-on-demand SQLite store for SEC EDGAR XBRL fundamentals.

Populated by experiments/seed_sec_fundamentals.py. Consulted by
_yfinance_fundamentals.fetch_statements / fetch_market_cap before the
yfinance live-fetch path, giving historically correct point-in-time data.

The store uses a *separate* SQLite file from the existing api_cache.db so it
can grow to GBs without loading into memory at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sqlite3
import threading

logger = logging.getLogger(__name__)

_DB_PATH = Path(".cache/sec_fundamentals.db")

_DDL = """
CREATE TABLE IF NOT EXISTS sec_facts (
    ticker   TEXT NOT NULL,
    section  TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    period   TEXT NOT NULL,
    period_end TEXT NOT NULL,
    filed    TEXT NOT NULL,
    value    REAL,
    unit     TEXT NOT NULL DEFAULT 'USD',
    PRIMARY KEY (ticker, section, fact_key, period, period_end, filed)
);
CREATE INDEX IF NOT EXISTS idx_sec_facts_lookup
    ON sec_facts(ticker, period, period_end DESC, filed DESC);
CREATE TABLE IF NOT EXISTS sec_meta (
    ticker       TEXT PRIMARY KEY,
    cik          TEXT NOT NULL,
    company_name TEXT,
    last_synced  TEXT NOT NULL
);
"""

# The fields that belong to each statement section.
_INCOME_FIELDS = frozenset(
    [
        "revenue",
        "cost_of_revenue",
        "gross_profit",
        "operating_income",
        "net_income_loss_attributable_common_shareholders",
        "basic_earnings_per_share",
        "basic_shares_outstanding",
        "research_development",
        "selling_general_administrative",
        "interest_expense",
        "ebitda",
        "depreciation_depletion_amortization",
    ]
)
_BALANCE_FIELDS = frozenset(
    [
        "total_assets",
        "total_liabilities",
        "total_current_assets",
        "total_current_liabilities",
        "cash_and_equivalents",
        "total_equity_attributable_to_parent",
        "total_equity",
        "long_term_debt_and_capital_lease_obligations",
        "debt_current",
        "total_debt",
        "goodwill",
        "intangible_assets_net",
        "retained_earnings_deficit",
        "receivables",
        "inventories",
    ]
)
_CASHFLOW_FIELDS = frozenset(
    [
        "net_cash_from_operating_activities",
        "purchase_of_property_plant_and_equipment",
        "dividends",
        "other_financing_activities",
        "free_cash_flow",
    ]
)

# Flow items are summed for TTM; balance / EPS items are NOT summed.
_FLOW_SECTIONS = frozenset(["income", "cashflow"])
_STOCK_SECTIONS = frozenset(["balance", "dei"])
# Within income, per-share and share-count items are instantaneous, not summed.
_INCOME_STOCK_KEYS = frozenset(["basic_earnings_per_share", "basic_shares_outstanding"])


@dataclass
class FactRow:
    """One row to be inserted into sec_facts."""

    ticker: str
    section: str
    fact_key: str
    period: str
    period_end: str
    filed: str
    value: float | None
    unit: str = "USD"


@dataclass
class _Bundle:
    """Internal accumulator before converting to StatementBundle."""

    period_end: str
    income: dict[str, float | None] = field(default_factory=dict)
    balance: dict[str, float | None] = field(default_factory=dict)
    cashflow: dict[str, float | None] = field(default_factory=dict)
    shares_outstanding: float | None = None


def _to_statement_bundle(b: _Bundle):
    """Convert an internal _Bundle to the public StatementBundle type.

    Mirrors the yfinance convention of injecting period_end into each section dict
    so that _build_financial_metrics can read income.get("period_end").
    """
    from src.tools._yfinance_fundamentals import StatementBundle

    income = {**b.income, "period_end": b.period_end}
    balance = {**b.balance, "period_end": b.period_end}
    cashflow = {**b.cashflow, "period_end": b.period_end}
    return StatementBundle(
        period_end=b.period_end,
        income=income,
        balance=balance,
        cashflow=cashflow,
        shares_outstanding=income.get("basic_shares_outstanding"),
    )


class SecStore:
    """Process-global singleton wrapping sec_fundamentals.db."""

    def __init__(self, db_path: Path | str = _DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_DDL)
        except Exception as exc:
            logger.warning("Could not initialize sec_fundamentals.db: %s", exc)

    # ------------------------------------------------------------------
    # Write path (called by seeder)
    # ------------------------------------------------------------------

    def upsert_facts(self, ticker: str, cik: str, company_name: str | None, rows: list[FactRow]) -> None:
        """Bulk-insert fact rows for a ticker and update sec_meta."""
        with self._lock:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO sec_facts (ticker, section, fact_key, period, period_end, filed, value, unit) VALUES (?,?,?,?,?,?,?,?)",
                    [(r.ticker, r.section, r.fact_key, r.period, r.period_end, r.filed, r.value, r.unit) for r in rows],
                )
                import datetime

                conn.execute(
                    "INSERT OR REPLACE INTO sec_meta (ticker, cik, company_name, last_synced) VALUES (?,?,?,?)",
                    (ticker, cik, company_name, datetime.date.today().isoformat()),
                )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def is_seeded(self, ticker: str) -> bool:
        """Return True if we have ever seeded this ticker."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT 1 FROM sec_meta WHERE ticker=? LIMIT 1", (ticker,)).fetchone()
                return row is not None
        except Exception:
            return False

    def get_statements(self, ticker: str, period: str, end_date: str, limit: int):
        """Return a list of StatementBundles or None if the ticker is not seeded.

        None  → caller should fall through to yfinance.
        []    → ticker is seeded but no data available as of end_date.
        """
        if not self.is_seeded(ticker):
            return None

        try:
            if period == "ttm":
                return self._get_ttm(ticker, end_date, limit)
            q_period = "quarterly" if period == "quarterly" else "annual"
            return self._get_period(ticker, q_period, end_date, limit)
        except Exception as exc:
            logger.warning("SecStore.get_statements(%s, %s, %s) failed: %s", ticker, period, end_date, exc)
            return None

    def get_shares_outstanding(self, ticker: str, end_date: str) -> float | None:
        """Return the most recently filed shares-outstanding value as of end_date."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM sec_facts WHERE ticker=? AND section='dei' AND fact_key='shares_outstanding' AND filed<=? ORDER BY filed DESC LIMIT 1",
                    (ticker, end_date),
                ).fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except Exception as exc:
            logger.warning("SecStore.get_shares_outstanding(%s, %s) failed: %s", ticker, end_date, exc)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_period(self, ticker: str, period: str, end_date: str, limit: int):
        """Fetch up to `limit` annual or quarterly StatementBundles, newest-first."""
        with self._connect() as conn:
            # Get distinct period_ends where at least one fact was filed by end_date
            period_ends = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT period_end FROM sec_facts WHERE ticker=? AND period=? AND filed<=? ORDER BY period_end DESC LIMIT ?",
                    (ticker, period, end_date, limit),
                ).fetchall()
            ]

            if not period_ends:
                return []

            bundles = []
            for pe in period_ends:
                b = _Bundle(period_end=pe)
                # For each fact, take the most recently filed value as of end_date
                rows = conn.execute(
                    "SELECT section, fact_key, value FROM sec_facts f "
                    "WHERE ticker=? AND period=? AND period_end=? AND filed<=? "
                    "AND filed=(SELECT MAX(filed) FROM sec_facts "
                    "           WHERE ticker=f.ticker AND section=f.section "
                    "           AND fact_key=f.fact_key AND period=f.period "
                    "           AND period_end=f.period_end AND filed<=?)",
                    (ticker, period, pe, end_date, end_date),
                ).fetchall()
                for row in rows:
                    section, fact_key, value = row["section"], row["fact_key"], row["value"]
                    _assign(b, section, fact_key, value)
                bundles.append(_to_statement_bundle(b))

            return bundles

    def _get_ttm(self, ticker: str, end_date: str, limit: int):
        """Compute TTM by summing trailing 4 quarters; append annual bundles for growth."""
        q_bundles = self._get_period(ticker, "quarterly", end_date, 4)
        if len(q_bundles) < 4:
            logger.warning("SecStore TTM %s: only %d/4 quarters available", ticker, len(q_bundles))
            if not q_bundles:
                return []

        ttm_income: dict[str, float | None] = {}
        ttm_cashflow: dict[str, float | None] = {}
        ttm_balance: dict[str, float | None] = {}

        quarters_used = q_bundles[:4]
        most_recent_pe = quarters_used[0].period_end

        # Sum flow items across quarters
        for key in _INCOME_FIELDS:
            if key in _INCOME_STOCK_KEYS:
                # Use most-recent quarter's value for per-share/share-count
                ttm_income[key] = quarters_used[0].income.get(key)
            else:
                total, found = 0.0, 0
                for qb in quarters_used:
                    v = qb.income.get(key)
                    if v is not None:
                        total += v
                        found += 1
                ttm_income[key] = total if found == len(quarters_used) else None

        for key in _CASHFLOW_FIELDS:
            total, found = 0.0, 0
            for qb in quarters_used:
                v = qb.cashflow.get(key)
                if v is not None:
                    total += v
                    found += 1
            ttm_cashflow[key] = total if found == len(quarters_used) else None

        # Balance: use most recent quarter
        ttm_balance = dict(quarters_used[0].balance)

        from src.tools._yfinance_fundamentals import StatementBundle

        ttm_income["period_end"] = most_recent_pe
        ttm_balance["period_end"] = most_recent_pe
        ttm_cashflow["period_end"] = most_recent_pe
        ttm_bundle = StatementBundle(
            period_end=most_recent_pe,
            income=ttm_income,
            balance=ttm_balance,
            cashflow=ttm_cashflow,
            shares_outstanding=ttm_income.get("basic_shares_outstanding"),
        )

        annual_bundles = self._get_period(ticker, "annual", end_date, limit - 1)
        return [ttm_bundle] + annual_bundles


def _assign(bundle: _Bundle, section: str, fact_key: str, value: float | None) -> None:
    if section == "income":
        bundle.income[fact_key] = value
    elif section == "balance":
        bundle.balance[fact_key] = value
        if fact_key in ("total_equity_attributable_to_parent", "total_equity"):
            bundle.balance.setdefault("total_equity", value)
    elif section == "cashflow":
        bundle.cashflow[fact_key] = value
    elif section == "dei":
        if fact_key == "shares_outstanding":
            bundle.shares_outstanding = value


_store: SecStore | None = None
_store_lock = threading.Lock()


def get_sec_store() -> SecStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SecStore()
    return _store
