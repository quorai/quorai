import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any

_CACHE_DIR = Path(".cache")
_DB_FILE = _CACHE_DIR / "api_cache.db"
_LEGACY_PKL = _CACHE_DIR / "api_cache.pkl"

logger = logging.getLogger(__name__)

_CREATE_STMTS = """
CREATE TABLE IF NOT EXISTS prices (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS financial_metrics (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line_items (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS insider_trades (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS company_news (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS market_cap (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ticker_cik (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


class Cache:
    """In-memory-plus-SQLite cache for API responses."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path is not None else str(_DB_FILE)
        self._lock = threading.Lock()
        # Persistent connection for :memory: (each new connect() creates a fresh DB)
        self._mem_conn: sqlite3.Connection | None = None
        # In-process write-through caches (avoid hitting disk for every read)
        self._prices: dict[str, list[dict[str, Any]]] = {}
        self._financial_metrics: dict[str, list[dict[str, Any]]] = {}
        self._line_items: dict[str, list[dict[str, Any]]] = {}
        self._insider_trades: dict[str, list[dict[str, Any]]] = {}
        self._company_news: dict[str, list[dict[str, Any]]] = {}
        self._market_cap: dict[str, float] = {}
        self._ticker_to_cik: dict[str, str] = {}  # ticker (upper) → zero-padded 10-digit CIK
        self._init_db()
        self._load()
        if _LEGACY_PKL.exists():
            logger.warning("Legacy pickle cache %s is no longer used; delete it to suppress this warning", _LEGACY_PKL)

    def _connect(self) -> sqlite3.Connection:
        if self._db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._mem_conn
        _CACHE_DIR.mkdir(exist_ok=True)
        needs_init = not Path(self._db_path).exists()
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        if needs_init:
            conn.executescript(_CREATE_STMTS)
            conn.commit()
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(_CREATE_STMTS)
        conn.commit()

    def _load(self) -> None:
        """Populate in-process caches from SQLite on startup."""
        try:
            conn = self._connect()
            for row in conn.execute("SELECT key, payload FROM prices"):
                self._prices[row[0]] = json.loads(row[1])
            for row in conn.execute("SELECT key, payload FROM financial_metrics"):
                self._financial_metrics[row[0]] = json.loads(row[1])
            for row in conn.execute("SELECT key, payload FROM line_items"):
                self._line_items[row[0]] = json.loads(row[1])
            for row in conn.execute("SELECT key, payload FROM insider_trades"):
                self._insider_trades[row[0]] = json.loads(row[1])
            for row in conn.execute("SELECT key, payload FROM company_news"):
                self._company_news[row[0]] = json.loads(row[1])
            for row in conn.execute("SELECT key, value FROM market_cap"):
                self._market_cap[row[0]] = row[1]
            for row in conn.execute("SELECT key, payload FROM ticker_cik"):
                self._ticker_to_cik.update(json.loads(row[1]))
        except Exception as e:
            logger.warning("Failed to load SQLite cache (starting empty): %s", e)

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """Merge existing and new data, avoiding duplicates based on a key field."""
        if not existing:
            return new_data
        existing_keys = {item[key_field] for item in existing}
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        return merged

    def _upsert(self, table: str, key: str, payload: list[dict]) -> None:
        conn = self._connect()
        conn.execute(
            f"INSERT OR REPLACE INTO {table} (key, payload) VALUES (?, ?)",  # noqa: S608
            (key, json.dumps(payload)),
        )
        conn.commit()

    # ── prices ──────────────────────────────────────────────────────────────

    def get_prices(self, key: str) -> list[dict[str, Any]] | None:
        return self._prices.get(key)

    def set_prices(self, key: str, data: list[dict[str, Any]]) -> None:
        with self._lock:
            merged = self._merge_data(self._prices.get(key), data, key_field="time")
            self._prices[key] = merged
            self._upsert("prices", key, merged)

    # ── financial_metrics ────────────────────────────────────────────────────

    def get_financial_metrics(self, key: str) -> list[dict[str, Any]] | None:
        return self._financial_metrics.get(key)

    def set_financial_metrics(self, key: str, data: list[dict[str, Any]]) -> None:
        with self._lock:
            merged = self._merge_data(self._financial_metrics.get(key), data, key_field="report_period")
            self._financial_metrics[key] = merged
            self._upsert("financial_metrics", key, merged)

    # ── line_items ───────────────────────────────────────────────────────────

    def get_line_items(self, key: str) -> list[dict[str, Any]] | None:
        return self._line_items.get(key)

    def set_line_items(self, key: str, data: list[dict[str, Any]]) -> None:
        with self._lock:
            merged = self._merge_data(self._line_items.get(key), data, key_field="report_period")
            self._line_items[key] = merged
            self._upsert("line_items", key, merged)

    # ── insider_trades ───────────────────────────────────────────────────────

    def get_insider_trades(self, key: str) -> list[dict[str, Any]] | None:
        return self._insider_trades.get(key)

    def set_insider_trades(self, key: str, data: list[dict[str, Any]]) -> None:
        with self._lock:
            merged = self._merge_data(self._insider_trades.get(key), data, key_field="filing_date")
            self._insider_trades[key] = merged
            self._upsert("insider_trades", key, merged)

    # ── company_news ─────────────────────────────────────────────────────────

    def get_company_news(self, key: str) -> list[dict[str, Any]] | None:
        return self._company_news.get(key)

    def set_company_news(self, key: str, data: list[dict[str, Any]]) -> None:
        with self._lock:
            merged = self._merge_data(self._company_news.get(key), data, key_field="date")
            self._company_news[key] = merged
            self._upsert("company_news", key, merged)

    # ── market_cap ───────────────────────────────────────────────────────────

    def get_market_cap(self, key: str) -> float | None:
        return self._market_cap.get(key)

    def set_market_cap(self, key: str, value: float) -> None:
        with self._lock:
            self._market_cap[key] = value
            conn = self._connect()
            conn.execute("INSERT OR REPLACE INTO market_cap (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    # ── ticker_cik ───────────────────────────────────────────────────────────

    def get_cik(self, ticker: str) -> str | None:
        return self._ticker_to_cik.get(ticker.upper())

    def set_cik_map(self, mapping: dict[str, str]) -> None:
        """Persist the full ticker→CIK mapping (called once after fetching company_tickers.json)."""
        with self._lock:
            self._ticker_to_cik.update(mapping)
            conn = self._connect()
            conn.execute("INSERT OR REPLACE INTO ticker_cik (key, payload) VALUES (?, ?)", ("cik_map", json.dumps(self._ticker_to_cik)))
            conn.commit()


# Global singleton
_cache = Cache()


def get_cache() -> Cache:
    """Return the global cache instance."""
    return _cache
