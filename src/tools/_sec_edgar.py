"""SEC EDGAR helpers: ticker→CIK lookup and filing RSS fetcher."""

from __future__ import annotations

import logging

import requests

from src.data.cache import get_cache

logger = logging.getLogger(__name__)

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_cache = get_cache()


def get_cik(ticker: str, user_agent: str) -> str | None:
    """Return the zero-padded 10-digit CIK for a ticker, or None if not found.

    The mapping is fetched from SEC once and cached indefinitely (it rarely changes).
    """
    cached = _cache.get_cik(ticker)
    if cached is not None:
        return cached

    try:
        resp = requests.get(_COMPANY_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=(5, 30))
        if resp.status_code != 200:
            logger.warning("SEC company_tickers.json returned %d", resp.status_code)
            return None
        data: dict = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch SEC company_tickers.json: %s", exc)
        return None

    # Build ticker→CIK mapping (SEC returns {index: {cik_str, ticker, title}})
    mapping: dict[str, str] = {}
    for entry in data.values():
        t = (entry.get("ticker") or "").upper()
        cik_raw = entry.get("cik_str")
        if t and cik_raw is not None:
            mapping[t] = str(cik_raw).zfill(10)

    _cache.set_cik_map(mapping)
    return mapping.get(ticker.upper())
