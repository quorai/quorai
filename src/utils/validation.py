from __future__ import annotations

import re

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def validate_ticker(ticker: str) -> str:
    """Validate and return a normalized ticker symbol.

    Raises ValueError if the ticker contains characters outside the allowed set
    (uppercase letters, digits, dots, hyphens) or exceeds 10 characters.
    """
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker: {ticker!r} — must match [A-Z0-9.\\-]{{1,10}}")
    return ticker
