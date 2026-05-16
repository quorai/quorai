from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Callable, TypeVar

T = TypeVar("T")

_DEFAULT_MAX_WORKERS = 4


def parallel_per_ticker(
    tickers: list[str],
    fn: Callable[[str], T],
    *,
    max_workers: int | None = None,
) -> dict[str, T]:
    """Run fn(ticker) for each ticker, concurrently when max_workers > 1.

    Returns a dict mapping ticker -> fn(ticker). Execution order is not guaranteed
    in parallel mode. Propagates the first exception raised by any worker.

    max_workers defaults to QUORAI_PARALLEL_TICKERS env var (int, default 4).
    Set to 1 to force serial execution.
    """
    if max_workers is None:
        max_workers = int(os.environ.get("QUORAI_PARALLEL_TICKERS", str(_DEFAULT_MAX_WORKERS)))

    if max_workers <= 1 or len(tickers) <= 1:
        return {ticker: fn(ticker) for ticker in tickers}

    effective_workers = min(max_workers, len(tickers))
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {ticker: executor.submit(fn, ticker) for ticker in tickers}

    return {ticker: fut.result() for ticker, fut in futures.items()}
