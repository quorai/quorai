"""Seed .cache/sec_fundamentals.db with SEC EDGAR XBRL Company Facts.

Downloads fundamentals for every US-listed SEC filer and stores them in a local
SQLite database.  The database is then consulted by fetch_statements / fetch_market_cap
before falling back to yfinance, providing historically correct point-in-time data.

SEC EDGAR access is free but requires a User-Agent identifying the requester.
See: https://www.sec.gov/os/accessing-edgar-data

Usage:
    # Seed the full US market universe (~10k tickers, 3-4 hours, ~5-10 GB)
    QUORAI_SEC_USER_AGENT="your.email@example.com" uv run python experiments/seed_sec_fundamentals.py

    # Seed a specific subset (fast smoke-test, ~10-30 s)
    uv run python experiments/seed_sec_fundamentals.py --tickers NVDA,MSFT,META,AVGO,AMD

    # Skip tickers last synced within N days
    uv run python experiments/seed_sec_fundamentals.py --refresh-older-than 30

    # Override User-Agent on the command line
    uv run python experiments/seed_sec_fundamentals.py --user-agent "your.email@example.com"
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

# Resolve imports relative to repo root when run with `uv run python experiments/...`
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

import httpx
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from src.data.sec_store import FactRow as StoreFactRow
from src.data.sec_store import SecStore
from src.tools._sec_fundamentals import FactRow, fetch_company_facts, parse_company_facts

logger = logging.getLogger(__name__)
console = Console(stderr=True)

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC limits to 10 requests/second per user-agent; 0.11s gives ~9/s with headroom.
_SEC_RATE_LIMIT_SLEEP = 0.11


def _fetch_all_tickers(user_agent: str) -> dict[str, str]:
    """Return {ticker_upper: zero-padded-10-digit-cik} for all SEC filers."""
    headers = {"User-Agent": user_agent}
    try:
        resp = httpx.get(_COMPANY_TICKERS_URL, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        console.print(f"[red]Failed to fetch SEC company tickers: {exc}[/red]")
        sys.exit(1)

    mapping: dict[str, str] = {}
    for entry in resp.json().values():
        ticker = (entry.get("ticker") or "").strip().upper()
        cik_raw = entry.get("cik_str")
        name = entry.get("title") or ""
        if ticker and cik_raw is not None:
            mapping[ticker] = (str(cik_raw).zfill(10), name)
    return mapping  # type: ignore[return-value]


def _already_synced(store: SecStore, ticker: str, refresh_older_than_days: int) -> bool:
    """Return True if the ticker was synced recently enough to skip."""
    try:
        import sqlite3

        with sqlite3.connect(str(store._db_path)) as conn:
            row = conn.execute("SELECT last_synced FROM sec_meta WHERE ticker=?", (ticker,)).fetchone()
            if row is None:
                return False
            last = datetime.date.fromisoformat(row[0])
            return (datetime.date.today() - last).days < refresh_older_than_days
    except Exception:
        return False


def _convert_fact_rows(rows: list[FactRow]) -> list[StoreFactRow]:
    """Convert parser FactRows to store FactRows (same fields, separate dataclasses)."""
    return [
        StoreFactRow(
            ticker=r.ticker,
            section=r.section,
            fact_key=r.fact_key,
            period=r.period,
            period_end=r.period_end,
            filed=r.filed,
            value=r.value,
            unit=r.unit,
        )
        for r in rows
    ]


def seed(
    tickers: list[str] | None,
    user_agent: str,
    refresh_older_than_days: int,
    dry_run: bool = False,
) -> None:
    store = SecStore()

    console.print("[bold]Fetching SEC ticker→CIK map …[/bold]")
    all_tickers = _fetch_all_tickers(user_agent)  # {ticker: (cik, name)}

    if tickers:
        targets = {t.upper(): all_tickers.get(t.upper()) for t in tickers if t.upper() in all_tickers}
        missing = [t.upper() for t in tickers if t.upper() not in all_tickers]
        if missing:
            console.print(f"[yellow]Tickers not found in SEC universe: {', '.join(missing)}[/yellow]")
    else:
        targets = all_tickers  # type: ignore[assignment]

    console.print(f"[bold]Seeding {len(targets):,} tickers into {store._db_path}[/bold]")
    if dry_run:
        console.print("[yellow]DRY RUN — no writes[/yellow]")

    skipped = errors = seeded = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Seeding…", total=len(targets))

        for ticker, info in targets.items():
            progress.advance(task)
            progress.update(task, description=f"[cyan]{ticker:<8}[/cyan]")

            if info is None:
                errors += 1
                continue

            cik, company_name = info

            if _already_synced(store, ticker, refresh_older_than_days):
                skipped += 1
                continue

            facts = fetch_company_facts(cik, user_agent)
            time.sleep(_SEC_RATE_LIMIT_SLEEP)

            if facts is None:
                errors += 1
                continue

            rows = parse_company_facts(facts, ticker, cik)
            if not rows:
                # Ticker has CIK but no XBRL facts (e.g. blank-check co) — still mark synced
                if not dry_run:
                    store.upsert_facts(ticker, cik, company_name, [])
                seeded += 1
                continue

            store_rows = _convert_fact_rows(rows)
            if not dry_run:
                store.upsert_facts(ticker, cik, company_name, store_rows)
            seeded += 1

    console.print(f"[green]Done.[/green] seeded={seeded} skipped={skipped} errors={errors}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed SEC EDGAR fundamentals into .cache/sec_fundamentals.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to seed (default: all US filers)",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        dest="user_agent",
        help="SEC User-Agent string (email address). Overrides QUORAI_SEC_USER_AGENT env var.",
    )
    parser.add_argument(
        "--refresh-older-than",
        type=int,
        default=30,
        dest="refresh_older_than",
        metavar="DAYS",
        help="Re-seed tickers last synced more than N days ago (default: 30).",
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Parse only; do not write to DB.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    user_agent = args.user_agent or os.environ.get("QUORAI_SEC_USER_AGENT", "")
    if not user_agent:
        console.print(
            "[red]Error:[/red] SEC User-Agent is required.\nSet QUORAI_SEC_USER_AGENT='your.email@example.com' or use --user-agent.",
            highlight=False,
        )
        sys.exit(1)

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    seed(
        tickers=tickers,
        user_agent=user_agent,
        refresh_older_than_days=args.refresh_older_than,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
