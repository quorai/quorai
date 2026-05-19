"""Tests for the backtesting CLI argument parsing."""

from __future__ import annotations

import argparse


def _parse(argv: list[str]) -> argparse.Namespace:
    """Build the standard run parser and parse the given argv."""
    from src.backtesting.cli import _add_common_args

    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    return parser.parse_args(argv)


def test_days_alias_still_works():
    """--days N must still be accepted without error (backwards-compatible alias)."""
    args = _parse(["--tickers", "AAPL", "--days", "30", "--model", "test-model"])
    assert args.days == 30


def test_calendar_days_accepted():
    """--calendar-days N must be accepted and stored in args.days."""
    args = _parse(["--tickers", "AAPL", "--calendar-days", "60", "--model", "test-model"])
    assert args.days == 60
