"""Tests for the backtesting CLI argument parsing."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


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


def test_main_run_writes_cli_args_and_result_to_manifest(monkeypatch):
    fake_metrics = {
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "max_drawdown": -5.0,
        "max_drawdown_date": "2024-01-15",
        "long_short_ratio": 0.8,
        "gross_exposure": 0.9,
        "net_exposure": 0.7,
    }
    fake_values = [{"Portfolio Value": 100_000.0}, {"Portfolio Value": 110_000.0}]

    captured: dict = {}

    def fake_update(run_id, patch, log_dir="logs"):
        captured["run_id"] = run_id
        captured["patch"] = patch

    monkeypatch.setattr("sys.argv", ["src.backtesting", "--tickers", "AAPL", "--model", "test-model"])

    with (
        patch("src.backtesting.cli.BacktestEngine") as MockEngine,
        patch("src.backtesting.cli.run_quorai"),
        patch("src.backtesting.cli.check_provider_api_key"),
        patch("src.backtesting.cli.validate_ticker", side_effect=lambda t: t),
        patch("src.backtesting.cli.select_model", return_value=("test-model", "test-provider")),
        patch("src.backtesting.cli.get_profile", return_value=None),
        patch("src.backtesting.cli.update_run_manifest", side_effect=fake_update),
    ):
        bm = MagicMock()
        bm.get_return_pct.return_value = 5.0
        engine_instance = MagicMock()
        engine_instance.run_backtest.return_value = fake_metrics
        engine_instance.get_portfolio_values.return_value = fake_values
        engine_instance.get_benchmark.return_value = bm
        engine_instance.run_id = "AAPL-2024-01-01-2024-01-31"
        MockEngine.return_value = engine_instance

        from src.backtesting.cli import _main_run

        rc = _main_run(["--tickers", "AAPL", "--model", "test-model"])

    assert rc == 0
    assert "cli_args" in captured["patch"]
    assert captured["patch"]["cli_args"]["parsed"]["seed"] == 42
    assert "AAPL" in " ".join(str(a) for a in captured["patch"]["cli_args"]["argv"])
    assert "result" in captured["patch"]
    assert captured["patch"]["result"]["total_return_pct"] == pytest.approx(10.0)
    assert captured["patch"]["result"]["metrics"]["sharpe_ratio"] == 1.5
    assert captured["patch"]["result"]["baselines"]["spy_return_pct"] == 5.0
    assert "AAPL" in captured["patch"]["result"]["baselines"]["tickers"]
