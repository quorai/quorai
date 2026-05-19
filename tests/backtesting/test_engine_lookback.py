"""Tests for BacktestEngine structural invariants."""

import inspect
from unittest.mock import MagicMock


class TestDefaultTemperature:
    def test_default_temperature_is_zero_for_backtest(self):
        """BacktestEngine must default llm_temperature to 0.0 for reproducibility."""
        from src.backtesting.engine import BacktestEngine

        engine = BacktestEngine(
            agent=MagicMock(),
            tickers=["AAPL"],
            start_date="2026-01-01",
            end_date="2026-01-31",
            initial_capital=100_000.0,
            model_name="test-model",
            model_provider="test-provider",
            selected_analysts=None,
            initial_margin_requirement=0.0,
        )
        assert engine._llm_temperature == 0.0

    def test_explicit_temperature_is_preserved(self):
        """When llm_temperature is provided explicitly it must not be overridden."""
        from src.backtesting.engine import BacktestEngine

        engine = BacktestEngine(
            agent=MagicMock(),
            tickers=["AAPL"],
            start_date="2026-01-01",
            end_date="2026-01-31",
            initial_capital=100_000.0,
            model_name="test-model",
            model_provider="test-provider",
            selected_analysts=None,
            initial_margin_requirement=0.0,
            llm_temperature=0.7,
        )
        assert engine._llm_temperature == 0.7


class TestEngineLookback:
    def test_per_cycle_lookback_uses_12_months(self):
        """
        The per-cycle lookback_start must use relativedelta(months=12) to feed
        252-day rolling indicators (historical vol, Hurst, 6-month momentum).
        Previously was relativedelta(months=1) which starved those indicators.
        """
        from src.backtesting import engine as engine_mod

        source = inspect.getsource(engine_mod)
        # Find the line that sets lookback_start in the per-cycle loop.
        lookback_lines = [line.strip() for line in source.splitlines() if "lookback_start" in line and "relativedelta" in line]
        assert lookback_lines, "Expected a line setting lookback_start with relativedelta in engine.py"
        assert any("months=12" in line for line in lookback_lines), f"lookback_start should use relativedelta(months=12); found: {lookback_lines}"
