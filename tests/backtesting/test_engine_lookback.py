"""Tests for R30: backtest engine per-cycle lookback must cover 12 months."""

import inspect


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
