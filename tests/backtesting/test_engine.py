"""Behavioural tests for BacktestEngine."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd


def test_holiday_dates_skipped_before_agent_call():
    """Dates absent from prefetched price data must not be passed to run_cycle.

    Week of 2025-01-06 has five business days (Mon–Fri). We simulate Mon 6
    being a NYSE holiday by excluding it from the injected price DataFrame.
    The engine must skip that date entirely rather than burning LLM tokens
    before the missing-data guard fires.
    """
    from src.backtesting.engine import BacktestEngine

    trading_dates = pd.DatetimeIndex(["2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"])
    price_df = pd.DataFrame(
        {"close": [100.0] * len(trading_dates), "open": [99.0] * len(trading_dates)},
        index=trading_dates,
    )

    engine = BacktestEngine(
        agent=MagicMock(),
        tickers=["AAPL"],
        start_date="2025-01-06",
        end_date="2025-01-10",
        initial_capital=100_000.0,
        model_name="test-model",
        model_provider="test-provider",
        selected_analysts=None,
        initial_margin_requirement=0.0,
    )
    engine._results = MagicMock()
    engine._results.build_day_rows.return_value = []
    engine._benchmark = MagicMock()
    engine._benchmark.get_return_pct.return_value = 0.0
    engine._executor = MagicMock()
    engine._executor.execute_trade.return_value = 0
    engine._perf = MagicMock()
    engine._perf.compute_metrics.return_value = None

    mock_ctx = MagicMock()
    mock_ctx.signal_log_path = None
    mock_ctx.run_cycle.return_value = {"decisions": {}}
    mock_ctx.token_summary.return_value = {}

    @contextmanager
    def _mock_build(**kwargs):
        yield mock_ctx

    def _fake_prefetch(self_engine):
        self_engine._prefetched_prices = {"AAPL": price_df}
        self_engine._spy_prices = pd.DataFrame()

    with (
        patch.object(BacktestEngine, "_prefetch_data", _fake_prefetch),
        patch("src.backtesting.engine.PipelineContext.build", _mock_build),
        patch("src.backtesting.engine.get_backtest_store"),
        patch("src.backtesting.engine.progress"),
    ):
        engine.run_backtest()

    called_dates = [c.kwargs["date"] for c in mock_ctx.run_cycle.call_args_list]
    assert "2025-01-06" not in called_dates, f"Holiday date was passed to run_cycle: {called_dates}"
    assert called_dates, "run_cycle was never called — price data may not have been loaded"
    assert "2025-01-07" in called_dates
