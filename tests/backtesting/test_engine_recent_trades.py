"""Tests that BacktestEngine accumulates recent_trades per-ticker and caps at 5 entries."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_engine(tickers: list[str], dates: pd.DatetimeIndex):
    from src.backtesting.engine import BacktestEngine

    engine = BacktestEngine(
        agent=MagicMock(),
        tickers=tickers,
        start_date=dates[0].strftime("%Y-%m-%d"),
        end_date=dates[-1].strftime("%Y-%m-%d"),
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
    engine._executor.execute_trade.side_effect = lambda ticker, action, qty, price, portfolio: qty
    engine._perf = MagicMock()
    engine._perf.compute_metrics.return_value = None
    return engine


def test_recent_trades_accumulates_executed_trades():
    """Executed trades are appended to _recent_trades after each cycle."""
    dates = pd.DatetimeIndex(["2025-01-07", "2025-01-08", "2025-01-09"])
    price_df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "open": [99.0, 100.0, 101.0]},
        index=dates,
    )

    engine = _make_engine(["MSFT"], dates)

    call_count = [0]

    def _run_cycle_side_effect(**kwargs):
        call_count[0] += 1
        return {"decisions": {"MSFT": {"action": "buy", "quantity": 5.0}}}

    mock_ctx = MagicMock()
    mock_ctx.signal_log_path = None
    mock_ctx.run_cycle.side_effect = _run_cycle_side_effect
    mock_ctx.token_summary.return_value = {}

    @contextmanager
    def _mock_build(**kwargs):
        yield mock_ctx

    def _fake_prefetch(self_engine):
        self_engine._prefetched_prices = {"MSFT": price_df}
        self_engine._spy_prices = pd.DataFrame()

    with (
        patch.object(engine.__class__, "_prefetch_data", _fake_prefetch),
        patch("src.backtesting.engine.PipelineContext.build", _mock_build),
        patch("src.backtesting.engine.get_backtest_store"),
        patch("src.backtesting.engine.progress"),
    ):
        engine.run_backtest()

    trades = engine._recent_trades["MSFT"]
    assert len(trades) == 3, f"Expected 3 trade entries, got {len(trades)}"
    assert trades[0]["action"] == "buy"
    assert trades[0]["qty"] == 5.0


def test_recent_trades_capped_at_five_entries():
    """After 6 executed trades the list stays at 5 (FIFO eviction)."""
    # Use 6 explicit business days across two weeks to avoid weekend gaps
    six_bdays = ["2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-13", "2025-01-14"]
    dates = pd.DatetimeIndex(six_bdays)
    price_df = pd.DataFrame(
        {"close": [100.0] * 6, "open": [99.0] * 6},
        index=dates,
    )

    engine = _make_engine(["MSFT"], dates)

    def _run_cycle_side_effect(**kwargs):
        return {"decisions": {"MSFT": {"action": "buy", "quantity": 1.0}}}

    mock_ctx = MagicMock()
    mock_ctx.signal_log_path = None
    mock_ctx.run_cycle.side_effect = _run_cycle_side_effect
    mock_ctx.token_summary.return_value = {}

    @contextmanager
    def _mock_build(**kwargs):
        yield mock_ctx

    def _fake_prefetch(self_engine):
        self_engine._prefetched_prices = {"MSFT": price_df}
        self_engine._spy_prices = pd.DataFrame()

    with (
        patch.object(engine.__class__, "_prefetch_data", _fake_prefetch),
        patch("src.backtesting.engine.PipelineContext.build", _mock_build),
        patch("src.backtesting.engine.get_backtest_store"),
        patch("src.backtesting.engine.progress"),
    ):
        engine.run_backtest()

    trades = engine._recent_trades["MSFT"]
    assert len(trades) == 5, f"Recent trades should be capped at 5, got {len(trades)}"
    # The oldest (day 1 = 2025-01-07) must be evicted; day 2 (2025-01-08) is now oldest
    assert trades[0]["date"] == "2025-01-08", f"Oldest entry should be 2025-01-08, got {trades[0]['date']}"
    assert trades[-1]["date"] == "2025-01-14", f"Newest entry should be 2025-01-14, got {trades[-1]['date']}"


def test_recent_trades_not_appended_for_zero_qty():
    """Trades with executed_qty == 0 (holds/failed trades) must not be added."""
    dates = pd.DatetimeIndex(["2025-01-07", "2025-01-08"])
    price_df = pd.DataFrame(
        {"close": [100.0, 101.0], "open": [99.0, 100.0]},
        index=dates,
    )

    engine = _make_engine(["MSFT"], dates)
    engine._executor.execute_trade.side_effect = lambda ticker, action, qty, price, portfolio: 0  # all fills = 0

    def _run_cycle_side_effect(**kwargs):
        return {"decisions": {"MSFT": {"action": "buy", "quantity": 5.0}}}

    mock_ctx = MagicMock()
    mock_ctx.signal_log_path = None
    mock_ctx.run_cycle.side_effect = _run_cycle_side_effect
    mock_ctx.token_summary.return_value = {}

    @contextmanager
    def _mock_build(**kwargs):
        yield mock_ctx

    def _fake_prefetch(self_engine):
        self_engine._prefetched_prices = {"MSFT": price_df}
        self_engine._spy_prices = pd.DataFrame()

    with (
        patch.object(engine.__class__, "_prefetch_data", _fake_prefetch),
        patch("src.backtesting.engine.PipelineContext.build", _mock_build),
        patch("src.backtesting.engine.get_backtest_store"),
        patch("src.backtesting.engine.progress"),
    ):
        engine.run_backtest()

    assert engine._recent_trades["MSFT"] == [], "Zero-qty fills must not be recorded"


def test_recent_trades_passed_to_run_cycle():
    """engine passes _recent_trades as recent_trades kwarg to ctx.run_cycle each call."""
    dates = pd.DatetimeIndex(["2025-01-07", "2025-01-08"])
    price_df = pd.DataFrame(
        {"close": [100.0, 101.0], "open": [99.0, 100.0]},
        index=dates,
    )

    engine = _make_engine(["MSFT"], dates)

    mock_ctx = MagicMock()
    mock_ctx.signal_log_path = None
    mock_ctx.run_cycle.return_value = {"decisions": {}}
    mock_ctx.token_summary.return_value = {}

    @contextmanager
    def _mock_build(**kwargs):
        yield mock_ctx

    def _fake_prefetch(self_engine):
        self_engine._prefetched_prices = {"MSFT": price_df}
        self_engine._spy_prices = pd.DataFrame()

    with (
        patch.object(engine.__class__, "_prefetch_data", _fake_prefetch),
        patch("src.backtesting.engine.PipelineContext.build", _mock_build),
        patch("src.backtesting.engine.get_backtest_store"),
        patch("src.backtesting.engine.progress"),
    ):
        engine.run_backtest()

    for call in mock_ctx.run_cycle.call_args_list:
        assert "recent_trades" in call.kwargs, "recent_trades kwarg missing from run_cycle call"
        assert isinstance(call.kwargs["recent_trades"], dict)
