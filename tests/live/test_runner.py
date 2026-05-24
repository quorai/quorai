from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.live.runner import LiveRunner

_NY = ZoneInfo("America/New_York")
_PRE_OPEN = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_NY)
_POST_OPEN = datetime(2024, 1, 15, 10, 0, 0, tzinfo=_NY)


def _make_broker(equity=100_000.0):
    broker = MagicMock()
    account = MagicMock()
    account.equity = equity
    broker.get_account.return_value = account
    broker.get_positions.return_value = []
    return broker


def _make_snapshot():
    return {"cash": 100_000.0, "portfolio_value": 0.0, "positions": {}}


def _make_runner(broker, **kwargs):
    defaults = dict(
        tickers=["AAPL"],
        model_name="test-model",
        model_provider="test",
        selected_analysts=None,
        broker=broker,
    )
    defaults.update(kwargs)
    return LiveRunner(**defaults)


def _make_ctx(decisions=None):
    """Build a mock PipelineContext instance ready for use as a context manager."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.signal_log_path = None
    ctx.run_cycle.return_value = {
        "decisions": decisions or {"AAPL": {"action": "hold", "quantity": 0}},
        "analyst_signals": {},
    }
    ctx.token_summary.return_value = {}
    return ctx


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_prepare_calls_to_snapshot_and_run_cycle(mock_pipeline_cls, mock_to_snapshot, tmp_path):
    broker = _make_broker()
    mock_to_snapshot.return_value = _make_snapshot()
    ctx = _make_ctx()
    mock_pipeline_cls.build.return_value = ctx

    runner = _make_runner(broker)

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save, patch("src.live.runner.now_ny", return_value=_PRE_OPEN):
        decisions, snapshot = runner.prepare()

    mock_to_snapshot.assert_called_once()
    ctx.run_cycle.assert_called_once()
    mock_save.assert_called_once_with(float(broker.get_account.return_value.equity), log_dir="logs/live")
    assert "AAPL" in decisions


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_sod_equity_saved_on_first_run(mock_pipeline_cls, mock_to_snapshot):
    broker = _make_broker(equity=95_000.0)
    mock_to_snapshot.return_value = _make_snapshot()
    mock_pipeline_cls.build.return_value = _make_ctx(decisions={})

    runner = _make_runner(broker)

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save, patch("src.live.runner.now_ny", return_value=_PRE_OPEN):
        runner.prepare()

    mock_save.assert_called_once_with(95_000.0, log_dir="logs/live")
    assert runner._sod_equity == 95_000.0


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_sod_equity_loaded_if_already_saved(mock_pipeline_cls, mock_to_snapshot):
    broker = _make_broker(equity=95_000.0)
    mock_to_snapshot.return_value = _make_snapshot()
    mock_pipeline_cls.build.return_value = _make_ctx(decisions={})

    runner = _make_runner(broker)

    with patch("src.live.runner.load_sod_equity", return_value=100_000.0), patch("src.live.runner.save_sod_equity") as mock_save:
        runner.prepare()

    mock_save.assert_not_called()
    assert runner._sod_equity == 100_000.0


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_dry_run_skips_sod_equity_save(mock_pipeline_cls, mock_to_snapshot):
    # dry_run=True must not call save_sod_equity so weekend/off-hours dry runs work
    # without requiring a manually-created sod-equity file.
    broker = _make_broker(equity=75_000.0)
    mock_to_snapshot.return_value = _make_snapshot()
    mock_pipeline_cls.build.return_value = _make_ctx(decisions={})

    runner = _make_runner(broker, dry_run=True)

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save:
        runner.prepare()

    mock_save.assert_not_called()
    assert runner._sod_equity == 75_000.0


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_catch_up_fetches_sod_from_broker_history(mock_pipeline_cls, mock_to_snapshot):
    """catch_up=True fetches SOD from broker history and saves with allow_intraday=True."""
    broker = _make_broker(equity=98_000.0)
    broker.get_sod_equity = MagicMock(return_value=100_000.0)
    mock_to_snapshot.return_value = _make_snapshot()
    mock_pipeline_cls.build.return_value = _make_ctx(decisions={})

    runner = _make_runner(broker, catch_up=True)

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save:
        runner.prepare()

    broker.get_sod_equity.assert_called_once()
    mock_save.assert_called_once_with(100_000.0, log_dir="logs/live", allow_intraday=True)
    assert runner._sod_equity == 100_000.0


def test_execute_allow_queue_skips_reconcile_pre_market():
    """allow_queue=True + market not open → submitted orders reported as 'queued (pending open)'."""
    broker = _make_broker()
    broker.is_market_open_today.return_value = False

    runner = _make_runner(broker, allow_queue=True)
    runner._sod_equity = 100_000.0

    fake_executor = MagicMock()
    fake_executor.submitted_orders = {"AAPL": "ord-pre-001"}
    fake_executor.execute_decisions.return_value = {"AAPL": "submitted"}

    with patch("src.live.runner.LiveExecutor", return_value=fake_executor):
        results = runner.execute({"AAPL": {"action": "sell", "quantity": 10}})

    assert results["AAPL"] == "queued (pending open)"


def test_execute_allow_queue_reconciles_normally_when_market_open():
    """allow_queue=True + market IS open → reconcile proceeds as normal."""
    broker = _make_broker()
    broker.is_market_open_today.return_value = True

    runner = _make_runner(broker, allow_queue=True)
    runner._sod_equity = 100_000.0

    fake_executor = MagicMock()
    fake_executor.submitted_orders = {"AAPL": "ord-001"}
    fake_executor.execute_decisions.return_value = {"AAPL": "submitted"}

    fake_reconciler = MagicMock()
    fake_reconciler.reconcile.return_value = {"ord-001": {"status": "filled", "filled_qty": 10.0, "filled_avg_price": 200.0, "ticker": "AAPL"}}

    with patch("src.live.runner.LiveExecutor", return_value=fake_executor), patch("src.live.reconciler.Reconciler", return_value=fake_reconciler):
        results = runner.execute({"AAPL": {"action": "sell", "quantity": 10}})

    assert "filled" in results["AAPL"]
    fake_reconciler.reconcile.assert_called_once()


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_prepare_post_open_falls_back_gracefully(mock_pipeline_cls, mock_to_snapshot):
    """RV-08: No SOD file + post-open time → saves with allow_intraday=True instead of raising."""
    broker = _make_broker(equity=92_000.0)
    mock_to_snapshot.return_value = _make_snapshot()
    mock_pipeline_cls.build.return_value = _make_ctx(decisions={})

    runner = _make_runner(broker)

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save, patch("src.live.runner.now_ny", return_value=_POST_OPEN):
        runner.prepare()

    mock_save.assert_called_once_with(92_000.0, log_dir="logs/live", allow_intraday=True)
    assert runner._sod_equity == 92_000.0
