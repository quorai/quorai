from unittest.mock import MagicMock, patch

from src.live.runner import LiveRunner


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

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save:
        decisions, snapshot = runner.prepare()

    mock_to_snapshot.assert_called_once()
    ctx.run_cycle.assert_called_once()
    mock_save.assert_called_once_with(float(broker.get_account.return_value.equity))
    assert "AAPL" in decisions


@patch("src.live.runner.to_snapshot")
@patch("src.live.runner.PipelineContext")
def test_sod_equity_saved_on_first_run(mock_pipeline_cls, mock_to_snapshot):
    broker = _make_broker(equity=95_000.0)
    mock_to_snapshot.return_value = _make_snapshot()
    mock_pipeline_cls.build.return_value = _make_ctx(decisions={})

    runner = _make_runner(broker)

    with patch("src.live.runner.load_sod_equity", return_value=None), patch("src.live.runner.save_sod_equity") as mock_save:
        runner.prepare()

    mock_save.assert_called_once_with(95_000.0)
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
