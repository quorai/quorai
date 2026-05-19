from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.llm.request import RunRequest
from src.orchestration.preflight import PipelineContext


def _make_output(decisions=None, analyst_signals=None, token_usage=None):
    return {
        "decisions": decisions or {},
        "analyst_signals": analyst_signals or {},
        "token_usage": token_usage or [],
    }


def _portfolio():
    return {
        "cash": 100_000.0,
        "margin_used": 0.0,
        "margin_requirement": 0.0,
        "positions": {},
        "realized_gains": {},
    }


def _ctx(**overrides) -> PipelineContext:
    """Return a PipelineContext with a mock controller; keyword args override defaults."""
    defaults = dict(
        agent=MagicMock(),
        tickers=["AAPL"],
        run_id="test-run",
        mode="backtest",
        model_name="test-model",
        model_provider="test",
        selected_analysts=None,
        llm_temperature=None,
        show_reasoning=False,
        use_regime_selection=False,
        use_conviction_weights=False,
        conviction_weights={},
        signal_logger=None,
        request=None,
    )
    defaults.update(overrides)
    ctx = PipelineContext(**defaults)
    ctx._controller = MagicMock()
    ctx._controller.run_agent.return_value = _make_output()
    return ctx


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


@patch("src.orchestration.preflight._atomic_json_write")
@patch("src.orchestration.preflight.SignalLogger")
def test_build_opens_signal_logger(mock_logger_cls, _mock_write):
    ctx = PipelineContext.build(
        agent=MagicMock(),
        tickers=["AAPL"],
        run_id="my-run",
        model_name="m",
        model_provider="p",
        enable_signal_log=True,
    )
    mock_logger_cls.assert_called_once_with("my-run", log_dir="logs")
    assert ctx.signal_log_path is not None


@patch("src.orchestration.preflight._atomic_json_write")
@patch("src.orchestration.preflight.SignalLogger")
def test_build_skips_signal_logger_when_disabled(mock_logger_cls, _mock_write):
    ctx = PipelineContext.build(
        agent=MagicMock(),
        tickers=["AAPL"],
        run_id="my-run",
        model_name="m",
        model_provider="p",
        enable_signal_log=False,
    )
    mock_logger_cls.assert_not_called()
    assert ctx.signal_log_path is None


@patch("src.orchestration.preflight._atomic_json_write")
@patch("src.orchestration.preflight.load_weights", return_value={"buffett": 0.8})
@patch("src.orchestration.preflight.SignalLogger")
def test_build_loads_conviction_weights(mock_logger_cls, mock_load_weights, _mock_write):
    ctx = PipelineContext.build(
        agent=MagicMock(),
        tickers=["AAPL"],
        run_id="r",
        model_name="m",
        model_provider="p",
        use_conviction_weights=True,
    )
    mock_load_weights.assert_called_once()
    assert ctx._conviction_weights == {"buffett": 0.8}


@patch("src.orchestration.preflight._atomic_json_write")
@patch("src.orchestration.preflight.load_weights", return_value={})
@patch("src.orchestration.preflight.SignalLogger")
def test_build_skips_load_weights_when_not_enabled(mock_logger_cls, mock_load_weights, _mock_write):
    PipelineContext.build(
        agent=MagicMock(),
        tickers=["AAPL"],
        run_id="r",
        model_name="m",
        model_provider="p",
        use_conviction_weights=False,
    )
    mock_load_weights.assert_not_called()


# ---------------------------------------------------------------------------
# run_cycle — request forwarding (regression test for the backtest missing-request bug)
# ---------------------------------------------------------------------------


def test_run_cycle_passes_request_to_controller():
    """Regression: BacktestEngine was missing request= kwarg; verify it flows through."""
    request = RunRequest(agent_models={"warren_buffett_agent": ("gpt-4", "OpenAI")})
    ctx = _ctx(request=request)

    ctx.run_cycle(
        date="2026-01-15",
        lookback_start="2025-12-15",
        portfolio=_portfolio(),
        signal_prices={"AAPL": 200.0},
    )

    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["request"] is request


def test_run_cycle_passes_none_request_when_not_set():
    ctx = _ctx(request=None)
    ctx.run_cycle(date="2026-01-15", lookback_start="2025-12-15", portfolio=_portfolio(), signal_prices={})
    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["request"] is None


# ---------------------------------------------------------------------------
# run_cycle — signal logging
# ---------------------------------------------------------------------------


def test_run_cycle_calls_log_day_with_correct_args():
    signal_logger = MagicMock()
    signal_logger.path = "/tmp/t.jsonl"
    signals = {"buffett_agent": {"AAPL": {"signal": "bullish", "confidence": 70}}}
    ctx = _ctx(signal_logger=signal_logger)
    ctx._controller.run_agent.return_value = _make_output(analyst_signals=signals)

    prices = {"AAPL": 200.0}
    ctx.run_cycle(date="2026-01-15", lookback_start="2025-12-15", portfolio=_portfolio(), signal_prices=prices)

    signal_logger.log_day.assert_called_once_with("2026-01-15", signals, prices)


def test_run_cycle_skips_log_day_when_no_signal_logger():
    ctx = _ctx(signal_logger=None)
    output = ctx.run_cycle(date="2026-01-15", lookback_start="2025-12-15", portfolio=_portfolio(), signal_prices={})
    assert output["decisions"] == {}


# ---------------------------------------------------------------------------
# run_cycle — regime selection
# ---------------------------------------------------------------------------


def _spy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [400.0] * 100},
        index=pd.date_range("2025-09-01", periods=100, freq="B"),
    )


def test_regime_selection_narrows_analysts():
    ctx = _ctx(
        selected_analysts=["all_analyst"],
        use_regime_selection=True,
    )

    with (
        patch("src.orchestration.preflight.classify_regime_with_indicators") as mock_classify,
        patch("src.orchestration.preflight.select_analysts_for_regime") as mock_select,
    ):
        from src.regime.classifier import MarketRegime

        mock_classify.return_value = (MarketRegime.BULL_TREND, {})
        mock_select.return_value = ["growth_analyst"]

        ctx.run_cycle(
            date="2026-01-15",
            lookback_start="2025-12-15",
            portfolio=_portfolio(),
            signal_prices={},
            spy_df=_spy_df(),
        )

    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["selected_analysts"] == ["growth_analyst"]


def test_regime_selection_falls_back_when_returns_none():
    base = ["analyst_a", "analyst_b"]
    ctx = _ctx(selected_analysts=base, use_regime_selection=True)

    with (
        patch("src.orchestration.preflight.classify_regime_with_indicators") as mock_classify,
        patch("src.orchestration.preflight.select_analysts_for_regime") as mock_select,
    ):
        from src.regime.classifier import MarketRegime

        mock_classify.return_value = (MarketRegime.NEUTRAL, {})
        mock_select.return_value = None  # None = all groups

        ctx.run_cycle(
            date="2026-01-15",
            lookback_start="2025-12-15",
            portfolio=_portfolio(),
            signal_prices={},
            spy_df=_spy_df(),
        )

    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["selected_analysts"] == base


def test_regime_selection_skips_when_spy_df_empty():
    base = ["analyst_a"]
    ctx = _ctx(selected_analysts=base, use_regime_selection=True)

    ctx.run_cycle(
        date="2026-01-15",
        lookback_start="2025-12-15",
        portfolio=_portfolio(),
        signal_prices={},
        spy_df=pd.DataFrame(),
    )

    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["selected_analysts"] == base


def test_regime_selection_skips_when_spy_df_is_none():
    base = ["analyst_a"]
    ctx = _ctx(selected_analysts=base, use_regime_selection=True)

    ctx.run_cycle(
        date="2026-01-15",
        lookback_start="2025-12-15",
        portfolio=_portfolio(),
        signal_prices={},
        spy_df=None,
    )

    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["selected_analysts"] == base


def test_regime_selection_disabled_ignores_spy_df():
    base = ["analyst_a"]
    ctx = _ctx(selected_analysts=base, use_regime_selection=False)

    with patch("src.orchestration.preflight.classify_regime_with_indicators") as mock_classify:
        ctx.run_cycle(
            date="2026-01-15",
            lookback_start="2025-12-15",
            portfolio=_portfolio(),
            signal_prices={},
            spy_df=_spy_df(),
        )
        mock_classify.assert_not_called()

    _, call_kwargs = ctx._controller.run_agent.call_args
    assert call_kwargs["selected_analysts"] == base


# ---------------------------------------------------------------------------
# token_summary
# ---------------------------------------------------------------------------


def test_token_usage_accumulates_across_cycles():
    ctx = _ctx()
    records_a = [{"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_creation_tokens": 0}]
    records_b = [{"input_tokens": 200, "output_tokens": 75, "cache_read_tokens": 10, "cache_creation_tokens": 5}]
    ctx._controller.run_agent.side_effect = [
        _make_output(token_usage=records_a),
        _make_output(token_usage=records_b),
    ]

    ctx.run_cycle(date="2026-01-15", lookback_start="2025-12-15", portfolio=_portfolio(), signal_prices={})
    ctx.run_cycle(date="2026-01-16", lookback_start="2025-12-16", portfolio=_portfolio(), signal_prices={})

    summary = ctx.token_summary()
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 300
    assert summary["output_tokens"] == 125
    assert summary["total_tokens"] == 425
    assert summary["cache_read_tokens"] == 10
    assert summary["cache_creation_tokens"] == 5


def test_token_summary_returns_empty_when_no_cycles():
    ctx = _ctx()
    summary = ctx.token_summary()
    # No LLM calls → no call stats, but failure counters are always present
    assert "calls" not in summary
    assert summary["bundle_write_failures"] == 0
    assert summary["manifest_write_failures"] == 0


@patch("src.orchestration.preflight._atomic_json_write", side_effect=OSError("disk full"))
def test_bundle_write_failure_counted(_mock_write):
    ctx = _ctx()
    ctx.run_cycle(date="2026-01-15", lookback_start="2025-12-15", portfolio=_portfolio(), signal_prices={})
    assert ctx._bundle_write_failures == 1
    assert ctx._cycle_files == [], "failed write must not be appended to _cycle_files"


@patch("src.orchestration.preflight._atomic_json_write", side_effect=OSError("disk full"))
def test_token_summary_includes_failure_count(_mock_write):
    ctx = _ctx()
    ctx.run_cycle(date="2026-01-15", lookback_start="2025-12-15", portfolio=_portfolio(), signal_prices={})
    summary = ctx.token_summary()
    assert summary["bundle_write_failures"] == 1


# ---------------------------------------------------------------------------
# close / context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_signal_logger():
    signal_logger = MagicMock()
    signal_logger.path = "/tmp/t.jsonl"
    ctx = _ctx(signal_logger=signal_logger)

    with ctx:
        pass

    signal_logger.close.assert_called_once()


def test_close_without_signal_logger_is_safe():
    ctx = _ctx(signal_logger=None)
    ctx.close()  # must not raise


def test_context_manager_closes_even_on_exception():
    signal_logger = MagicMock()
    signal_logger.path = "/tmp/t.jsonl"
    ctx = _ctx(signal_logger=signal_logger)

    with pytest.raises(RuntimeError):
        with ctx:
            raise RuntimeError("boom")

    signal_logger.close.assert_called_once()
