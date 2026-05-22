"""Tests that the regime tag detected in PipelineContext.run_cycle reaches run_agent."""

from unittest.mock import MagicMock, patch

import pandas as pd

import src.backtesting  # noqa: F401 — break circular: backtesting.__init__ loads controller before engine
from src.orchestration.preflight import PipelineContext


def _make_output():
    return {
        "decisions": {},
        "analyst_signals": {},
        "token_usage": [],
        "group_signals": {},
        "debate_summaries": {},
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


def _spy_bull_df() -> pd.DataFrame:
    """Minimal SPY DataFrame that classifies as BULL_TREND (above 20-SMA, low vol ratio)."""
    import numpy as np

    dates = pd.date_range("2023-07-01", periods=90, freq="B")
    # Steadily rising prices — current > SMA20, low volatility
    prices = np.linspace(400, 430, len(dates))
    df = pd.DataFrame({"close": prices}, index=dates)
    return df


def _spy_empty_df() -> pd.DataFrame:
    return pd.DataFrame()


class TestRegimePropagation:
    def test_regime_off_passes_none_to_run_agent(self):
        ctx = _ctx(use_regime_selection=False)
        with patch.object(ctx._controller, "run_agent", return_value=_make_output()) as mock_run:
            ctx.run_cycle(date="2023-10-01", lookback_start="2023-09-01", portfolio=_portfolio(), signal_prices={"AAPL": 150.0})
        _, kwargs = mock_run.call_args
        assert kwargs.get("regime") is None

    def test_regime_on_empty_spy_passes_none_to_run_agent(self):
        ctx = _ctx(use_regime_selection=True)
        with patch.object(ctx._controller, "run_agent", return_value=_make_output()) as mock_run:
            ctx.run_cycle(
                date="2023-10-01",
                lookback_start="2023-09-01",
                portfolio=_portfolio(),
                signal_prices={"AAPL": 150.0},
                spy_df=_spy_empty_df(),
            )
        _, kwargs = mock_run.call_args
        assert kwargs.get("regime") is None

    def test_regime_on_bull_spy_passes_bull_trend_to_run_agent(self):
        ctx = _ctx(use_regime_selection=True)
        spy_df = _spy_bull_df()
        as_of = spy_df.index[-1].strftime("%Y-%m-%d")
        with patch.object(ctx._controller, "run_agent", return_value=_make_output()) as mock_run:
            ctx.run_cycle(
                date=as_of,
                lookback_start="2023-07-01",
                portfolio=_portfolio(),
                signal_prices={"AAPL": 150.0},
                spy_df=spy_df,
            )
        _, kwargs = mock_run.call_args
        assert kwargs.get("regime") == "bull_trend"
