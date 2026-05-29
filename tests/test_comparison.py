from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.backtesting.comparison import RunConfig, RunResult, _print_comparison_table, run_comparison


def _make_metrics(sharpe: float = 1.0, sortino: float = 1.5, drawdown: float = -0.05):
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": drawdown,
        "long_short_ratio": None,
        "gross_exposure": None,
        "net_exposure": None,
    }


def _make_config(label: str = "Test") -> RunConfig:
    return RunConfig(
        label=label,
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-31",
        initial_capital=1000.0,
        selected_analysts=["fundamentals_agent"],
        model_name="test-model",
        model_provider="OpenAI",
    )


@patch("src.backtesting.comparison.BacktestEngine")
def test_run_comparison_returns_results(mock_engine_cls, tmp_path):
    engine_instance = MagicMock()
    engine_instance.run_backtest.return_value = _make_metrics()
    engine_instance.get_portfolio_values.return_value = [{"Portfolio Value": 1100.0}]
    engine_instance.get_token_summary.return_value = {"calls": 5, "input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    engine_instance.get_cost_summary.return_value = {"total_slippage": 0.0, "total_commission": 0.0, "total_borrow": 0.0, "total_costs": 0.0}
    mock_engine_cls.return_value = engine_instance

    configs = [_make_config("Run A"), _make_config("Run B")]
    results = run_comparison(configs, output_dir=str(tmp_path))

    assert len(results) == 2
    assert results[0].label == "Run A"
    assert results[1].label == "Run B"
    assert results[0].final_value == 1100.0
    assert results[0].token_summary["calls"] == 5


@patch("src.backtesting.comparison.BacktestEngine")
def test_run_comparison_writes_json(mock_engine_cls, tmp_path):
    engine_instance = MagicMock()
    engine_instance.run_backtest.return_value = _make_metrics(sharpe=0.5)
    engine_instance.get_portfolio_values.return_value = [{"Portfolio Value": 950.0}]
    engine_instance.get_token_summary.return_value = {"calls": 2, "input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    engine_instance.get_cost_summary.return_value = {"total_slippage": 0.0, "total_commission": 0.0, "total_borrow": 0.0, "total_costs": 0.0}
    mock_engine_cls.return_value = engine_instance

    configs = [_make_config("Only")]
    run_comparison(configs, output_dir=str(tmp_path))

    out_files = list(tmp_path.glob("comparison-*.json"))
    assert len(out_files) == 1

    data = json.loads(out_files[0].read_text())
    assert isinstance(data, list)
    assert data[0]["label"] == "Only"
    assert "metrics" in data[0]
    assert "token_summary" in data[0]
    assert "final_value" in data[0]
    assert "duration_seconds" in data[0]


def test_print_comparison_table_smoke():
    results = [
        RunResult(
            label="A",
            metrics=_make_metrics(),
            token_summary={"calls": 3, "total_tokens": 200},
            final_value=1050.0,
            duration_seconds=12.3,
        ),
        RunResult(
            label="B",
            metrics=_make_metrics(sharpe=0.8, drawdown=-0.10),
            token_summary={"calls": 5, "total_tokens": 400},
            final_value=980.0,
            duration_seconds=18.7,
        ),
    ]
    _print_comparison_table(results, initial_capital=1000.0)


@patch("src.backtesting.comparison.BacktestEngine")
def test_run_comparison_empty_portfolio_values(mock_engine_cls, tmp_path):
    engine_instance = MagicMock()
    engine_instance.run_backtest.return_value = _make_metrics()
    engine_instance.get_portfolio_values.return_value = []
    engine_instance.get_token_summary.return_value = {"calls": 1, "total_tokens": 15}
    engine_instance.get_cost_summary.return_value = {"total_slippage": 0.0, "total_commission": 0.0, "total_borrow": 0.0, "total_costs": 0.0}
    mock_engine_cls.return_value = engine_instance

    cfg = _make_config("Empty PV")
    results = run_comparison([cfg], output_dir=str(tmp_path))

    assert results[0].final_value == cfg.initial_capital
