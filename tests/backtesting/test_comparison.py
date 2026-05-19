import re

from src.backtesting.comparison import RunConfig
from src.backtesting.engine import BacktestEngine
from src.main import run_quorai


def _make_engine(label: str) -> BacktestEngine:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return BacktestEngine(
        agent=run_quorai,
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-02",
        end_date="2024-01-31",
        initial_capital=100_000.0,
        model_name="deepseek/deepseek-v4-flash",
        model_provider="OpenRouter",
        selected_analysts=None,
        initial_margin_requirement=0.0,
        run_label=slug,
    )


def test_run_ids_differ_across_configs():
    cfg_a = RunConfig(label="Full analyst set", tickers=["AAPL", "MSFT"], start_date="2024-01-02", end_date="2024-01-31", initial_capital=100_000.0)
    cfg_b = RunConfig(label="Quant only", tickers=["AAPL", "MSFT"], start_date="2024-01-02", end_date="2024-01-31", initial_capital=100_000.0)

    slug_a = re.sub(r"[^a-z0-9]+", "-", cfg_a.label.lower()).strip("-")
    slug_b = re.sub(r"[^a-z0-9]+", "-", cfg_b.label.lower()).strip("-")

    engine_a = _make_engine(cfg_a.label)
    engine_b = _make_engine(cfg_b.label)

    assert engine_a.run_id != engine_b.run_id
    assert slug_a in engine_a.run_id
    assert slug_b in engine_b.run_id


def test_run_id_no_label_omits_suffix():
    engine = BacktestEngine(
        agent=run_quorai,
        tickers=["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-31",
        initial_capital=100_000.0,
        model_name="deepseek/deepseek-v4-flash",
        model_provider="OpenRouter",
        selected_analysts=None,
        initial_margin_requirement=0.0,
    )
    assert engine.run_id == "AAPL-2024-01-02-2024-01-31"


def test_run_id_slug_normalises_label():
    engine = BacktestEngine(
        agent=run_quorai,
        tickers=["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-31",
        initial_capital=100_000.0,
        model_name="deepseek/deepseek-v4-flash",
        model_provider="OpenRouter",
        selected_analysts=None,
        initial_margin_requirement=0.0,
        run_label="full-analyst-set",
    )
    assert engine.run_id == "AAPL-2024-01-02-2024-01-31-full-analyst-set"
