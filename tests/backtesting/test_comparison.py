import re
from unittest.mock import patch

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
    from datetime import datetime
    import re

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
    today = datetime.now().strftime("%Y-%m-%d")
    # Format: YYYY-MM-DD-HHMMSS-AAPL-2024-01-02-2024-01-31-<8charhash>
    assert re.match(rf"^{re.escape(today)}-\d{{6}}-AAPL-2024-01-02-2024-01-31-[0-9a-f]{{8}}$", engine.run_id)


def test_run_id_slug_normalises_label():
    from datetime import datetime
    import re

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
    today = datetime.now().strftime("%Y-%m-%d")
    # Format: YYYY-MM-DD-HHMMSS-AAPL-2024-01-02-2024-01-31-full-analyst-set-<8charhash>
    assert re.match(rf"^{re.escape(today)}-\d{{6}}-AAPL-2024-01-02-2024-01-31-full-analyst-set-[0-9a-f]{{8}}$", engine.run_id)


def test_runconfig_uses_settings_model():
    """RunConfig.model_name / model_provider must reflect Settings, not hardcoded values."""
    from src.config import Settings

    custom_settings = Settings(DEFAULT_MODEL="custom/model-v1", DEFAULT_PROVIDER="CustomProvider")

    with patch("src.backtesting.comparison.get_settings", return_value=custom_settings):
        cfg = RunConfig(
            label="test",
            tickers=["AAPL"],
            start_date="2024-01-02",
            end_date="2024-01-31",
            initial_capital=100_000.0,
        )

    assert cfg.model_name == "custom/model-v1"
    assert cfg.model_provider == "CustomProvider"
