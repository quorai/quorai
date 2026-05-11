from dotenv import load_dotenv

load_dotenv()

from src.backtesting.engine import BacktestEngine  # noqa: E402
from src.main import run_quorai  # noqa: E402
from src.utils.analysts import ALL_ANALYST_KEYS  # noqa: E402

engine = BacktestEngine(
    agent=run_quorai,
    tickers=["AAPL", "MSFT"],
    start_date="2026-04-11",
    end_date="2026-05-11",
    initial_capital=1_000.0,
    model_name="deepseek/deepseek-v4-flash",
    model_provider="OpenRouter",
    selected_analysts=ALL_ANALYST_KEYS,
    initial_margin_requirement=0.0,
)

metrics = engine.run_backtest()
values = engine.get_portfolio_values()

if values:
    first = values[0]["Portfolio Value"]
    last = values[-1]["Portfolio Value"]
    total_return = (last / first - 1.0) * 100.0 if first else 0.0
    print(f"\nTotal Return: {total_return:+.2f}%")
    print(f"Start: ${first:,.2f}  →  End: ${last:,.2f}")

if metrics:
    for k in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "max_drawdown_date"):
        if metrics.get(k) is not None:
            print(f"{k}: {metrics[k]}")
