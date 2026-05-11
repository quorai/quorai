from dotenv import load_dotenv

load_dotenv()

from src.backtesting.engine import BacktestEngine  # noqa: E402
from src.main import run_quorai  # noqa: E402
from src.utils.analysts import ANALYST_ORDER  # noqa: E402

ALL_ANALYSTS = [key for _, key in ANALYST_ORDER]

engine = BacktestEngine(
    agent=run_quorai,
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"],
    start_date="2026-02-01",
    end_date="2026-02-28",
    initial_capital=1_000.0,
    model_name="google/gemini-2.5-flash-lite",
    model_provider="OpenRouter",
    selected_analysts=["technical_analyst", "fundamentals_analyst", "warren_buffett", "stanley_druckenmiller", "cathie_wood", "nassim_taleb"],
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
