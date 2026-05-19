"""RV-23: equity curve must have exactly one entry per trading day (no duplicate day-0 seed)."""

import pandas as pd

from src.backtesting.engine import BacktestEngine
from tests.backtesting.integration.mocks import MockConfigurableAgent


def test_equity_curve_has_no_duplicate_day0():
    """Equity curve must not have a spurious pre-seeded day-0 entry.

    The engine used to initialise _portfolio_values with the starting capital at
    dates[0] before entering the loop, which then appended a second point for the
    same date.  After the fix there is exactly one entry per business day.
    """
    tickers = ["AAPL"]
    start_date = "2024-03-01"
    end_date = "2024-03-08"

    agent = MockConfigurableAgent([{}, {}], tickers)
    engine = BacktestEngine(
        agent=agent,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100_000.0,
        model_name="test-model",
        model_provider="test-provider",
        selected_analysts=None,
        initial_margin_requirement=0.0,
    )
    engine.run_backtest()
    portfolio_values = engine.get_portfolio_values()

    assert len(portfolio_values) >= 2, "Expected at least two trading days"

    # Ensure no duplicate dates
    dates = [pv["Date"] for pv in portfolio_values]
    assert len(dates) == len(set(str(d) for d in dates)), f"Duplicate dates in equity curve: {dates}"

    # First two entries must be different dates
    assert portfolio_values[0]["Date"] != portfolio_values[1]["Date"], f"Day-0 duplicate found: both entries have Date={portfolio_values[0]['Date']}"

    # Length must match the number of business days in the window
    expected_days = len(pd.date_range(start_date, end_date, freq="B"))
    assert len(portfolio_values) == expected_days, f"Expected {expected_days} entries in equity curve, got {len(portfolio_values)}"
