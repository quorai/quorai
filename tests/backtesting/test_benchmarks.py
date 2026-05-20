from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.benchmarks import BenchmarkCalculator


def _make_df(prices: list[float], dates: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"close": prices}, index=pd.to_datetime(dates))
    df.index.name = "Date"
    return df


START = "2024-01-02"
END = "2024-01-05"


@pytest.fixture()
def calc() -> BenchmarkCalculator:
    bm = BenchmarkCalculator()
    # SPY: 100 → 110 (+10%)
    bm.load(_make_df([100.0, 105.0, 108.0, 110.0], ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]), "SPY")
    # AAPL: 200 → 220 (+10%), same dates
    bm.load(_make_df([200.0, 210.0, 215.0, 220.0], ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]), "AAPL")
    return bm


def test_spy_return(calc: BenchmarkCalculator) -> None:
    assert calc.get_return_pct("SPY", START, END) == pytest.approx(10.0)


def test_aapl_return(calc: BenchmarkCalculator) -> None:
    assert calc.get_return_pct("AAPL", START, END) == pytest.approx(10.0)


def test_tickers_are_independent(calc: BenchmarkCalculator) -> None:
    # SPY and AAPL have the same % return here, but different absolute prices
    spy_ret = calc.get_return_pct("SPY", START, END)
    aapl_ret = calc.get_return_pct("AAPL", START, END)
    assert spy_ret == aapl_ret  # both +10%; if cross-contaminated one would differ

    # Now load a new ticker with a different return and confirm it doesn't affect the others
    calc.load(_make_df([50.0, 60.0], ["2024-01-02", "2024-01-05"]), "MSFT")
    assert calc.get_return_pct("MSFT", START, END) == pytest.approx(20.0)
    assert calc.get_return_pct("SPY", START, END) == pytest.approx(10.0)
    assert calc.get_return_pct("AAPL", START, END) == pytest.approx(10.0)


def test_unknown_ticker_returns_none(calc: BenchmarkCalculator) -> None:
    assert calc.get_return_pct("UNKNOWN", START, END) is None


def test_default_ticker_is_spy() -> None:
    bm = BenchmarkCalculator()
    bm.load(_make_df([100.0, 115.0], ["2024-01-02", "2024-01-05"]))  # no ticker arg → "SPY"
    assert bm.get_return_pct("SPY", START, END) == pytest.approx(15.0)


def test_empty_window_returns_none(calc: BenchmarkCalculator) -> None:
    assert calc.get_return_pct("SPY", "2023-01-01", "2023-01-31") is None
