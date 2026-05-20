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


# ---------------------------------------------------------------------------
# get_daily_returns
# ---------------------------------------------------------------------------


def test_daily_returns_length(calc: BenchmarkCalculator) -> None:
    # 4 price points → 3 daily returns after pct_change + dropna
    series = calc.get_daily_returns("SPY", START, END)
    assert series is not None
    assert len(series) == 3


def test_daily_returns_values(calc: BenchmarkCalculator) -> None:
    series = calc.get_daily_returns("SPY", START, END)
    assert series is not None
    # SPY prices: 100, 105, 108, 110 → returns: 0.05, 3/105≈0.02857, 2/108≈0.01852
    assert series.iloc[0] == pytest.approx(0.05)
    assert series.iloc[1] == pytest.approx(3.0 / 105.0)
    assert series.iloc[2] == pytest.approx(2.0 / 108.0)


def test_daily_returns_unknown_ticker_none(calc: BenchmarkCalculator) -> None:
    assert calc.get_daily_returns("UNKNOWN", START, END) is None


def test_daily_returns_out_of_range_none(calc: BenchmarkCalculator) -> None:
    assert calc.get_daily_returns("SPY", "2023-01-01", "2023-01-31") is None


# ---------------------------------------------------------------------------
# get_basket_daily_returns
# ---------------------------------------------------------------------------


def test_basket_daily_returns_single_ticker(calc: BenchmarkCalculator) -> None:
    # Basket of one ticker should equal that ticker's daily returns
    basket = calc.get_basket_daily_returns(["SPY"], START, END)
    spy = calc.get_daily_returns("SPY", START, END)
    assert basket is not None and spy is not None
    for a, b in zip(basket.values, spy.values):
        assert a == pytest.approx(b)


def test_basket_daily_returns_two_identical_path_tickers() -> None:
    # Two tickers with the exact same price trajectory → basket == either individual
    bm = BenchmarkCalculator()
    prices = [100.0, 105.0, 108.0, 110.0]
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    bm.load(_make_df(prices, dates), "A")
    bm.load(_make_df(prices, dates), "B")
    basket = bm.get_basket_daily_returns(["A", "B"], START, END)
    single = bm.get_daily_returns("A", START, END)
    assert basket is not None and single is not None
    for a, b in zip(basket.values, single.values):
        assert a == pytest.approx(b, abs=1e-10)


def test_basket_daily_returns_missing_ticker_none(calc: BenchmarkCalculator) -> None:
    assert calc.get_basket_daily_returns(["SPY", "UNKNOWN"], START, END) is None
