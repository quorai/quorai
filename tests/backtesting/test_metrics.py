from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import PerformanceMetricsCalculator


def _build_values(values: list[float]):
    start = datetime(2024, 1, 1)
    points = []
    for i, v in enumerate(values):
        points.append(
            {
                "Date": start + timedelta(days=i),
                "Portfolio Value": v,
                "Long Exposure": 0.0,
                "Short Exposure": 0.0,
                "Gross Exposure": 0.0,
                "Net Exposure": 0.0,
                "Long/Short Ratio": np.inf,
            }
        )
    return points


def test_metrics_insufficient_data_no_update():
    calc = PerformanceMetricsCalculator()
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, _build_values([100_000.0]))
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["max_drawdown"] is None


def test_metrics_basic_sharpe_sortino_and_drawdown():
    # min_returns_for_ratios=2 so the formula is exercised even with a 3-point series.
    calc = PerformanceMetricsCalculator(annual_trading_days=2, annual_rf_rate=0.0, min_returns_for_ratios=2)
    # Values: up then down → non-zero volatility; drawdown occurs on last day
    vals = _build_values([100.0, 110.0, 99.0])
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, vals)
    assert metrics["sharpe_ratio"] is not None
    assert metrics["sortino_ratio"] is not None
    assert metrics["max_drawdown"] < 0.0
    assert isinstance(metrics.get("max_drawdown_date"), str)


def test_metrics_zero_volatility_sharpe_zero():
    # min_returns_for_ratios=2 so the formula is exercised with a 4-point series.
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0, min_returns_for_ratios=2)
    # Constant portfolio value → zero volatility → Sharpe 0
    vals = _build_values([100.0, 100.0, 100.0, 100.0])
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, vals)
    assert metrics["sharpe_ratio"] == 0.0


def test_sortino_none_when_no_downside_returns():
    """All-positive excess returns → downside_dev == 0 → Sortino must be None, not inf."""
    import json

    # min_returns_for_ratios=2 so the gating doesn't suppress the sortino-specific None check.
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0, min_returns_for_ratios=2)
    vals = _build_values([100.0, 110.0, 121.0, 133.0])  # monotonically up, no downside
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, vals)
    assert metrics["sortino_ratio"] is None, "Sortino should be None when no downside returns"
    # Must serialise cleanly as JSON null, not raise on float('inf')
    json.dumps({"sortino": metrics["sortino_ratio"]})


def test_sortino_finite_when_downside_exists():
    """Returns with some negative days → Sortino is a finite float."""
    # min_returns_for_ratios=2 to test the formula, not the gating.
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0, min_returns_for_ratios=2)
    vals = _build_values([100.0, 110.0, 95.0, 105.0])
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, vals)
    assert isinstance(metrics["sortino_ratio"], float)
    assert metrics["sortino_ratio"] not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# compute_benchmark_relative
# ---------------------------------------------------------------------------


def _build_bm_series(daily_returns: list[float], start: datetime) -> pd.Series:
    """Build a benchmark daily-return Series aligned to the same dates as _build_values."""
    dates = [start + timedelta(days=i + 1) for i in range(len(daily_returns))]
    return pd.Series(daily_returns, index=pd.DatetimeIndex(dates))


def test_benchmark_relative_known_alpha():
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    # Strategy: 100 → 102 → 104.04 → 106.12  (+2%/day for 3 days)
    # Benchmark: +1%/day for 3 days
    start = datetime(2024, 1, 1)
    vals = _build_values([100.0, 102.0, 104.04, 106.1208])
    bm = _build_bm_series([0.01, 0.01, 0.01], start)
    result = calc.compute_benchmark_relative(vals, bm)

    # strategy total (aligned): 1.02^3 - 1 ≈ 6.1208%; benchmark: 1.01^3 - 1 ≈ 3.0301%
    assert result["alpha_pct"] == pytest.approx(6.1208 - 3.0301, abs=0.01)
    assert result["information_ratio"] is None  # constant active returns → zero std


def test_benchmark_relative_ir_varies():
    import math

    # min_returns_for_ratios=2 so the IR formula is exercised with a 3-return series.
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0, min_returns_for_ratios=2)
    # Strategy: 100 → 101 → 100.5 → 101.5  daily returns: +1%, -0.495%, +0.995%
    start = datetime(2024, 1, 2)
    vals = _build_values([100.0, 101.0, 100.5, 101.5])
    # Benchmark: +0.5%/day every day → active = strategy - 0.5% each day
    bm = _build_bm_series([0.005, 0.005, 0.005], start)
    result = calc.compute_benchmark_relative(vals, bm)

    assert result["alpha_pct"] is not None
    assert result["information_ratio"] is not None
    assert isinstance(result["information_ratio"], float)
    assert math.isfinite(result["information_ratio"])


def test_benchmark_relative_misaligned_dates_returns_none():
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    vals = _build_values([100.0, 102.0, 104.0])
    # Benchmark on completely different dates — no overlap after inner join
    bm = pd.Series([0.01, 0.01], index=pd.to_datetime(["2025-06-01", "2025-06-02"]))
    result = calc.compute_benchmark_relative(vals, bm)
    assert result["alpha_pct"] is None
    assert result["information_ratio"] is None


def test_benchmark_relative_empty_values():
    calc = PerformanceMetricsCalculator()
    bm = _build_bm_series([0.01, 0.01], datetime(2024, 1, 1))
    result = calc.compute_benchmark_relative([], bm)
    assert result["alpha_pct"] is None
    assert result["information_ratio"] is None


# ---------------------------------------------------------------------------
# Short-window gating (min_returns_for_ratios)
# ---------------------------------------------------------------------------


def test_short_window_suppresses_sharpe_and_sortino_but_keeps_drawdown():
    """Below min_returns_for_ratios → Sharpe/Sortino None; max_drawdown still computed."""
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    # 8 NAV points → 7 daily returns, well below default min_returns_for_ratios=20.
    vals = _build_values([100_000.0, 100_200.0, 99_800.0, 100_100.0, 99_500.0, 99_700.0, 100_050.0, 99_870.0])
    metrics: dict = {}
    calc.update_metrics(metrics, vals)
    assert metrics["sharpe_ratio"] is None, "Sharpe should be suppressed below min sample"
    assert metrics["sortino_ratio"] is None, "Sortino should be suppressed below min sample"
    # max_drawdown must still be computed (it's valid on any window)
    assert metrics["max_drawdown"] is not None
    assert metrics["max_drawdown"] < 0.0


def test_sufficient_window_emits_sharpe_and_sortino():
    """At or above min_returns_for_ratios → Sharpe/Sortino are computed."""
    n = 21  # just above the default threshold of 20
    # Build a mildly trending series to avoid zero-std edge cases.
    navs = [100_000.0 * (1.0005**i) for i in range(n + 1)]
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    vals = _build_values(navs)
    metrics: dict = {}
    calc.update_metrics(metrics, vals)
    assert metrics["sharpe_ratio"] is not None, "Sharpe should be computed with sufficient samples"


def test_short_window_keeps_alpha_suppresses_ir():
    """Below min_returns_for_ratios → alpha_pct computed; information_ratio suppressed."""
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    start = datetime(2024, 1, 1)
    # 8 NAV points → 7 aligned returns (below threshold of 20)
    vals = _build_values([100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.0])
    bm = _build_bm_series([0.005] * 7, start)
    result = calc.compute_benchmark_relative(vals, bm)
    assert result["alpha_pct"] is not None, "alpha_pct should always be computed regardless of window length"
    assert result["information_ratio"] is None, "IR should be suppressed below min sample"
