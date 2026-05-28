from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import pandas as pd

from .types import PerformanceMetrics, PortfolioValuePoint


class PerformanceMetricsCalculator:
    """Concrete metrics calculator like sharpe ratio, sortino ratio, max drawdown, etc."""

    def __init__(
        self,
        *,
        annual_trading_days: int = 252,
        annual_rf_rate: float = 0.0434,
        min_returns_for_ratios: int = 20,
    ) -> None:
        self.annual_trading_days = annual_trading_days
        self.annual_rf_rate = annual_rf_rate
        # Minimum number of daily returns required to emit Sharpe/Sortino/IR.
        # Below this threshold √252 annualisation amplifies tiny sample variance
        # into statistically meaningless extreme values.
        self.min_returns_for_ratios = min_returns_for_ratios

    def update_metrics(self, metrics: PerformanceMetrics, values: Sequence[PortfolioValuePoint]) -> None:
        """Deprecated: mutate provided dict. Kept for backward compatibility."""
        computed = self.compute_metrics(values)
        if not computed:
            return
        metrics.update(computed)

    def compute_metrics(self, values: Sequence[PortfolioValuePoint]) -> PerformanceMetrics:
        import numpy as np
        import pandas as pd

        if not values:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

        df = pd.DataFrame(values)
        if df.empty or "Portfolio Value" not in df:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

        df = df.set_index("Date")
        df["Daily Return"] = df["Portfolio Value"].pct_change()
        clean_returns = df["Daily Return"].dropna()
        if len(clean_returns) < 2:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

        # Suppress risk-adjusted ratios on short windows — √252 annualisation
        # amplifies a tiny, noisy mean/std ratio into statistically meaningless values.
        enough_samples = len(clean_returns) >= self.min_returns_for_ratios

        if enough_samples:
            daily_rf = self.annual_rf_rate / self.annual_trading_days
            excess = clean_returns - daily_rf
            mean_excess = excess.mean()
            std_excess = excess.std()

            if std_excess > 1e-12:
                sharpe: float | None = float(np.sqrt(self.annual_trading_days) * (mean_excess / std_excess))
            else:
                sharpe = 0.0

            # Target downside deviation: sqrt(mean(min(excess, 0)^2)) over all periods
            downside_diff = np.minimum(excess, 0)
            downside_dev = float(np.sqrt(np.mean(downside_diff**2)))
            if downside_dev > 1e-12:
                sortino: float | None = float(np.sqrt(self.annual_trading_days) * (mean_excess / downside_dev))
            else:
                sortino = None if mean_excess > 0 else 0.0
        else:
            sharpe = None
            sortino = None

        rolling_max = df["Portfolio Value"].cummax()
        drawdown = (df["Portfolio Value"] - rolling_max) / rolling_max
        if len(drawdown) > 0:
            min_dd = float(drawdown.min())
            max_drawdown = float(min_dd * 100.0)
            if min_dd < 0:
                max_drawdown_date = drawdown.idxmin().strftime("%Y-%m-%d")
            else:
                max_drawdown_date = None
        else:
            max_drawdown = 0.0
            max_drawdown_date = None

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_date": max_drawdown_date,
        }

    def compute_benchmark_relative(
        self,
        values: Sequence[PortfolioValuePoint],
        benchmark_daily_returns: "pd.Series",
    ) -> dict[str, float | None]:
        """Compute alpha and information ratio vs a benchmark daily-return series.

        Returns {"alpha_pct": float|None, "information_ratio": float|None}.
        Alpha is in percentage points. IR is annualised.
        """
        import numpy as np
        import pandas as pd

        _empty: dict[str, float | None] = {"alpha_pct": None, "information_ratio": None}

        if not values:
            return _empty

        df = pd.DataFrame(values).set_index("Date")
        if "Portfolio Value" not in df or df.empty:
            return _empty

        strategy_daily = df["Portfolio Value"].pct_change().dropna()
        if strategy_daily.empty:
            return _empty

        # Align on date (inner join)
        aligned = pd.concat(
            {"strategy": strategy_daily, "benchmark": benchmark_daily_returns},
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            return _empty

        strategy_total = float((1 + aligned["strategy"]).prod() - 1)
        benchmark_total = float((1 + aligned["benchmark"]).prod() - 1)
        alpha_pct = (strategy_total - benchmark_total) * 100.0

        # IR requires a minimum sample for √252 annualisation to be meaningful.
        # alpha_pct is a raw total-return spread and is reported regardless of window length.
        if len(aligned) >= self.min_returns_for_ratios:
            active = aligned["strategy"] - aligned["benchmark"]
            active_std = float(active.std())
            if active_std > 1e-12:
                ir: float | None = float(np.sqrt(self.annual_trading_days) * active.mean() / active_std)
            else:
                ir = None
        else:
            ir = None

        return {"alpha_pct": alpha_pct, "information_ratio": ir}
