from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Callable, Dict, Sequence

from dateutil.relativedelta import relativedelta
import pandas as pd

from src.llm.request import RunRequest
from src.orchestration.preflight import PipelineContext
from src.tools._yfinance_fundamentals import get_yfinance_news
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_price_data,
    get_sec_filings_as_news,
)

from .benchmarks import BenchmarkCalculator
from .metrics import PerformanceMetricsCalculator
from .output import OutputBuilder
from .portfolio import Portfolio
from .trader import TradeExecutor
from .types import PerformanceMetrics, PortfolioValuePoint
from .valuation import calculate_portfolio_value, compute_exposures

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Coordinates the backtest loop.

    Orchestrates agent decisions, trade execution, valuation, exposures,
    and performance metrics over a date range.
    """

    def __init__(
        self,
        *,
        agent: Callable[..., Any],
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        initial_margin_requirement: float,
        llm_temperature: float | None = None,
        show_reasoning: bool = False,
        use_regime_selection: bool = False,
        use_conviction_weights: bool = False,
        request: RunRequest | None = None,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = float(initial_capital)
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._llm_temperature = llm_temperature
        self._show_reasoning = show_reasoning
        self._use_regime_selection = use_regime_selection
        self._use_conviction_weights = use_conviction_weights
        self._request = request

        self._portfolio = Portfolio(
            tickers=tickers,
            initial_cash=initial_capital,
            margin_requirement=initial_margin_requirement,
        )
        self._executor = TradeExecutor()
        self._perf = PerformanceMetricsCalculator()
        self._results = OutputBuilder(initial_capital=self._initial_capital)

        # Benchmark calculator
        self._benchmark = BenchmarkCalculator()

        self._portfolio_values: list[PortfolioValuePoint] = []
        self._table_rows: list[list] = []
        self._performance_metrics: PerformanceMetrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "long_short_ratio": None,
            "gross_exposure": None,
            "net_exposure": None,
        }
        self._prefetched_prices: dict[str, pd.DataFrame] = {}
        self._signal_log_path: str | None = None
        self._token_summary_data: dict = {}

    def _prefetch_data(self) -> None:
        end_date_dt = datetime.strptime(self._end_date, "%Y-%m-%d")
        start_date_str = (end_date_dt - relativedelta(years=1)).strftime("%Y-%m-%d")

        for ticker in self._tickers:
            self._prefetched_prices[ticker] = get_price_data(ticker, start_date_str, self._end_date)
            get_financial_metrics(ticker, self._end_date, limit=10)
            get_insider_trades(ticker, self._end_date, start_date=self._start_date, limit=1000)
            get_company_news(ticker, self._end_date, start_date=self._start_date, limit=1000)
            get_yfinance_news(ticker, self._end_date, start_date=self._start_date, limit=1000)
            get_sec_filings_as_news(ticker, self._end_date, start_date=self._start_date, limit=100)

        spy_df = get_price_data("SPY", self._start_date, self._end_date)
        self._benchmark.load(spy_df)
        self._spy_prices: pd.DataFrame = spy_df

    def get_signal_log_path(self) -> str | None:
        return self._signal_log_path

    def run_backtest(self) -> PerformanceMetrics:
        self._prefetch_data()

        run_id = f"{'-'.join(self._tickers)}-{self._start_date}-{self._end_date}"

        dates = pd.date_range(self._start_date, self._end_date, freq="B")
        if len(dates) > 0:
            self._portfolio_values = [{"Date": dates[0], "Portfolio Value": self._initial_capital}]
        else:
            self._portfolio_values = []

        with PipelineContext.build(
            agent=self._agent,
            tickers=self._tickers,
            run_id=run_id,
            model_name=self._model_name,
            model_provider=self._model_provider,
            selected_analysts=self._selected_analysts,
            llm_temperature=self._llm_temperature,
            show_reasoning=self._show_reasoning,
            use_regime_selection=self._use_regime_selection,
            use_conviction_weights=self._use_conviction_weights,
            request=self._request,
        ) as ctx:
            self._signal_log_path = ctx.signal_log_path

            for i, current_date in enumerate(dates):
                lookback_start = (current_date - relativedelta(months=1)).strftime("%Y-%m-%d")
                current_date_str = current_date.strftime("%Y-%m-%d")
                if lookback_start == current_date_str:
                    continue

                # Agents form their view using data through current_date (signal bar).
                # Trades fill at the next trading day's open to avoid same-bar lookahead.
                has_next = i + 1 < len(dates)
                next_date_str = dates[i + 1].strftime("%Y-%m-%d") if has_next else None

                try:
                    # Prices for portfolio valuation use the signal bar's close.
                    signal_prices: Dict[str, float] = {}
                    # Prices for trade fills use next day's open (or signal close when no next day).
                    fill_prices: Dict[str, float] = {}
                    missing_data = False
                    for ticker in self._tickers:
                        try:
                            df = self._prefetched_prices.get(ticker)
                            if df is None or df.empty:
                                missing_data = True
                                break
                            sliced = df[df.index <= pd.Timestamp(current_date_str)]
                            if sliced.empty:
                                missing_data = True
                                break
                            signal_prices[ticker] = float(sliced.iloc[-1]["close"])

                            # Fill at next day's open if available, else fall back to signal close
                            if next_date_str is not None:
                                next_sliced = df[df.index == pd.Timestamp(next_date_str)]
                                if not next_sliced.empty and "open" in next_sliced.columns:
                                    fill_prices[ticker] = float(next_sliced.iloc[0]["open"])
                                else:
                                    fill_prices[ticker] = signal_prices[ticker]
                            else:
                                fill_prices[ticker] = signal_prices[ticker]
                        except Exception:
                            logger.warning("Failed to get price data for %s on %s", ticker, current_date_str)
                            missing_data = True
                            break
                    if missing_data:
                        continue
                except Exception:
                    logger.warning("Unexpected error processing date %s; skipping", current_date_str)
                    continue

                agent_output = ctx.run_cycle(
                    date=current_date_str,
                    lookback_start=lookback_start,
                    portfolio=self._portfolio,
                    signal_prices=signal_prices,
                    spy_df=self._spy_prices,
                )
                decisions = agent_output["decisions"]

                executed_trades: Dict[str, float] = {}
                for ticker in self._tickers:
                    d = decisions.get(ticker, {"action": "hold", "quantity": 0})
                    action = d.get("action", "hold")
                    qty = d.get("quantity", 0)
                    executed_qty = self._executor.execute_trade(ticker, action, qty, fill_prices[ticker], self._portfolio)
                    executed_trades[ticker] = executed_qty

                current_prices = signal_prices
                total_value = calculate_portfolio_value(self._portfolio, current_prices)
                exposures = compute_exposures(self._portfolio, current_prices)

                point: PortfolioValuePoint = {
                    "Date": current_date,
                    "Portfolio Value": total_value,
                    "Long Exposure": exposures["Long Exposure"],
                    "Short Exposure": exposures["Short Exposure"],
                    "Gross Exposure": exposures["Gross Exposure"],
                    "Net Exposure": exposures["Net Exposure"],
                    "Long/Short Ratio": exposures["Long/Short Ratio"],
                }
                self._portfolio_values.append(point)

                rows = self._results.build_day_rows(
                    date_str=current_date_str,
                    tickers=self._tickers,
                    agent_output=agent_output,
                    executed_trades=executed_trades,
                    current_prices=current_prices,
                    portfolio=self._portfolio,
                    performance_metrics=self._performance_metrics,
                    total_value=total_value,
                    benchmark_return_pct=self._benchmark.get_return_pct("SPY", self._start_date, current_date_str),
                )
                self._table_rows.extend(rows)
                self._results.print_rows(rows)

                if len(self._portfolio_values) > 3:
                    computed = self._perf.compute_metrics(self._portfolio_values)
                    if computed:
                        self._performance_metrics.update(computed)

        self._token_summary_data = ctx.token_summary()
        return self._performance_metrics

    def get_portfolio_values(self) -> Sequence[PortfolioValuePoint]:
        return list(self._portfolio_values)

    def get_token_summary(self) -> dict:
        """Return aggregated token-usage stats from the completed backtest."""
        return self._token_summary_data
