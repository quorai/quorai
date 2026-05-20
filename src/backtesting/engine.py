from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Callable, Dict, Sequence

from dateutil.relativedelta import relativedelta
import pandas as pd

from src.agents._data_bundle import ALL_LINE_ITEMS
from src.data.backtest_store import _TickerStore, get_backtest_store
from src.llm.request import RunRequest
from src.orchestration.preflight import PipelineContext
from src.risk_profiles import RiskProfile
from src.tools._yfinance_fundamentals import get_yfinance_news
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_price_data,
    get_prices,
    get_sec_filings_as_news,
    prices_to_df,
    search_line_items,
)
from src.utils.progress import progress

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
        risk_profile: RiskProfile | None = None,
        run_label: str = "",
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = float(initial_capital)
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._llm_temperature = llm_temperature if llm_temperature is not None else 0.0
        self._show_reasoning = show_reasoning
        self._use_regime_selection = use_regime_selection
        self._use_conviction_weights = use_conviction_weights
        self._request = request
        self._risk_profile = risk_profile
        self._run_label = run_label

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
        """Bulk-fetch all ticker data for the backtest window and install the in-memory store.

        After this runs, every api.py fetcher (get_financial_metrics, search_line_items,
        get_insider_trades, get_company_news, get_yfinance_news, get_sec_filings_as_news,
        get_prices, get_market_cap) will serve per-day agent requests from memory with
        zero HTTP calls.
        """
        end_date_dt = datetime.strptime(self._end_date, "%Y-%m-%d")
        # Prices need 1 year of lookback for valuation and technical analysis
        price_start_str = (end_date_dt - relativedelta(years=1)).strftime("%Y-%m-%d")

        store_data: dict[str, _TickerStore] = {}

        for ticker in self._tickers:
            logger.info("Prefetching data for %s", ticker)

            # Prices (full year window; also used by the day-loop valuation slice).
            # yfinance fetches with auto_adjust=True (see api.py:get_prices), so all
            # prices are already split/dividend-adjusted.  Portfolio share counts track
            # unadjusted quantities; because the adjustment is baked into the price
            # series rather than the share count, no per-position correction is needed.
            prices = get_prices(ticker, price_start_str, self._end_date)
            self._prefetched_prices[ticker] = prices_to_df(prices)

            # Financial metrics — pre-fetch both "ttm" and "annual" at full limit
            # so every agent's metrics_period / metrics_limit combination is covered.
            metrics_ttm = get_financial_metrics(ticker, self._end_date, period="ttm", limit=10)
            metrics_annual = get_financial_metrics(ticker, self._end_date, period="annual", limit=10)

            # Line items — superset of all fields used by any agent; period="annual" (the
            # default for all agents).  limit=20 covers 5× the max any agent requests (5)
            # and provides enough history for multi-year trend analysis.
            line_items_annual = search_line_items(ticker, ALL_LINE_ITEMS, self._end_date, period="annual", limit=20)

            # Insider trades, news — full backtest window
            insider_trades = get_insider_trades(ticker, self._end_date, start_date=self._start_date, limit=2000)
            company_news = get_company_news(ticker, self._end_date, start_date=self._start_date, limit=2000)
            yf_news = get_yfinance_news(ticker, self._end_date, start_date=self._start_date, limit=200)
            sec_news = get_sec_filings_as_news(ticker, self._end_date, start_date=self._start_date, limit=200)

            # Derive shares_outstanding from the most recent annual line-item period
            shares: float | None = None
            if line_items_annual:
                shares = getattr(line_items_annual[0], "outstanding_shares", None)

            store_data[ticker] = _TickerStore(
                prices=prices,
                financial_metrics={"ttm": metrics_ttm, "annual": metrics_annual},
                line_items={"annual": line_items_annual},
                insider_trades=insider_trades,
                company_news=company_news,
                yfinance_news=yf_news,
                sec_news=sec_news,
                shares_outstanding=shares,
            )

        get_backtest_store().install(store_data, (self._start_date, self._end_date))

        spy_df = get_price_data("SPY", self._start_date, self._end_date)
        self._benchmark.load(spy_df, "SPY")
        for ticker in self._tickers:
            self._benchmark.load(self._prefetched_prices[ticker], ticker)
        self._spy_prices: pd.DataFrame = spy_df

    def get_signal_log_path(self) -> str | None:
        return self._signal_log_path

    def get_benchmark(self) -> BenchmarkCalculator:
        return self._benchmark

    @property
    def run_id(self) -> str:
        base = f"{'-'.join(self._tickers)}-{self._start_date}-{self._end_date}"
        return f"{base}-{self._run_label}" if self._run_label else base

    def run_backtest(self) -> PerformanceMetrics:
        if self._risk_profile is not None:
            logger.info("Risk profile: %s (base_limit=%.2f, notional_cap=$%.0f)", self._risk_profile.name, self._risk_profile.base_limit, self._risk_profile.max_order_notional)

        self._prefetch_data()

        run_id = self.run_id

        valid_trading_days: set[pd.Timestamp] = set()
        for df in self._prefetched_prices.values():
            valid_trading_days.update(df.index)

        business_days = pd.date_range(self._start_date, self._end_date, freq="B")
        if valid_trading_days:
            dates = [d for d in business_days if d in valid_trading_days]
        else:
            logger.warning("No prefetched price data found; falling back to business-day calendar")
            dates = list(business_days)
        self._portfolio_values = []

        try:
            with (
                PipelineContext.build(
                    agent=self._agent,
                    tickers=self._tickers,
                    run_id=run_id,
                    mode="backtest",
                    model_name=self._model_name,
                    model_provider=self._model_provider,
                    selected_analysts=self._selected_analysts,
                    llm_temperature=self._llm_temperature,
                    show_reasoning=self._show_reasoning,
                    use_regime_selection=self._use_regime_selection,
                    use_conviction_weights=self._use_conviction_weights,
                    request=self._request,
                    risk_profile=self._risk_profile,
                ) as ctx,
                progress.display(),
            ):
                self._signal_log_path = ctx.signal_log_path
                from src.orchestration.price_feed import BacktestPriceFeed  # lazy: avoids circular import via package __init__

                price_feed = BacktestPriceFeed(self._prefetched_prices, self._spy_prices)

                for i, current_date in enumerate(dates):
                    # 12-month lookback to feed longest indicator (252-day vol, 126-day Hurst, 6-month momentum + buffer).
                    lookback_start = (current_date - relativedelta(months=12)).strftime("%Y-%m-%d")
                    current_date_str = current_date.strftime("%Y-%m-%d")

                    logger.info("Backtest day %d/%d (%s)", i + 1, len(dates), current_date_str)
                    progress.print(f"Day {i + 1}/{len(dates)} — {current_date_str}", style="bold cyan")

                    # Agents form their view using data through current_date (signal bar).
                    # Trades fill at the first actual trading day's open after current_date to
                    # avoid same-bar lookahead, even across NYSE holidays (which freq="B" includes
                    # as iteration dates but which have no bar in yfinance price data).

                    try:
                        # Prices for portfolio valuation use the signal bar's close.
                        signal_prices: Dict[str, float] = price_feed.get_signal_prices(self._tickers, current_date_str, lookback_start)
                        if not signal_prices:
                            continue
                        # Prices for trade fills use the next available bar's open.
                        # Using the first bar strictly after current_date avoids the
                        # holiday fallback that occurred with exact next_date_str matching.
                        fill_prices: Dict[str, float] = {}
                        for ticker in signal_prices:
                            try:
                                df = self._prefetched_prices.get(ticker)
                                # Fill at the open of the first available bar after current_date.
                                # Exact-match on the calendar's "next business day" silently falls
                                # back to same-bar close on NYSE holidays (no bar exists that day).
                                future_bars = df[df.index > pd.Timestamp(current_date_str)]
                                if not future_bars.empty and "open" in future_bars.columns:
                                    fill_prices[ticker] = float(future_bars.iloc[0]["open"])
                                else:
                                    fill_prices[ticker] = signal_prices[ticker]
                            except Exception:
                                fill_prices[ticker] = signal_prices[ticker]
                    except Exception:
                        logger.warning("Unexpected error processing date %s; skipping", current_date_str)
                        continue

                    agent_output = ctx.run_cycle(
                        date=current_date_str,
                        lookback_start=lookback_start,
                        portfolio=self._portfolio,
                        signal_prices=signal_prices,
                        spy_df=price_feed.get_spy_df(lookback_start, current_date_str),
                    )
                    decisions = agent_output["decisions"]

                    # Compute pre-trade NAV at today's close before mutating the portfolio.
                    # This ensures the equity-curve point for current_date reflects only
                    # positions that were already held — it does not include phantom PnL
                    # from the spread between today's close and tomorrow's fill open.
                    current_prices = signal_prices
                    total_value = calculate_portfolio_value(self._portfolio, current_prices)
                    exposures = compute_exposures(self._portfolio, current_prices)

                    executed_trades: Dict[str, float] = {}
                    for ticker in self._tickers:
                        if ticker not in fill_prices:
                            executed_trades[ticker] = 0
                            continue
                        d = decisions.get(ticker, {"action": "hold", "quantity": 0})
                        action = d.get("action", "hold")
                        qty = d.get("quantity", 0)
                        executed_qty = self._executor.execute_trade(ticker, action, qty, fill_prices[ticker], self._portfolio)
                        executed_trades[ticker] = executed_qty

                    ctx.finalize_cycle(
                        date=current_date_str,
                        fill_prices=fill_prices,
                        executed_trades=executed_trades,
                        portfolio_after=self._portfolio,
                    )

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

                computed = self._perf.compute_metrics(self._portfolio_values)
                if computed:
                    self._performance_metrics.update(computed)

                self._token_summary_data = ctx.token_summary()
        finally:
            get_backtest_store().uninstall()

        return self._performance_metrics

    def get_portfolio_values(self) -> Sequence[PortfolioValuePoint]:
        return list(self._portfolio_values)

    def get_token_summary(self) -> dict:
        """Return aggregated token-usage stats from the completed backtest."""
        return self._token_summary_data
