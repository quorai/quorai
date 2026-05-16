from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import pandas as pd

from src.backtesting.controller import AgentController
from src.backtesting.signal_log import SignalLogger
from src.backtesting.types import AgentOutput, PortfolioSnapshot
from src.feedback.loader import load_weights
from src.llm.request import RunRequest
from src.regime import classify_regime, select_analysts_for_regime
from src.risk_profiles import RiskProfile

if TYPE_CHECKING:
    from src.backtesting.portfolio import Portfolio

logger = logging.getLogger(__name__)


class PipelineContext:
    """Stable per-run orchestration state shared by live and backtest flows.

    Encapsulates regime selection, conviction-weights loading, signal logging,
    and AgentController invocation so both flows share a single implementation.

    Callers supply per-cycle data (date, prices, portfolio, spy_df) to
    run_cycle(); all setup state (weights, logger, request) is held here.

    Typical usage::

        with PipelineContext.build(agent=run_quorai, ...) as ctx:
            output = ctx.run_cycle(
                date="2026-01-15",
                lookback_start="2025-12-15",
                portfolio=snapshot,
                signal_prices=prices,
                spy_df=spy_df,
            )
        summary = ctx.token_summary()
    """

    def __init__(
        self,
        *,
        agent: Callable[..., AgentOutput],
        tickers: list[str],
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        llm_temperature: float | None,
        show_reasoning: bool,
        use_regime_selection: bool,
        conviction_weights: dict[str, float],
        signal_logger: SignalLogger | None,
        request: RunRequest | None,
        risk_profile: RiskProfile | None = None,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._llm_temperature = llm_temperature
        self._show_reasoning = show_reasoning
        self._use_regime_selection = use_regime_selection
        self._conviction_weights = conviction_weights
        self._signal_logger = signal_logger
        self._request = request
        self._risk_profile = risk_profile
        self._controller = AgentController()
        self._token_usage: list[dict] = []

    @classmethod
    def build(
        cls,
        *,
        agent: Callable[..., AgentOutput],
        tickers: list[str],
        run_id: str,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None = None,
        llm_temperature: float | None = None,
        show_reasoning: bool = False,
        use_regime_selection: bool = False,
        use_conviction_weights: bool = False,
        enable_signal_log: bool = True,
        request: RunRequest | None = None,
        risk_profile: RiskProfile | None = None,
    ) -> PipelineContext:
        """Create a PipelineContext, loading weights and opening the signal logger."""
        conviction_weights: dict[str, float] = {}
        if use_conviction_weights:
            conviction_weights = load_weights()
            if not conviction_weights:
                logger.warning("--use-conviction-weights set but weights.json is missing or empty; running with uniform weights")
            else:
                logger.info("Loaded conviction weights for %d agents", len(conviction_weights))

        signal_logger: SignalLogger | None = SignalLogger(run_id) if enable_signal_log else None

        return cls(
            agent=agent,
            tickers=tickers,
            model_name=model_name,
            model_provider=model_provider,
            selected_analysts=selected_analysts,
            llm_temperature=llm_temperature,
            show_reasoning=show_reasoning,
            use_regime_selection=use_regime_selection,
            conviction_weights=conviction_weights,
            signal_logger=signal_logger,
            request=request,
            risk_profile=risk_profile,
        )

    def run_cycle(
        self,
        *,
        date: str,
        lookback_start: str,
        portfolio: Portfolio | PortfolioSnapshot,
        signal_prices: dict[str, float],
        spy_df: pd.DataFrame | None = None,
    ) -> AgentOutput:
        """Run one agent cycle and log signals; return the full AgentOutput.

        ``date`` is the as-of date (YYYY-MM-DD); ``lookback_start`` is the
        earliest data date. Regime narrowing uses ``spy_df`` when provided.
        """
        effective_analysts = self._selected_analysts
        if self._use_regime_selection and spy_df is not None and not spy_df.empty:
            regime = classify_regime(spy_df, date)
            regime_analysts = select_analysts_for_regime(regime)
            effective_analysts = regime_analysts if regime_analysts else self._selected_analysts
            logger.debug(
                "Regime %s → %d analysts on %s",
                regime.value,
                len(effective_analysts or []),
                date,
            )

        output = self._controller.run_agent(
            self._agent,
            tickers=self._tickers,
            start_date=lookback_start,
            end_date=date,
            portfolio=portfolio,
            model_name=self._model_name,
            model_provider=self._model_provider,
            selected_analysts=effective_analysts,
            llm_temperature=self._llm_temperature,
            show_reasoning=self._show_reasoning,
            conviction_weights=self._conviction_weights,
            request=self._request,
            risk_profile=self._risk_profile,
        )

        if self._signal_logger is not None:
            self._signal_logger.log_day(date, output.get("analyst_signals", {}), signal_prices)

        self._token_usage.extend(output.get("token_usage") or [])

        return output

    @property
    def signal_log_path(self) -> str | None:
        return self._signal_logger.path if self._signal_logger is not None else None

    def token_summary(self) -> dict:
        """Return aggregated token-usage stats accumulated across all run_cycle calls."""
        log = self._token_usage
        if not log:
            return {}
        return {
            "calls": len(log),
            "input_tokens": sum(e["input_tokens"] for e in log),
            "output_tokens": sum(e["output_tokens"] for e in log),
            "total_tokens": sum(e["input_tokens"] + e["output_tokens"] for e in log),
            "cache_read_tokens": sum(e.get("cache_read_tokens", 0) for e in log),
            "cache_creation_tokens": sum(e.get("cache_creation_tokens", 0) for e in log),
        }

    def close(self) -> None:
        if self._signal_logger is not None:
            self._signal_logger.close()

    def __enter__(self) -> PipelineContext:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
