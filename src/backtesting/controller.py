from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Sequence, cast

from src.llm.request import RunRequest
from src.risk_profiles import RiskProfile
from src.utils.llm import get_token_log, reset_token_log

from .portfolio import Portfolio
from .types import Action, ActionLiteral, AgentDecisions, AgentOutput, PortfolioSnapshot

logger = logging.getLogger(__name__)


class AgentController:
    """Responsible for invoking the trading agent and normalizing outputs."""

    def run_agent(
        self,
        agent: Callable[..., AgentOutput],
        *,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        portfolio: Portfolio | PortfolioSnapshot,
        model_name: str,
        model_provider: str,
        selected_analysts: Sequence[str] | None,
        llm_temperature: float | None = None,
        show_reasoning: bool = False,
        conviction_weights: dict[str, float] | None = None,
        request: RunRequest | None = None,
        risk_profile: RiskProfile | None = None,
        regime: str | None = None,
        recent_trades: dict[str, list[dict]] | None = None,
    ) -> AgentOutput:
        # Ensure we pass a plain snapshot dict to preserve legacy expectations
        if isinstance(portfolio, Portfolio):
            portfolio_payload: PortfolioSnapshot = portfolio.get_snapshot()
        else:
            portfolio_payload = portfolio

        reset_token_log()
        output = agent(
            tickers=list(tickers),
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_payload,
            model_name=model_name,
            model_provider=model_provider,
            selected_analysts=list(selected_analysts) if selected_analysts is not None else None,
            llm_temperature=llm_temperature,
            show_reasoning=show_reasoning,
            conviction_weights=conviction_weights,
            request=request,
            risk_profile=risk_profile,
            regime=regime,
            recent_trades=recent_trades,
        )

        # Normalize outputs to avoid None/missing keys
        decisions_in: Dict[str, Any] = dict(output.get("decisions", {})) if isinstance(output, dict) else {}
        analyst_signals_in: Dict[str, Any] = dict(output.get("analyst_signals", {})) if isinstance(output, dict) else {}

        normalized_decisions: AgentDecisions = {}
        for ticker in tickers:
            d = decisions_in.get(ticker, {})
            action = d.get("action", "hold")
            qty = d.get("quantity", 0)
            # Basic coercions mirroring Backtester expectations
            try:
                qty_val = float(qty)
            except Exception:
                logger.exception("Agent output coercion failed for %s; defaulting to HOLD 0", ticker)
                qty_val = 0.0
            try:
                action = Action(action).value  # validate/coerce
            except Exception:
                logger.exception("Agent output coercion failed for %s; defaulting to HOLD 0", ticker)
                action = Action.HOLD.value
            normalized_decisions[ticker] = {"action": cast(ActionLiteral, action), "quantity": qty_val}

        # Preserve any agent-provided analyst signals without modification
        normalized_output: AgentOutput = {
            "decisions": normalized_decisions,
            "analyst_signals": analyst_signals_in,
            "token_usage": get_token_log(),
            # Full PM decisions preserve confidence and reasoning dropped by normalization
            "pm_decisions": decisions_in,
            "group_signals": dict(output.get("group_signals", {})) if isinstance(output, dict) else {},
            "debate_summaries": dict(output.get("debate_summaries", {})) if isinstance(output, dict) else {},
            "current_prices": dict(output.get("current_prices", {})) if isinstance(output, dict) else {},
        }
        return normalized_output
