from __future__ import annotations

import logging

from .portfolio import Portfolio
from .types import Action, ActionLiteral

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades against a Portfolio with Backtester-identical semantics."""

    def execute_trade(
        self,
        ticker: str,
        action: ActionLiteral,
        quantity: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> float:
        if quantity is None or quantity <= 0:
            return 0

        # Coerce to enum if strings provided
        try:
            action_enum = Action(action) if not isinstance(action, Action) else action
        except Exception:
            logger.warning("Unknown action %r for %s; defaulting to HOLD", action, ticker)
            action_enum = Action.HOLD

        if action_enum == Action.BUY:
            return portfolio.apply_long_buy(ticker, quantity, float(current_price))
        if action_enum == Action.SELL:
            return portfolio.apply_long_sell(ticker, quantity, float(current_price))
        if action_enum == Action.SHORT:
            return portfolio.apply_short_open(ticker, quantity, float(current_price))
        if action_enum == Action.COVER:
            return portfolio.apply_short_cover(ticker, quantity, float(current_price))

        # hold or unknown action
        return 0
