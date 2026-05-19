import logging

from src.config import Settings, get_settings
from src.live.audit_journal import AuditJournal

logger = logging.getLogger(__name__)


class RiskGate:
    def __init__(self, settings: Settings, journal: AuditJournal) -> None:
        self._settings = settings
        self._journal = journal

    def check(
        self,
        *,
        ticker: str,
        action: str,
        side: str,
        qty: float,
        price: float,
        account_equity: float,
        sod_equity: float,
        current_long: float = 0.0,
        current_short: float = 0.0,
    ) -> tuple[bool, str]:
        """Returns (allowed, reason). Records rejection to journal if not allowed."""
        reason = self._evaluate(
            action=action,
            qty=qty,
            price=price,
            account_equity=account_equity,
            sod_equity=sod_equity,
            current_long=current_long,
            current_short=current_short,
        )
        if reason:
            logger.warning("[risk_gate] Rejecting %s %s: %s", action, ticker, reason)
            self._journal.record(
                ticker=ticker,
                action=action,
                qty=qty,
                side=side,
                status="rejected",
                reason=reason,
            )
            return False, reason
        return True, ""

    def _evaluate(
        self,
        *,
        action: str,
        qty: float,
        price: float,
        account_equity: float,
        sod_equity: float,
        current_long: float,
        current_short: float,
    ) -> str:
        s = self._settings
        if s.KILL_SWITCH or get_settings().KILL_SWITCH:
            return "kill_switch_active"
        if price <= 0:
            return "missing_price"
        # Closing trades reduce exposure and bypass the notional cap.
        # Only the portion that opens new exposure (e.g. a sell that exceeds the long) is capped.
        if action == "sell":
            closing_qty = min(qty, current_long)
        elif action == "cover":
            closing_qty = min(qty, current_short)
        else:
            closing_qty = 0.0
        opening_qty = max(0.0, qty - closing_qty)
        if opening_qty * price > s.MAX_ORDER_NOTIONAL:
            return "notional_exceeded"
        if qty > s.MAX_ORDER_QTY:
            return "qty_exceeded"
        if sod_equity > 0 and (account_equity / sod_equity - 1) <= -s.DAILY_LOSS_LIMIT_PCT:
            return "daily_loss_limit"
        return ""
