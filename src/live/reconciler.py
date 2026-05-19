from __future__ import annotations

import logging
import time

from src.broker import Broker
from src.live.audit_journal import AuditJournal

logger = logging.getLogger(__name__)


def _parse_price(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return None
    return val if val > 0 else None


_TERMINAL_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "done_for_day",
        "stopped",
        "suspended",
        "replaced",
    }
)


class Reconciler:
    """Poll submitted Alpaca orders until they reach a terminal state and journal the result."""

    def __init__(self, broker: Broker, journal: AuditJournal | None = None) -> None:
        self._broker = broker
        self._journal = journal

    def reconcile(
        self,
        order_ids: list[str],
        *,
        timeout_seconds: int = 60,
        poll_interval_seconds: int = 3,
    ) -> dict[str, dict]:
        """Poll each order until terminal or timeout. Returns {order_id: info_dict}.

        info_dict keys: status (fill_status string), filled_qty (float),
        filled_avg_price (float|None), ticker (str).
        On timeout, status is "timeout" and filled_qty reflects the last observed value.
        """
        results: dict[str, dict] = {}
        pending = list(order_ids)
        deadline = time.monotonic() + timeout_seconds
        # Track last-observed state so timeout branch can report real partial-fill values.
        last_observed: dict[str, dict] = {}

        while pending and time.monotonic() < deadline:
            still_pending: list[str] = []
            for order_id in pending:
                try:
                    order = self._broker.get_order(order_id)
                except Exception as exc:
                    logger.warning("[reconciler] get_order(%s) failed: %s", order_id, exc)
                    still_pending.append(order_id)
                    continue

                fill_status = getattr(order.status, "value", str(order.status))
                filled_qty = float(order.filled_qty or "0")
                raw_price = order.filled_avg_price
                filled_avg_price = _parse_price(raw_price)
                ticker = order.symbol or ""

                last_observed[order_id] = {
                    "filled_qty": filled_qty,
                    "filled_avg_price": filled_avg_price,
                    "ticker": ticker,
                }

                if fill_status in _TERMINAL_STATUSES:
                    logger.info(
                        "[reconciler] %s %s terminal: status=%s filled=%.4f",
                        ticker,
                        order_id,
                        fill_status,
                        filled_qty,
                    )
                    info = {
                        "status": fill_status,
                        "filled_qty": filled_qty,
                        "filled_avg_price": filled_avg_price,
                        "ticker": ticker,
                    }
                    results[order_id] = info
                    if self._journal:
                        self._journal.record_reconciliation(
                            order_id=order_id,
                            ticker=ticker,
                            status=fill_status,
                            filled_qty=filled_qty,
                            filled_avg_price=filled_avg_price,
                        )
                else:
                    still_pending.append(order_id)

            if not still_pending:
                pending = []
                break
            pending = still_pending
            if time.monotonic() < deadline:
                time.sleep(poll_interval_seconds)

        # Any orders still pending after the deadline are recorded as "timeout".
        # Use the last-observed poll values so partial fills aren't silently zeroed out.
        for order_id in pending:
            obs = last_observed.get(order_id, {})
            t_qty = obs.get("filled_qty", 0.0)
            t_price = obs.get("filled_avg_price")
            t_ticker = obs.get("ticker", "")
            logger.warning(
                "[reconciler] %s timed out waiting for terminal status (last_filled_qty=%.4f)",
                order_id,
                t_qty,
            )
            info = {"status": "timeout", "filled_qty": t_qty, "filled_avg_price": t_price, "ticker": t_ticker}
            results[order_id] = info
            if self._journal:
                self._journal.record_reconciliation(
                    order_id=order_id,
                    ticker=t_ticker,
                    status="timeout",
                    filled_qty=t_qty,
                    filled_avg_price=t_price,
                )

        return results
