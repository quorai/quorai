from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from src.backtesting.types import AgentDecisions
from src.broker import Broker
from src.live.audit_journal import AuditJournal
from src.live.risk_gate import RiskGate
from src.utils.tz import now_ny

if TYPE_CHECKING:
    from src.live.idempotency_guard import IdempotencyGuard

logger = logging.getLogger(__name__)

_BUY_ACTIONS = {"buy", "cover"}
_SELL_ACTIONS = {"sell", "short"}

_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended", "replaced"}


class LiveExecutor:
    def __init__(
        self,
        broker: Broker,
        risk_gate: RiskGate | None = None,
        journal: AuditJournal | None = None,
        sod_equity: float = 0.0,
        idempotency_guard: IdempotencyGuard | None = None,
    ) -> None:
        self._broker = broker
        self._risk_gate = risk_gate
        self._journal = journal
        self._sod_equity = sod_equity
        self._idempotency_guard = idempotency_guard
        # Populated by execute_decisions(); maps ticker → broker order ID for submitted orders.
        self.submitted_orders: dict[str, str] = {}

    def execute_decisions(
        self,
        decisions: AgentDecisions,
        dry_run: bool = False,
        current_prices: dict[str, float] | None = None,
    ) -> dict[str, str]:
        """Execute decisions. Returns {ticker: "submitted"|"skipped"|"error: ..."}.
        If dry_run=True, logs what would be submitted but does not call the API.
        Submitted broker order IDs are stored in self.submitted_orders after the call."""
        results: dict[str, str] = {}
        prices = current_prices or {}
        self.submitted_orders = {}

        if not dry_run and self._idempotency_guard is not None:
            allowed, reason = self._idempotency_guard.check()
            if not allowed:
                logger.warning("[executor] idempotency guard blocked run: %s", reason)
                return {ticker: f"skipped (idempotency: {reason})" for ticker in decisions}

        pending: set[tuple[str, str]] = set()
        if not dry_run:
            for order in self._broker.get_open_orders():
                if order.symbol is not None and order.side is not None:
                    pending.add((order.symbol, order.side.value))

        # Pre-fetch positions for risk-gate closing-trade exemption + cover qty clamping.
        current_longs: dict[str, float] = {}
        current_shorts: dict[str, float] = {}
        has_cover = any(d.get("action") == "cover" for d in decisions.values())
        needs_positions = (self._risk_gate is not None or has_cover) and not dry_run
        if needs_positions:
            for pos in self._broker.get_positions():
                pq = float(pos.qty)
                if pq > 0:
                    current_longs[pos.symbol] = pq
                elif pq < 0:
                    current_shorts[pos.symbol] = abs(pq)

        account_equity = 0.0
        if self._risk_gate and not dry_run:
            account = self._broker.get_account()
            account_equity = float(account.equity or "0")

        for ticker, decision in decisions.items():
            action = decision.get("action", "hold")
            qty = round(float(decision.get("quantity", 0)), 3)

            if action == "hold" or qty == 0:
                results[ticker] = "skipped"
                continue

            # Alpaca requires whole shares for short/cover; floor with 0.6 threshold (0.5→0, 0.6→1)
            if action in {"short", "cover"}:
                qty = float(math.floor(qty + 0.4))
                if qty == 0:
                    logger.warning("[executor] %s %s qty rounds to 0, skipping", ticker, action)
                    results[ticker] = "skipped (qty rounds to 0)"
                    continue

            if action in _BUY_ACTIONS:
                side = "buy"
                if action == "cover" and not dry_run:
                    actual_short = current_shorts.get(ticker, 0.0)
                    if actual_short == 0.0:
                        logger.warning("[executor] %s cover requested but no short position, skipping", ticker)
                        if self._journal:
                            self._journal.record(
                                ticker=ticker,
                                action=action,
                                qty=qty,
                                side="buy",
                                status="skipped",
                                reason="cover: no short to close",
                            )
                        results[ticker] = "skipped (cover: no short)"
                        continue
                    if qty > actual_short:
                        logger.warning(
                            "[executor] %s cover qty=%.3f clamped to actual short=%.3f",
                            ticker,
                            qty,
                            actual_short,
                        )
                        qty = float(math.floor(actual_short))
                        if qty == 0:
                            logger.warning("[executor] %s cover qty rounds to 0 after floor, skipping", ticker)
                            results[ticker] = "skipped (cover qty rounds to 0)"
                            continue
            elif action in _SELL_ACTIONS:
                side = "sell"
                if action == "short":
                    if not dry_run:
                        asset = self._broker.get_asset(ticker)
                        if not asset.shortable:
                            logger.warning("[executor] %s is not shortable, skipping", ticker)
                            results[ticker] = "skipped (not shortable)"
                            continue
            else:
                results[ticker] = f"error: unknown action '{action}'"
                continue

            if (ticker, side) in pending:
                logger.warning("[executor] %s already has an open %s order, skipping", ticker, side)
                results[ticker] = "skipped (open order exists)"
                continue

            # Risk gate check
            if self._risk_gate and not dry_run:
                price = prices.get(ticker, 0.0)
                allowed, reason = self._risk_gate.check(
                    ticker=ticker,
                    action=action,
                    side=side,
                    qty=qty,
                    price=price,
                    account_equity=account_equity,
                    sod_equity=self._sod_equity,
                    current_long=current_longs.get(ticker, 0.0),
                    current_short=current_shorts.get(ticker, 0.0),
                )
                if not allowed:
                    logger.warning("[executor] %s rejected by risk gate: %s", ticker, reason)
                    results[ticker] = f"rejected: {reason}"
                    continue

            logger.info("[executor] %s %.3f %s", side.upper(), qty, ticker)

            if dry_run:
                results[ticker] = "submitted"
                continue

            # Deterministic client_order_id lets the broker reject a duplicate
            # submission if we retry after a crash — the journal "pending" row
            # written below provides a local crash-recovery audit trail.
            date_prefix = now_ny().strftime("%Y-%m-%d")
            client_order_id = f"{date_prefix}-{ticker}-{action}"

            if self._journal:
                self._journal.record(
                    ticker=ticker,
                    action=action,
                    qty=qty,
                    side=side,
                    status="pending",
                    order_id=client_order_id,
                )

            try:
                order = self._broker.submit_order(ticker=ticker, side=side, qty=qty, client_order_id=client_order_id)
                broker_order_id = str(order.id)
                self.submitted_orders[ticker] = broker_order_id
                if self._journal:
                    self._journal.record(
                        ticker=ticker,
                        action=action,
                        qty=qty,
                        side=side,
                        status="submitted",
                        order_id=broker_order_id,
                    )
                results[ticker] = "submitted"
            except Exception as exc:
                logger.error("[executor] Failed to submit order for %s: %s", ticker, exc)
                if self._journal:
                    self._journal.record(
                        ticker=ticker,
                        action=action,
                        qty=qty,
                        side=side,
                        status="error",
                        reason=str(exc),
                    )
                results[ticker] = f"error: {exc}"

        return results
