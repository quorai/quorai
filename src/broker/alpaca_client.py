from datetime import date, datetime
import logging
import random
import time
from typing import cast

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Asset, Order, Position, TradeAccount
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest, MarketOrderRequest
from requests.exceptions import ConnectionError as _ConnError
from requests.exceptions import Timeout as _Timeout

from src.config import get_settings

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (0.5, 2.0)  # delays between 3 attempts
_SIDE_MAP = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
_CANCEL_TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended", "replaced"})


def _resolve_side(side: str) -> OrderSide:
    try:
        return _SIDE_MAP[side.lower()]
    except KeyError:
        raise ValueError(f"Unknown order side: {side!r}. Expected 'buy' or 'sell'.")


def _retry_api_call(fn, *args, **kwargs):
    """Retry fn up to 3 times on transient 5xx or network errors.

    4xx errors (bad request, auth, duplicate client_order_id) are not retried —
    they indicate a caller bug and retrying would just repeat the failure.
    The Alpaca SDK already handles 429 internally; we only need to cover 5xx and
    connection-level failures.
    """
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            return fn(*args, **kwargs)
        except APIError as exc:
            status = exc.status_code
            if status is not None and status < 500:
                raise
            last_exc = exc
        except (_ConnError, _Timeout) as exc:
            last_exc = exc
        if attempt < 3:
            delay = _RETRY_DELAYS[attempt - 1]
            jittered = delay * (1 + random.uniform(0.0, 0.3))
            logger.warning(
                "Alpaca request failed (attempt %d/3), retrying in %.2fs: %s",
                attempt,
                jittered,
                last_exc,
            )
            time.sleep(jittered)
    assert last_exc is not None
    raise last_exc


class AlpacaClient:
    def __init__(self) -> None:
        """Reads ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER from env.
        Raises ValueError if ALPACA_PAPER is not explicitly set to 'true'
        (safety guard against accidental live trading)."""
        settings = get_settings()
        api_key = settings.ALPACA_API_KEY
        secret_key = settings.ALPACA_SECRET_KEY

        if not api_key:
            raise ValueError("ALPACA_API_KEY environment variable is not set")
        if not secret_key:
            raise ValueError("ALPACA_SECRET_KEY environment variable is not set")
        if not settings.ALPACA_PAPER:
            raise ValueError("ALPACA_PAPER must be set to 'true' to prevent accidental live trading")

        self._client = TradingClient(api_key=api_key, secret_key=secret_key, paper=settings.ALPACA_PAPER)

    def get_account(self) -> TradeAccount:
        """Returns the Alpaca Account object (cash, equity, buying_power, etc.)"""
        return cast(TradeAccount, _retry_api_call(self._client.get_account))

    def get_positions(self) -> list[Position]:
        """Returns all open positions."""
        return cast(list[Position], _retry_api_call(self._client.get_all_positions))

    def get_open_orders(self) -> list[Order]:
        """Returns all open (pending) orders."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return cast(list[Order], _retry_api_call(self._client.get_orders, filter=request))

    def get_asset(self, ticker: str) -> Asset:
        """Returns asset details including tradability and shortability."""
        return cast(Asset, _retry_api_call(self._client.get_asset, ticker))

    def get_order(self, order_id: str) -> Order:
        """Returns a single order by its Alpaca order ID."""
        return cast(Order, _retry_api_call(self._client.get_order_by_id, order_id))

    def list_orders(
        self,
        *,
        status: str = "all",
        after: str | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Returns orders filtered by status. `after` is an ISO-8601 timestamp string."""
        qs = QueryOrderStatus(status)
        after_dt = datetime.fromisoformat(after) if after else None
        request = GetOrdersRequest(status=qs, after=after_dt, limit=limit)
        return cast(list[Order], _retry_api_call(self._client.get_orders, filter=request))

    def submit_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        order_type: str = "market",
        client_order_id: str | None = None,
    ) -> Order:
        """Submits an order. Raises on API error.

        client_order_id, if provided, is passed to Alpaca so that an identical
        re-submission is rejected by the broker rather than double-filled.
        """
        order_side = _resolve_side(side)
        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = cast(Order, _retry_api_call(self._client.submit_order, order_data=request))
        logger.info("Submitted %s order: %s %s qty=%.4f", order_type, side.upper(), ticker, qty)
        return order

    def cancel_order(self, order_id: str) -> dict:
        """Cancels an open order and waits for a terminal status.

        Returns {"status": str, "filled_qty": float} reflecting the post-cancel
        state. A brief poll loop handles the partial-fill race where the broker
        acknowledges the cancel (HTTP 200) while residual quantity continues to fill.
        """
        _retry_api_call(self._client.cancel_order_by_id, order_id)
        order = None
        for _ in range(3):
            time.sleep(0.5)
            order = cast(Order, _retry_api_call(self._client.get_order_by_id, order_id))
            if str(order.status) in _CANCEL_TERMINAL_STATUSES:
                break
        if order is None:
            return {"status": "unknown", "filled_qty": 0.0}
        status = str(order.status)
        filled_qty = float(order.filled_qty or 0)
        logger.info("Cancelled order %s → status=%s filled_qty=%.4f", order_id, status, filled_qty)
        return {"status": status, "filled_qty": filled_qty}

    def is_market_open_today(self) -> bool:
        clock = _retry_api_call(self._client.get_clock)
        return bool(clock.is_open)

    def get_sod_equity(self, date: date) -> float:
        """Return the prior-close equity for `date`, used as the SOD loss-limit baseline.

        Alpaca's base_value for a 1-day portfolio history query is the previous trading
        day's close — the correct anchor for the daily loss limit (no intraday P&L yet).
        """
        req = GetPortfolioHistoryRequest(period="1D", timeframe="1H", date_end=date)
        history = _retry_api_call(self._client.get_portfolio_history, history_filter=req)
        if history.base_value is None:
            raise RuntimeError(f"Alpaca returned no base_value for portfolio history on {date} — cannot establish SOD equity; create logs/live/sod-equity/sod-equity-{date}.json manually")
        return float(history.base_value)
