from datetime import datetime, timezone
import logging
from typing import cast

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Asset, Order, Position, TradeAccount
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from src.config import get_settings

logger = logging.getLogger(__name__)


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

        self._client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    def get_account(self) -> TradeAccount:
        """Returns the Alpaca Account object (cash, equity, buying_power, etc.)"""
        return cast(TradeAccount, self._client.get_account())

    def get_positions(self) -> list[Position]:
        """Returns all open positions."""
        return cast(list[Position], self._client.get_all_positions())

    def get_open_orders(self) -> list[Order]:
        """Returns all open (pending) orders."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return cast(list[Order], self._client.get_orders(filter=request))

    def get_asset(self, ticker: str) -> Asset:
        """Returns asset details including tradability and shortability."""
        return cast(Asset, self._client.get_asset(ticker))

    def submit_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        order_type: str = "market",
    ) -> Order:
        """Submits an order. Raises on API error."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = cast(Order, self._client.submit_order(order_data=request))
        logger.info("Submitted %s order: %s %s qty=%.4f", order_type, side.upper(), ticker, qty)
        return order

    def cancel_order(self, order_id: str) -> None:
        """Cancels an open order by ID."""
        self._client.cancel_order_by_id(order_id)
        logger.info("Cancelled order %s", order_id)

    def is_market_open_today(self) -> bool:
        clock = self._client.get_clock()
        today = datetime.now(timezone.utc).date()
        return bool(clock.is_open) or clock.next_open.date() == today
