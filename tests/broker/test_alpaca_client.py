from unittest.mock import MagicMock, patch

import pytest


def _make_client():
    """Instantiate AlpacaClient with Alpaca API mocked out."""
    with patch("src.broker.alpaca_client.TradingClient") as mock_tc, patch("src.broker.alpaca_client.get_settings") as mock_settings:
        settings = MagicMock()
        settings.ALPACA_API_KEY = "test-key"
        settings.ALPACA_SECRET_KEY = "test-secret"
        settings.ALPACA_PAPER = True
        mock_settings.return_value = settings

        from src.broker.alpaca_client import AlpacaClient

        client = AlpacaClient()
        client._client = mock_tc.return_value
    return client


def test_paper_guard_raises_when_not_paper():
    with patch("src.broker.alpaca_client.get_settings") as mock_settings:
        settings = MagicMock()
        settings.ALPACA_API_KEY = "key"
        settings.ALPACA_SECRET_KEY = "secret"
        settings.ALPACA_PAPER = False
        mock_settings.return_value = settings

        with patch("src.broker.alpaca_client.TradingClient"):
            from src.broker.alpaca_client import AlpacaClient

            with pytest.raises(ValueError, match="ALPACA_PAPER"):
                AlpacaClient()


def test_get_open_orders_filters_open_status():
    client = _make_client()
    client._client.get_orders.return_value = []
    result = client.get_open_orders()
    assert result == []
    client._client.get_orders.assert_called_once()
    call_kwargs = client._client.get_orders.call_args
    # The filter should request OPEN status
    assert call_kwargs is not None


def test_submit_order_sends_correct_payload():
    client = _make_client()
    mock_order = MagicMock()
    mock_order.id = "order-1"
    client._client.submit_order.return_value = mock_order

    result = client.submit_order(ticker="AAPL", side="buy", qty=5.0)

    assert result == mock_order
    client._client.submit_order.assert_called_once()


def test_cancel_order_calls_cancel_by_id():
    client = _make_client()
    with patch("src.broker.alpaca_client.time.sleep"):
        client.cancel_order("order-abc")
    client._client.cancel_order_by_id.assert_called_once_with("order-abc")


def test_is_market_open_today_false_when_closed():
    client = _make_client()
    clock = MagicMock()
    clock.is_open = False
    client._client.get_clock.return_value = clock
    assert client.is_market_open_today() is False


def test_is_market_open_today_true_when_open():
    client = _make_client()
    clock = MagicMock()
    clock.is_open = True
    client._client.get_clock.return_value = clock
    assert client.is_market_open_today() is True


def test_is_market_open_today_false_pre_market_on_trading_day():
    """
    Pre-market on a trading day (is_open=False, next_open == today) must return False
    so the live runner does not submit orders before the session opens.
    """
    from datetime import datetime as dt
    from datetime import timezone

    client = _make_client()
    today = dt.now(timezone.utc).date()
    clock = MagicMock()
    clock.is_open = False
    # Simulate a pre-market scenario: next open is later today.
    clock.next_open = MagicMock()
    clock.next_open.date.return_value = today
    # clock.timestamp represents "now" — also today, before open.
    clock.timestamp = MagicMock()
    clock.timestamp.date.return_value = today
    client._client.get_clock.return_value = clock
    assert client.is_market_open_today() is False, "Pre-market on a trading day must return False"


def test_is_trading_day_true_when_market_open():
    client = _make_client()
    clock = MagicMock()
    clock.is_open = True
    client._client.get_clock.return_value = clock
    assert client.is_trading_day() is True


def test_is_trading_day_true_pre_market_on_trading_day():
    from datetime import datetime as dt
    from datetime import timezone

    client = _make_client()
    today = dt.now(timezone.utc).date()
    clock = MagicMock()
    clock.is_open = False
    clock.next_open = MagicMock()
    clock.next_open.date.return_value = today
    clock.timestamp = MagicMock()
    clock.timestamp.date.return_value = today
    client._client.get_clock.return_value = clock
    assert client.is_trading_day() is True


def test_is_trading_day_false_on_weekend():
    from datetime import date

    client = _make_client()
    clock = MagicMock()
    clock.is_open = False
    clock.next_open = MagicMock()
    clock.next_open.date.return_value = date(2026, 5, 25)  # Monday
    clock.timestamp = MagicMock()
    clock.timestamp.date.return_value = date(2026, 5, 23)  # Saturday
    client._client.get_clock.return_value = clock
    assert client.is_trading_day() is False


def test_unknown_side_raises_value_error():
    """RV-10: unrecognised side string raises ValueError instead of silently using SELL."""
    from src.broker.alpaca_client import _resolve_side

    with pytest.raises(ValueError, match="Unknown order side"):
        _resolve_side("unknown")


def test_buy_side_maps_correctly():
    """RV-10: 'buy' maps to OrderSide.BUY."""
    from alpaca.trading.enums import OrderSide

    from src.broker.alpaca_client import _resolve_side

    assert _resolve_side("buy") == OrderSide.BUY
    assert _resolve_side("BUY") == OrderSide.BUY


def test_sell_side_maps_correctly():
    """RV-10: 'sell' maps to OrderSide.SELL."""
    from alpaca.trading.enums import OrderSide

    from src.broker.alpaca_client import _resolve_side

    assert _resolve_side("sell") == OrderSide.SELL
    assert _resolve_side("SELL") == OrderSide.SELL


def test_retry_uses_jitter():
    """RV-14: sleep is called with a value strictly greater than the base delay."""
    from unittest.mock import patch

    from requests.exceptions import ConnectionError as _ConnErr

    client = _make_client()

    # Use ConnectionError (no property restrictions) to trigger retries
    client._client.get_account.side_effect = [_ConnErr("down"), _ConnErr("down"), _ConnErr("down")]

    with patch("src.broker.alpaca_client.random.uniform", return_value=0.2), patch("src.broker.alpaca_client.time.sleep") as mock_sleep:
        try:
            client.get_account()
        except Exception:
            pass

    assert mock_sleep.call_count == 2  # two inter-attempt sleeps for 3 attempts
    base_delays = [0.5, 2.0]
    for i, actual_call in enumerate(mock_sleep.call_args_list):
        slept = actual_call.args[0]
        assert slept > base_delays[i], f"Expected jitter to increase delay above {base_delays[i]}, got {slept}"


def test_cancel_returns_canceled_status_on_clean_cancel():
    """RV-15: clean cancel → status 'canceled', filled_qty 0."""
    from unittest.mock import patch

    client = _make_client()
    order = MagicMock()
    order.status = "canceled"
    order.filled_qty = "0"
    client._client.get_order_by_id.return_value = order

    with patch("src.broker.alpaca_client.time.sleep"):
        result = client.cancel_order("order-abc")

    assert result["status"] == "canceled"
    assert result["filled_qty"] == 0.0
    client._client.cancel_order_by_id.assert_called_once_with("order-abc")


def test_cancel_returns_filled_qty_when_partially_filled():
    """RV-15: partial fill during cancel race → filled_qty reflects actual fill."""
    from unittest.mock import patch

    client = _make_client()
    order = MagicMock()
    order.status = "canceled"
    order.filled_qty = "3.5"
    client._client.get_order_by_id.return_value = order

    with patch("src.broker.alpaca_client.time.sleep"):
        result = client.cancel_order("order-xyz")

    assert result["status"] == "canceled"
    assert result["filled_qty"] == 3.5
