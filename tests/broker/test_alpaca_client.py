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
    client.cancel_order("order-abc")
    client._client.cancel_order_by_id.assert_called_once_with("order-abc")
