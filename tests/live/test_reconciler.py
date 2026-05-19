"""Tests for Reconciler — post-submit order fill polling."""

from unittest.mock import MagicMock

import pytest


def _make_order(order_id: str, status_value: str, symbol: str = "AAPL", filled_qty: str = "0", filled_avg_price: str | None = None) -> MagicMock:
    order = MagicMock()
    order.id = order_id
    order.symbol = symbol
    order.status = MagicMock()
    order.status.value = status_value
    order.filled_qty = filled_qty
    order.filled_avg_price = filled_avg_price
    return order


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.live.reconciler.time.sleep", lambda _: None)


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    """Make time.monotonic advance enough to expire timeouts in test_timeout."""
    _calls = {"n": 0}

    def _mono():
        _calls["n"] += 1
        # Advance very slowly normally, but for the timeout test we patch individually.
        return _calls["n"] * 0.01

    monkeypatch.setattr("src.live.reconciler.time.monotonic", _mono)


def test_filled_immediately():
    from src.live.reconciler import Reconciler

    broker = MagicMock()
    broker.get_order.return_value = _make_order("ord-1", "filled", filled_qty="5", filled_avg_price="150.25")

    recon = Reconciler(broker=broker)
    results = recon.reconcile(["ord-1"])

    assert results["ord-1"]["status"] == "filled"
    assert results["ord-1"]["filled_qty"] == 5.0
    assert results["ord-1"]["filled_avg_price"] == 150.25
    assert results["ord-1"]["ticker"] == "AAPL"
    broker.get_order.assert_called_once_with("ord-1")


def test_partial_then_filled():
    from src.live.reconciler import Reconciler

    broker = MagicMock()
    broker.get_order.side_effect = [
        _make_order("ord-1", "partially_filled", filled_qty="3"),
        _make_order("ord-1", "filled", filled_qty="5"),
    ]

    recon = Reconciler(broker=broker)
    results = recon.reconcile(["ord-1"])

    assert results["ord-1"]["status"] == "filled"
    assert results["ord-1"]["filled_qty"] == 5.0
    assert broker.get_order.call_count == 2


def test_canceled():
    from src.live.reconciler import Reconciler

    broker = MagicMock()
    journal = MagicMock()
    broker.get_order.return_value = _make_order("ord-1", "canceled", filled_qty="0")

    recon = Reconciler(broker=broker, journal=journal)
    results = recon.reconcile(["ord-1"])

    assert results["ord-1"]["status"] == "canceled"
    journal.record_reconciliation.assert_called_once_with(
        order_id="ord-1",
        ticker="AAPL",
        status="canceled",
        filled_qty=0.0,
        filled_avg_price=None,
    )


def test_timeout(monkeypatch):
    from src.live.reconciler import Reconciler

    # Make monotonic tick past timeout_seconds=1 immediately after the second call.
    _calls = {"n": 0}

    def _mono():
        _calls["n"] += 1
        return 0 if _calls["n"] <= 1 else 999.0

    monkeypatch.setattr("src.live.reconciler.time.monotonic", _mono)

    broker = MagicMock()
    broker.get_order.return_value = _make_order("ord-1", "new", filled_qty="0")

    recon = Reconciler(broker=broker)
    results = recon.reconcile(["ord-1"], timeout_seconds=1, poll_interval_seconds=1)

    assert results["ord-1"]["status"] == "timeout"
    assert results["ord-1"]["filled_qty"] == 0.0


def test_journal_optional():
    from src.live.reconciler import Reconciler

    broker = MagicMock()
    broker.get_order.return_value = _make_order("ord-1", "filled", filled_qty="2")

    recon = Reconciler(broker=broker, journal=None)
    results = recon.reconcile(["ord-1"])

    assert results["ord-1"]["status"] == "filled"


def test_multiple_orders():
    from src.live.reconciler import Reconciler

    broker = MagicMock()
    broker.get_order.side_effect = lambda oid: _make_order(oid, "filled", symbol=oid, filled_qty="1")

    recon = Reconciler(broker=broker)
    results = recon.reconcile(["ord-A", "ord-B"])

    assert results["ord-A"]["status"] == "filled"
    assert results["ord-B"]["status"] == "filled"


def test_broker_error_retried_later():
    from src.live.reconciler import Reconciler

    broker = MagicMock()
    broker.get_order.side_effect = [
        Exception("network blip"),
        _make_order("ord-1", "filled", filled_qty="3"),
    ]

    recon = Reconciler(broker=broker)
    results = recon.reconcile(["ord-1"])

    assert results["ord-1"]["status"] == "filled"
    assert broker.get_order.call_count == 2


def test_zero_string_fill_price_excluded():
    """RV-16: raw filled_avg_price of '0' must map to None."""
    from src.live.reconciler import _parse_price

    assert _parse_price("0") is None


def test_zero_decimal_fill_price_excluded():
    """RV-16: '0.00' and '0.0' must also map to None."""
    from src.live.reconciler import _parse_price

    assert _parse_price("0.00") is None
    assert _parse_price("0.0") is None


def test_valid_fill_price_parsed():
    """RV-16: a valid positive price string is parsed to float."""
    from src.live.reconciler import _parse_price

    assert _parse_price("123.45") == 123.45
    assert _parse_price(None) is None
    assert _parse_price("") is None
