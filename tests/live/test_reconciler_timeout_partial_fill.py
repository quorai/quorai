"""Tests for R38: reconciler timeout preserves last-observed partial-fill state."""

from unittest.mock import MagicMock

import pytest


def _make_order(order_id, status, symbol="AAPL", filled_qty="0", filled_avg_price=None):
    order = MagicMock()
    order.id = order_id
    order.symbol = symbol
    order.status = MagicMock()
    order.status.value = status
    order.filled_qty = filled_qty
    order.filled_avg_price = filled_avg_price
    return order


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.live.reconciler.time.sleep", lambda _: None)


class TestReconcilerTimeoutPartialFill:
    def test_timeout_uses_last_observed_qty(self, monkeypatch):
        """
        When the poll loop times out, the returned record must carry the last
        observed filled_qty/ticker/filled_avg_price, not hardcoded 0.0/""/None.
        """
        from src.live.reconciler import Reconciler

        partial_order = _make_order("ord1", "partially_filled", symbol="AAPL", filled_qty="5", filled_avg_price="150.0")
        broker = MagicMock()
        broker.get_order.return_value = partial_order

        call_count = {"n": 0}

        def fast_clock():
            call_count["n"] += 1
            # Calls 1-2: within deadline so the loop body executes and populates last_observed.
            # Calls 3+: past deadline so the loop exits and the timeout branch fires.
            return 0.0 if call_count["n"] <= 2 else 9999.0

        monkeypatch.setattr("src.live.reconciler.time.monotonic", fast_clock)

        recon = Reconciler(broker=broker)
        results = recon.reconcile(["ord1"], timeout_seconds=1, poll_interval_seconds=0)

        info = results["ord1"]
        assert info["status"] == "timeout"
        assert info["filled_qty"] == pytest.approx(5.0), "timeout branch must use last observed qty, not 0.0"
        assert info["ticker"] == "AAPL", "timeout branch must use last observed ticker, not empty string"
        assert info["filled_avg_price"] == pytest.approx(150.0), "timeout branch must use last observed price"

    def test_timeout_zero_fill_when_never_polled(self, monkeypatch):
        """
        If the order was never successfully polled (get_order always raises),
        the timeout record defaults to zero/empty — that's the correct fallback.
        """
        from src.live.reconciler import Reconciler

        broker = MagicMock()
        broker.get_order.side_effect = RuntimeError("broker down")

        call_count = {"n": 0}

        def fast_clock():
            call_count["n"] += 1
            return 0.0 if call_count["n"] <= 2 else 9999.0

        monkeypatch.setattr("src.live.reconciler.time.monotonic", fast_clock)

        recon = Reconciler(broker=broker)
        results = recon.reconcile(["ord1"], timeout_seconds=1, poll_interval_seconds=0)

        info = results["ord1"]
        assert info["status"] == "timeout"
        assert info["filled_qty"] == pytest.approx(0.0)
        assert info["ticker"] == ""

    def test_journal_records_partial_qty_on_timeout(self, monkeypatch):
        """Journal must receive the real partial-fill values, not zeros."""
        from src.live.reconciler import Reconciler

        partial_order = _make_order("ord2", "partially_filled", symbol="MSFT", filled_qty="3", filled_avg_price="300.0")
        broker = MagicMock()
        broker.get_order.return_value = partial_order

        call_count = {"n": 0}

        def fast_clock():
            call_count["n"] += 1
            return 0.0 if call_count["n"] <= 2 else 9999.0

        monkeypatch.setattr("src.live.reconciler.time.monotonic", fast_clock)

        journal = MagicMock()
        recon = Reconciler(broker=broker, journal=journal)
        recon.reconcile(["ord2"], timeout_seconds=1, poll_interval_seconds=0)

        journal.record_reconciliation.assert_called_once_with(
            order_id="ord2",
            ticker="MSFT",
            status="timeout",
            filled_qty=pytest.approx(3.0),
            filled_avg_price=pytest.approx(300.0),
        )
