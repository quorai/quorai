"""Tests for R44: executor re-floors cover qty after clamping to actual short."""

from unittest.mock import MagicMock

from src.live.executor import LiveExecutor


def _make_broker(short_qty: float):
    broker = MagicMock()
    broker.get_open_orders.return_value = []
    account = MagicMock()
    account.equity = 100_000.0
    broker.get_account.return_value = account
    pos = MagicMock()
    pos.symbol = "AAPL"
    pos.qty = str(-short_qty)  # negative qty = short position
    broker.get_positions.return_value = [pos]
    submitted = MagicMock()
    submitted.id = "order-xyz"
    broker.submit_order.return_value = submitted
    return broker


class TestExecutorCoverFloor:
    def test_cover_qty_floored_when_actual_short_fractional(self):
        """
        actual_short=10.5, requested qty=15 → clamped to 10.5 → floored to 10.
        Alpaca rejects fractional cover orders; the floor must happen after the clamp.
        """
        broker = _make_broker(short_qty=10.5)
        executor = LiveExecutor(broker=broker)
        results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 15}})

        assert results["AAPL"] not in ("skipped", "skipped (cover: no short)", "skipped (cover qty rounds to 0)")
        call_kwargs = broker.submit_order.call_args.kwargs
        assert call_kwargs["qty"] == 10.0, f"Cover qty should be floored to 10, got {call_kwargs['qty']}"

    def test_cover_qty_skipped_when_floor_is_zero(self):
        """
        actual_short=0.4 → requested qty=5 → clamped to 0.4 → floored to 0 → skipped.
        """
        broker = _make_broker(short_qty=0.4)
        executor = LiveExecutor(broker=broker)
        results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 5}})

        assert "rounds to 0" in results.get("AAPL", ""), f"Expected skip for 0-floor cover, got: {results}"
        broker.submit_order.assert_not_called()

    def test_cover_qty_unchanged_when_no_clamp_needed(self):
        """actual_short=20, requested qty=10 → no clamp → qty stays 10."""
        broker = _make_broker(short_qty=20.0)
        executor = LiveExecutor(broker=broker)
        executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 10}})

        call_kwargs = broker.submit_order.call_args.kwargs
        assert call_kwargs["qty"] == 10.0
