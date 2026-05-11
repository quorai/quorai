from unittest.mock import MagicMock

from src.live.executor import LiveExecutor


def _make_order(symbol: str, side_value: str) -> MagicMock:
    order = MagicMock()
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side_value
    order.id = "order-123"
    return order


def _make_broker(open_orders=None, account_equity=100_000.0):
    broker = MagicMock()
    broker.get_open_orders.return_value = open_orders or []
    account = MagicMock()
    account.equity = account_equity
    broker.get_account.return_value = account
    submitted_order = MagicMock()
    submitted_order.id = "order-xyz"
    broker.submit_order.return_value = submitted_order
    asset = MagicMock()
    asset.shortable = True
    broker.get_asset.return_value = asset
    return broker


def test_hold_is_skipped():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "hold", "quantity": 0}})
    assert results["AAPL"] == "skipped"
    broker.submit_order.assert_not_called()


def test_zero_qty_is_skipped():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 0}})
    assert results["AAPL"] == "skipped"


def test_buy_submitted():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once_with(ticker="AAPL", side="buy", qty=5.0)


def test_sell_submitted():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "sell", "quantity": 3.0}})
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once_with(ticker="AAPL", side="sell", qty=3.0)


def test_cover_maps_to_buy():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 2.0}})
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once_with(ticker="AAPL", side="buy", qty=2.0)


def test_short_maps_to_sell():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "short", "quantity": 4.0}})
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once_with(ticker="AAPL", side="sell", qty=4.0)


def test_duplicate_open_order_skipped():
    existing = _make_order("AAPL", "buy")
    broker = _make_broker(open_orders=[existing])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 1.0}})
    assert results["AAPL"] == "skipped (open order exists)"
    broker.submit_order.assert_not_called()


def test_risk_rejection():
    broker = _make_broker()
    risk_gate = MagicMock()
    risk_gate.check.return_value = (False, "kill_switch_active")
    executor = LiveExecutor(broker=broker, risk_gate=risk_gate, sod_equity=100_000.0)
    results = executor.execute_decisions(
        {"AAPL": {"action": "buy", "quantity": 5.0}},
        current_prices={"AAPL": 100.0},
    )
    assert "rejected" in results["AAPL"]
    broker.submit_order.assert_not_called()


def test_order_error_journaled(tmp_path):
    broker = _make_broker()
    broker.submit_order.side_effect = RuntimeError("API error")
    from src.live.audit_journal import AuditJournal

    journal = AuditJournal(log_dir=str(tmp_path))
    executor = LiveExecutor(broker=broker, journal=journal)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 1.0}})
    assert "error" in results["AAPL"]
    from datetime import date
    import json

    log_file = tmp_path / f"trades-{date.today()}.jsonl"
    entry = json.loads(log_file.read_text().strip())
    assert entry["status"] == "error"


def test_dry_run_does_not_submit():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}}, dry_run=True)
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_not_called()


def test_idempotency_guard_blocks_real_run():
    broker = _make_broker()
    guard = MagicMock()
    guard.check.return_value = (False, "denied:reject")
    executor = LiveExecutor(broker=broker, idempotency_guard=guard)

    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})

    assert results["AAPL"] == "skipped (idempotency: denied:reject)"
    broker.submit_order.assert_not_called()


def test_idempotency_guard_allows_real_run():
    broker = _make_broker()
    guard = MagicMock()
    guard.check.return_value = (True, "override_approved")
    executor = LiveExecutor(broker=broker, idempotency_guard=guard)

    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})

    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once()


def test_idempotency_guard_skipped_on_dry_run():
    broker = _make_broker()
    guard = MagicMock()
    guard.check.side_effect = AssertionError("guard must not be called on dry run")
    executor = LiveExecutor(broker=broker, idempotency_guard=guard)

    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}}, dry_run=True)

    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_not_called()
