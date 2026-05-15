from unittest.mock import MagicMock

from src.live.executor import LiveExecutor


def _make_order(symbol: str, side_value: str) -> MagicMock:
    order = MagicMock()
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side_value
    order.id = "order-123"
    return order


def _make_broker(open_orders=None, account_equity=100_000.0, positions=None):
    broker = MagicMock()
    broker.get_open_orders.return_value = open_orders or []
    broker.get_positions.return_value = positions or []
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


def _make_position(symbol: str, qty: str) -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    return pos


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
    call_kwargs = broker.submit_order.call_args.kwargs
    assert call_kwargs["ticker"] == "AAPL"
    assert call_kwargs["side"] == "buy"
    assert call_kwargs["qty"] == 5.0
    assert "client_order_id" in call_kwargs


def test_sell_submitted():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "sell", "quantity": 3.0}})
    assert results["AAPL"] == "submitted"
    call_kwargs = broker.submit_order.call_args.kwargs
    assert call_kwargs["ticker"] == "AAPL"
    assert call_kwargs["side"] == "sell"
    assert call_kwargs["qty"] == 3.0
    assert "client_order_id" in call_kwargs


def test_cover_maps_to_buy():
    pos = _make_position("AAPL", "-10")
    broker = _make_broker(positions=[pos])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 2.0}})
    assert results["AAPL"] == "submitted"
    call_kwargs = broker.submit_order.call_args.kwargs
    assert call_kwargs["side"] == "buy"
    assert "client_order_id" in call_kwargs


def test_short_maps_to_sell():
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "short", "quantity": 4.0}})
    assert results["AAPL"] == "submitted"
    call_kwargs = broker.submit_order.call_args.kwargs
    assert call_kwargs["side"] == "sell"
    assert "client_order_id" in call_kwargs


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

    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    log_file = tmp_path / f"trades-{ny_date}.jsonl"
    lines = [json.loads(ln) for ln in log_file.read_text().splitlines() if ln.strip()]
    # Two records: pending (written before submit) + error (written after failure)
    assert lines[0]["status"] == "pending"
    assert lines[-1]["status"] == "error"


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


def test_cover_clamped_to_actual_short():
    pos = _make_position("AAPL", "-5")
    broker = _make_broker(positions=[pos])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 10.0}})
    assert results["AAPL"] == "submitted"
    call_kwargs = broker.submit_order.call_args.kwargs
    assert call_kwargs["qty"] == 5.0  # clamped to actual short
    assert call_kwargs["side"] == "buy"


def test_cover_skipped_when_no_short():
    broker = _make_broker(positions=[])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 3.0}})
    assert results["AAPL"] == "skipped (cover: no short)"
    broker.submit_order.assert_not_called()
