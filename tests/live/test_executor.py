from unittest.mock import MagicMock

from src.live.executor import LiveExecutor


def _make_order(symbol: str, side_value: str, action: str | None = None) -> MagicMock:
    order = MagicMock()
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side_value
    order.id = "order-123"
    # client_order_id format mirrors executor: YYYY-MM-DD-{TICKER}-{action}
    order.client_order_id = f"2024-01-15-{symbol}-{action or side_value}"
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

    from datetime import datetime
    import json
    from zoneinfo import ZoneInfo

    ny_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    log_file = tmp_path / "trades" / f"trades-{ny_date}.jsonl"
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


# R24 — cover qty whole-share flooring


def test_cover_fractional_qty_above_threshold_rounds_to_one():
    """cover qty=0.6 meets the 0.6 threshold and rounds to 1 whole share."""
    pos = _make_position("AAPL", "-10")
    broker = _make_broker(positions=[pos])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 0.6}})
    assert results["AAPL"] == "submitted"
    assert broker.submit_order.call_args.kwargs["qty"] == 1.0


def test_cover_fractional_qty_below_threshold_is_skipped():
    """cover qty=0.5 is below the 0.6 threshold, rounds to 0, and is skipped."""
    pos = _make_position("AAPL", "-10")
    broker = _make_broker(positions=[pos])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "cover", "quantity": 0.5}})
    assert "skipped" in results["AAPL"]
    broker.submit_order.assert_not_called()


def test_short_fractional_qty_rounds_consistently():
    """short qty flooring behaves the same as cover: 0.6 → 1, 0.5 → skipped."""
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)

    results_ok = executor.execute_decisions({"AAPL": {"action": "short", "quantity": 0.6}})
    assert results_ok["AAPL"] == "submitted"
    assert broker.submit_order.call_args.kwargs["qty"] == 1.0

    broker.submit_order.reset_mock()

    results_skip = executor.execute_decisions({"AAPL": {"action": "short", "quantity": 0.5}})
    assert "skipped" in results_skip["AAPL"]
    broker.submit_order.assert_not_called()


# R25 — NY date for client_order_id


def test_client_order_id_uses_ny_date(tmp_path):
    """client_order_id prefix must use NY date, not UTC, to avoid cross-midnight drift."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    from src.live.audit_journal import AuditJournal

    # 20:00 ET on Monday = 01:00 UTC Tuesday — NY date is Monday
    ny_time = datetime(2024, 1, 15, 20, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    broker = _make_broker()
    journal = AuditJournal(log_dir=str(tmp_path))
    executor = LiveExecutor(broker=broker, journal=journal)

    with patch("src.live.executor.now_ny", return_value=ny_time):
        results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 1.0}})

    assert results["AAPL"] == "submitted"
    order_id = broker.submit_order.call_args.kwargs["client_order_id"]
    # Must use 2024-01-15 (NY date), not 2024-01-16 (UTC date after midnight)
    assert order_id.startswith("2024-01-15"), f"Got client_order_id: {order_id}"


# R25b — sell vs short (and buy vs cover) must produce distinct client_order_ids


def test_client_order_id_sell_vs_short_are_distinct():
    """sell and short both map to side='sell' but must have different client_order_ids."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    ny_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    with patch("src.live.executor.now_ny", return_value=ny_time):
        broker_sell = _make_broker()
        executor_sell = LiveExecutor(broker=broker_sell)
        executor_sell.execute_decisions({"AAPL": {"action": "sell", "quantity": 5.0}})
        id_sell = broker_sell.submit_order.call_args.kwargs["client_order_id"]

        broker_short = _make_broker()
        executor_short = LiveExecutor(broker=broker_short)
        executor_short.execute_decisions({"AAPL": {"action": "short", "quantity": 5.0}})
        id_short = broker_short.submit_order.call_args.kwargs["client_order_id"]

    assert id_sell != id_short, f"sell and short produced same id: {id_sell}"
    assert id_sell.endswith("-sell"), id_sell
    assert id_short.endswith("-short"), id_short


def test_client_order_id_buy_vs_cover_are_distinct():
    """buy and cover both map to side='buy' but must have different client_order_ids."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    ny_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    pos = _make_position("AAPL", "-10")

    with patch("src.live.executor.now_ny", return_value=ny_time):
        broker_buy = _make_broker()
        executor_buy = LiveExecutor(broker=broker_buy)
        executor_buy.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})
        id_buy = broker_buy.submit_order.call_args.kwargs["client_order_id"]

        broker_cover = _make_broker(positions=[pos])
        executor_cover = LiveExecutor(broker=broker_cover)
        executor_cover.execute_decisions({"AAPL": {"action": "cover", "quantity": 5.0}})
        id_cover = broker_cover.submit_order.call_args.kwargs["client_order_id"]

    assert id_buy != id_cover, f"buy and cover produced same id: {id_buy}"
    assert id_buy.endswith("-buy"), id_buy
    assert id_cover.endswith("-cover"), id_cover


def test_missing_price_rejected_by_risk_gate(tmp_path):
    """RV-01: missing price in current_prices causes risk gate to reject the order."""
    from src.config import Settings
    from src.live.audit_journal import AuditJournal
    from src.live.risk_gate import RiskGate

    settings = Settings(
        ALPACA_API_KEY="x",
        ALPACA_SECRET_KEY="x",
        MAX_ORDER_NOTIONAL=10_000.0,
        MAX_ORDER_QTY=1_000.0,
        DAILY_LOSS_LIMIT_PCT=0.05,
        KILL_SWITCH=False,
    )
    journal = AuditJournal(log_dir=str(tmp_path))
    risk_gate = RiskGate(settings=settings, journal=journal)
    broker = _make_broker()
    executor = LiveExecutor(broker=broker, risk_gate=risk_gate, sod_equity=100_000.0)

    results = executor.execute_decisions(
        {"AAPL": {"action": "buy", "quantity": 5.0}},
        current_prices={},  # no price for AAPL
    )

    assert "rejected" in results["AAPL"]
    assert "missing_price" in results["AAPL"]
    broker.submit_order.assert_not_called()


def test_pending_sell_does_not_suppress_short():
    """RV-03: a pending sell order must not suppress a new short on the same ticker."""
    existing = _make_order("AAPL", "sell", action="sell")
    broker = _make_broker(open_orders=[existing])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "short", "quantity": 3.0}})
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once()


def test_pending_short_does_not_suppress_sell():
    """RV-03: a pending short order must not suppress a new sell on the same ticker."""
    pos = _make_position("AAPL", "10")
    existing = _make_order("AAPL", "sell", action="short")
    broker = _make_broker(open_orders=[existing], positions=[pos])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "sell", "quantity": 3.0}})
    assert results["AAPL"] == "submitted"
    broker.submit_order.assert_called_once()


def test_pending_buy_still_suppresses_duplicate_buy():
    """RV-03: dedup still works for the same action — a pending buy blocks another buy."""
    existing = _make_order("AAPL", "buy", action="buy")
    broker = _make_broker(open_orders=[existing])
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})
    assert results["AAPL"] == "skipped (open order exists)"
    broker.submit_order.assert_not_called()


def test_execute_decisions_checks_kill_switch_directly(tmp_path, monkeypatch):
    """RV-02: execute_decisions blocks orders when kill switch is toggled after construction."""
    from unittest.mock import MagicMock

    from src.config import Settings
    from src.live.audit_journal import AuditJournal
    from src.live.risk_gate import RiskGate

    # Gate constructed before the kill switch was toggled.
    settings_at_start = Settings(
        ALPACA_API_KEY="x",
        ALPACA_SECRET_KEY="x",
        MAX_ORDER_NOTIONAL=10_000.0,
        MAX_ORDER_QTY=1_000.0,
        DAILY_LOSS_LIMIT_PCT=0.05,
        KILL_SWITCH=False,
    )
    journal = AuditJournal(log_dir=str(tmp_path))
    risk_gate = RiskGate(settings=settings_at_start, journal=journal)
    broker = _make_broker()
    executor = LiveExecutor(broker=broker, risk_gate=risk_gate, sod_equity=100_000.0)

    # Simulate .env toggle: KILL_SWITCH=True after process start.
    live_settings = Settings(
        ALPACA_API_KEY="x",
        ALPACA_SECRET_KEY="x",
        MAX_ORDER_NOTIONAL=10_000.0,
        MAX_ORDER_QTY=1_000.0,
        DAILY_LOSS_LIMIT_PCT=0.05,
        KILL_SWITCH=True,
    )
    mock_get = MagicMock(return_value=live_settings)
    monkeypatch.setattr("src.live.risk_gate.get_settings", mock_get)
    monkeypatch.setattr("src.live.executor.get_settings", mock_get)

    results = executor.execute_decisions(
        {"AAPL": {"action": "buy", "quantity": 5.0}},
        current_prices={"AAPL": 150.0},
    )

    assert results["AAPL"] == "rejected: kill_switch_active"
    broker.submit_order.assert_not_called()


def _make_override_executor(tmp_path):
    """Helper: executor wired with a journal and an override-approved idempotency guard."""
    journal = __import__("src.live.audit_journal", fromlist=["AuditJournal"]).AuditJournal(log_dir=str(tmp_path))
    broker = _make_broker()
    guard = MagicMock()
    guard.check.return_value = (True, "override_approved")
    executor = LiveExecutor(broker=broker, idempotency_guard=guard, journal=journal)
    return executor, broker, journal


def test_override_suffix_skips_error_entries(tmp_path, monkeypatch):
    """RV-05: a prior -r1 entry with status=error must cause the new order to use -r2."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    ny_tz = ZoneInfo("America/New_York")
    fixed_now = datetime(2024, 1, 15, 10, 0, tzinfo=ny_tz)

    with patch("src.live.executor.now_ny", return_value=fixed_now), patch("src.live.audit_journal.now_ny", return_value=fixed_now):
        executor, broker, journal = _make_override_executor(tmp_path)

        # Pre-populate journal: prior attempt ended with error
        journal.record(
            ticker="AAPL",
            action="buy",
            qty=5.0,
            side="buy",
            status="error",
            order_id="2024-01-15-AAPL-buy-r1",
        )

        executor.execute_decisions(
            {"AAPL": {"action": "buy", "quantity": 5.0}},
            current_prices={"AAPL": 150.0},
        )

    order_id = broker.submit_order.call_args.kwargs["client_order_id"]
    assert order_id == "2024-01-15-AAPL-buy-r2", f"Expected -r2, got: {order_id}"


def test_negative_qty_skipped():
    """RV-06: negative quantity is rejected before reaching the broker."""
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": -5}})
    assert "skipped" in results["AAPL"]
    broker.submit_order.assert_not_called()


def test_nan_qty_skipped():
    """RV-06: NaN quantity is rejected before reaching the broker."""
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": float("nan")}})
    assert "skipped" in results["AAPL"]
    broker.submit_order.assert_not_called()


def test_inf_qty_skipped():
    """RV-06: infinite quantity is rejected before reaching the broker."""
    broker = _make_broker()
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions({"AAPL": {"action": "buy", "quantity": float("inf")}})
    assert "skipped" in results["AAPL"]
    broker.submit_order.assert_not_called()


def test_override_suffix_skips_rejected_entries(tmp_path, monkeypatch):
    """RV-05: a prior -r1 entry with status=rejected must cause the new order to use -r2."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    ny_tz = ZoneInfo("America/New_York")
    fixed_now = datetime(2024, 1, 15, 10, 0, tzinfo=ny_tz)

    with patch("src.live.executor.now_ny", return_value=fixed_now), patch("src.live.audit_journal.now_ny", return_value=fixed_now):
        executor, broker, journal = _make_override_executor(tmp_path)

        journal.record(
            ticker="AAPL",
            action="buy",
            qty=5.0,
            side="buy",
            status="rejected",
            order_id="2024-01-15-AAPL-buy-r1",
        )

        executor.execute_decisions(
            {"AAPL": {"action": "buy", "quantity": 5.0}},
            current_prices={"AAPL": 150.0},
        )

    order_id = broker.submit_order.call_args.kwargs["client_order_id"]
    assert order_id == "2024-01-15-AAPL-buy-r2", f"Expected -r2, got: {order_id}"


def _make_broker_with_equity(equity_value):
    """Helper: broker whose get_account().equity returns equity_value (str, None, or '')."""
    broker = _make_broker()
    account = MagicMock()
    account.equity = equity_value
    broker.get_account.return_value = account
    return broker


def test_raises_when_equity_missing():
    """RV-07: RuntimeError when get_account().equity is None."""
    import pytest

    broker = _make_broker_with_equity(None)
    risk_gate = MagicMock()
    risk_gate.check.return_value = (True, None)
    executor = LiveExecutor(broker=broker, risk_gate=risk_gate, sod_equity=100_000.0)
    with pytest.raises(RuntimeError, match="no equity value"):
        executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})


def test_raises_when_equity_empty_string():
    """RV-07: RuntimeError when get_account().equity is an empty string."""
    import pytest

    broker = _make_broker_with_equity("")
    risk_gate = MagicMock()
    risk_gate.check.return_value = (True, None)
    executor = LiveExecutor(broker=broker, risk_gate=risk_gate, sod_equity=100_000.0)
    with pytest.raises(RuntimeError, match="no equity value"):
        executor.execute_decisions({"AAPL": {"action": "buy", "quantity": 5.0}})


def test_equity_refreshed_every_n_orders(monkeypatch):
    """RV-09: get_account() is called once pre-loop plus once per N submitted orders."""
    from unittest.mock import patch

    from src.config import Settings

    broker = _make_broker(account_equity="100000")
    executor = LiveExecutor(broker=broker)

    risk_gate = MagicMock()
    risk_gate.check.return_value = (True, None)
    executor._risk_gate = risk_gate

    settings = Settings(EQUITY_REFRESH_INTERVAL=2)
    with patch("src.live.executor.get_settings", return_value=settings):
        executor.execute_decisions(
            {
                "AAPL": {"action": "buy", "quantity": 1.0},
                "MSFT": {"action": "buy", "quantity": 1.0},
                "GOOG": {"action": "buy", "quantity": 1.0},
                "AMZN": {"action": "buy", "quantity": 1.0},
            }
        )

    # 1 pre-loop + 2 mid-loop refreshes (at order counts 2 and 4)
    assert broker.get_account.call_count == 3


def test_override_journal_scanned_once(tmp_path):
    """RV-12: list_all_today() is called at most once regardless of how many decisions there are."""
    from datetime import datetime
    from unittest.mock import patch
    from zoneinfo import ZoneInfo

    ny_tz = ZoneInfo("America/New_York")
    fixed_now = datetime(2024, 1, 15, 10, 0, tzinfo=ny_tz)

    with patch("src.live.executor.now_ny", return_value=fixed_now), patch("src.live.audit_journal.now_ny", return_value=fixed_now):
        executor, broker, journal = _make_override_executor(tmp_path)

        with patch.object(journal, "list_all_today", wraps=journal.list_all_today) as mock_scan:
            executor.execute_decisions(
                {
                    "AAPL": {"action": "buy", "quantity": 1.0},
                    "MSFT": {"action": "buy", "quantity": 1.0},
                    "GOOG": {"action": "buy", "quantity": 1.0},
                },
                current_prices={"AAPL": 150.0, "MSFT": 300.0, "GOOG": 120.0},
            )

    assert mock_scan.call_count <= 1, f"list_all_today called {mock_scan.call_count} times; expected at most 1"


def test_abort_on_error_stops_remaining_orders():
    """RV-13: when abort_on_error=True, a submission error stops the batch immediately."""
    broker = _make_broker()
    broker.submit_order.side_effect = [RuntimeError("broker down"), MagicMock(id="order-2")]
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions(
        {
            "AAPL": {"action": "buy", "quantity": 1.0},
            "MSFT": {"action": "buy", "quantity": 1.0},
        },
        abort_on_error=True,
    )
    assert "error" in results["AAPL"]
    assert results.get("batch_aborted") == "1"
    assert "MSFT" not in results  # second ticker never attempted
    assert broker.submit_order.call_count == 1


def test_no_abort_continues_on_error():
    """RV-13: when abort_on_error=False, all tickers are attempted despite an error."""
    broker = _make_broker()
    ok_order = MagicMock()
    ok_order.id = "order-2"
    broker.submit_order.side_effect = [RuntimeError("broker down"), ok_order]
    executor = LiveExecutor(broker=broker)
    results = executor.execute_decisions(
        {
            "AAPL": {"action": "buy", "quantity": 1.0},
            "MSFT": {"action": "buy", "quantity": 1.0},
        },
        abort_on_error=False,
    )
    assert "error" in results["AAPL"]
    assert results["MSFT"] == "submitted"
    assert "batch_aborted" not in results
    assert broker.submit_order.call_count == 2
