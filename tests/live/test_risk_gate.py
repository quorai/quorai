import pytest

from src.config import Settings
from src.live.audit_journal import AuditJournal
from src.live.risk_gate import RiskGate


def _make_gate(tmp_path, **overrides) -> RiskGate:
    defaults = dict(
        ALPACA_API_KEY="x",
        ALPACA_SECRET_KEY="x",
        MAX_ORDER_NOTIONAL=10_000.0,
        MAX_ORDER_QTY=1_000.0,
        DAILY_LOSS_LIMIT_PCT=0.05,
        KILL_SWITCH=False,
    )
    defaults.update(overrides)
    settings = Settings(**defaults)
    journal = AuditJournal(log_dir=str(tmp_path))
    return RiskGate(settings=settings, journal=journal)


def _check(gate: RiskGate, **kwargs) -> tuple[bool, str]:
    defaults = dict(ticker="AAPL", action="buy", side="buy", qty=10.0, price=100.0, account_equity=100_000.0, sod_equity=100_000.0)
    defaults.update(kwargs)
    return gate.check(**defaults)


def test_happy_path(tmp_path):
    gate = _make_gate(tmp_path)
    allowed, reason = _check(gate)
    assert allowed is True
    assert reason == ""


def test_kill_switch(tmp_path):
    gate = _make_gate(tmp_path, KILL_SWITCH=True)
    allowed, reason = _check(gate)
    assert allowed is False
    assert reason == "kill_switch_active"


def test_notional_exceeded(tmp_path):
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=500.0)
    allowed, reason = _check(gate, qty=10.0, price=100.0)  # notional = 1000 > 500
    assert allowed is False
    assert reason == "notional_exceeded"


def test_qty_exceeded(tmp_path):
    gate = _make_gate(tmp_path, MAX_ORDER_QTY=5.0)
    allowed, reason = _check(gate, qty=10.0, price=1.0)  # qty=10 > 5
    assert allowed is False
    assert reason == "qty_exceeded"


def test_daily_loss_limit(tmp_path):
    gate = _make_gate(tmp_path, DAILY_LOSS_LIMIT_PCT=0.05)
    # equity dropped 10% below SOD
    allowed, reason = _check(gate, account_equity=90_000.0, sod_equity=100_000.0)
    assert allowed is False
    assert reason == "daily_loss_limit"


def test_daily_loss_limit_zero_sod(tmp_path):
    gate = _make_gate(tmp_path, DAILY_LOSS_LIMIT_PCT=0.05)
    # sod_equity=0 should not trigger loss limit
    allowed, reason = _check(gate, account_equity=0.0, sod_equity=0.0)
    assert allowed is True


def test_rejection_is_journaled(tmp_path):
    from datetime import date
    import json

    gate = _make_gate(tmp_path, KILL_SWITCH=True)
    _check(gate)

    log_file = tmp_path / "trades" / f"trades-{date.today()}.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text().strip())
    assert entry["status"] == "rejected"
    assert entry["reason"] == "kill_switch_active"


@pytest.mark.parametrize(
    "condition,kwargs,expected_reason",
    [
        ("kill_switch", {"KILL_SWITCH": True}, "kill_switch_active"),
        ("notional", {"MAX_ORDER_NOTIONAL": 1.0}, "notional_exceeded"),
        ("qty", {"MAX_ORDER_QTY": 0.1}, "qty_exceeded"),
    ],
)
def test_rejection_conditions(tmp_path, condition, kwargs, expected_reason):
    gate = _make_gate(tmp_path, **kwargs)
    allowed, reason = _check(gate)
    assert allowed is False
    assert reason == expected_reason


def test_daily_loss_limit_exact_boundary(tmp_path):
    """
    R48: Drawdown of exactly DAILY_LOSS_LIMIT_PCT must trip the gate.
    Before the fix: `<` let the exact boundary pass through.
    After the fix: `<=` rejects at the limit.
    """
    gate = _make_gate(tmp_path, DAILY_LOSS_LIMIT_PCT=0.03)
    # account_equity / sod_equity - 1 == -0.03 exactly
    allowed, reason = _check(gate, account_equity=97_000.0, sod_equity=100_000.0)
    assert allowed is False, "Exact daily loss limit must trigger rejection"
    assert reason == "daily_loss_limit"


def test_daily_loss_limit_just_below_boundary(tmp_path):
    """Drawdown just short of the limit is still permitted."""
    gate = _make_gate(tmp_path, DAILY_LOSS_LIMIT_PCT=0.03)
    # -2.99% drawdown — just under the 3% limit
    allowed, reason = _check(gate, account_equity=97_010.0, sod_equity=100_000.0)
    assert allowed is True, f"Drawdown below limit must be allowed, got: {reason}"


# --- Closing-trade notional exemption ---


def test_sell_closing_long_exempt_from_notional_cap(tmp_path):
    """Selling all of a held long position bypasses the notional cap."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0)
    # 100 shares × $300 = $30K gross, but entire qty closes existing long → allowed
    allowed, reason = _check(gate, action="sell", side="sell", qty=100.0, price=300.0, current_long=100.0)
    assert allowed is True, f"Expected allowed, got: {reason}"


def test_cover_closing_short_exempt_from_notional_cap(tmp_path):
    """Covering all of a held short position bypasses the notional cap."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0)
    # 200 shares × $80 = $16K gross, but entire qty covers existing short → allowed
    allowed, reason = _check(gate, action="cover", side="buy", qty=200.0, price=80.0, current_short=200.0)
    assert allowed is True, f"Expected allowed, got: {reason}"


def test_sell_partial_close_then_flip_capped(tmp_path):
    """A sell that exceeds the long position: only the flip portion counts against the cap."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0)
    # 200 qty, 50 long → closing=50, opening=150; 150 × $100 = $15K > $10K → rejected
    allowed, reason = _check(gate, action="sell", side="sell", qty=200.0, price=100.0, current_long=50.0)
    assert allowed is False
    assert reason == "notional_exceeded"


def test_buy_not_exempt(tmp_path):
    """Buy orders are never closing trades — the full notional is checked."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0)
    allowed, reason = _check(gate, action="buy", side="buy", qty=100.0, price=300.0, current_long=0.0)
    assert allowed is False
    assert reason == "notional_exceeded"


def test_short_not_exempt(tmp_path):
    """Short orders open new exposure — the full notional is checked."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0)
    allowed, reason = _check(gate, action="short", side="sell", qty=100.0, price=300.0, current_short=0.0)
    assert allowed is False
    assert reason == "notional_exceeded"


def test_closing_sell_still_hits_daily_loss_limit(tmp_path):
    """Closing a long does not bypass the daily loss limit."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0, DAILY_LOSS_LIMIT_PCT=0.05)
    allowed, reason = _check(
        gate,
        action="sell",
        side="sell",
        qty=100.0,
        price=300.0,
        current_long=100.0,
        account_equity=90_000.0,
        sod_equity=100_000.0,
    )
    assert allowed is False
    assert reason == "daily_loss_limit"


def test_closing_sell_still_hits_qty_cap(tmp_path):
    """Closing a long does not bypass the per-order quantity cap."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0, MAX_ORDER_QTY=50.0)
    allowed, reason = _check(gate, action="sell", side="sell", qty=100.0, price=1.0, current_long=100.0)
    assert allowed is False
    assert reason == "qty_exceeded"


def test_backward_compat_no_position_args(tmp_path):
    """Calling check() without current_long/current_short treats the order as opening (old behavior)."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=10_000.0)
    # Would be a closing sell, but no position provided → treated as opening → capped
    allowed, reason = _check(gate, action="sell", side="sell", qty=100.0, price=300.0)
    assert allowed is False
    assert reason == "notional_exceeded"


# --- Missing / zero price guard (RV-01) ---


def test_rejects_when_price_is_zero(tmp_path):
    gate = _make_gate(tmp_path)
    allowed, reason = _check(gate, price=0.0)
    assert allowed is False
    assert reason == "missing_price"


def test_rejects_when_price_is_negative(tmp_path):
    gate = _make_gate(tmp_path)
    allowed, reason = _check(gate, price=-1.0)
    assert allowed is False
    assert reason == "missing_price"


def test_missing_price_rejected_before_notional(tmp_path):
    """Zero price must be caught before the notional calculation (which would pass at 0 * qty = 0)."""
    gate = _make_gate(tmp_path, MAX_ORDER_NOTIONAL=1.0)
    allowed, reason = _check(gate, qty=1_000_000.0, price=0.0)
    assert allowed is False
    assert reason == "missing_price"


# --- Kill-switch hot-reload (RV-02) ---


def test_kill_switch_respected_after_settings_change(tmp_path, monkeypatch):
    """RV-02: kill switch toggled in .env after construction is picked up by the risk gate."""
    from unittest.mock import MagicMock

    gate = _make_gate(tmp_path, KILL_SWITCH=False)

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

    allowed, reason = _check(gate)
    assert allowed is False
    assert reason == "kill_switch_active"
