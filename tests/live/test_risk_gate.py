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

    log_file = tmp_path / f"trades-{date.today()}.jsonl"
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
