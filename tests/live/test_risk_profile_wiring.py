"""Verify risk profile caps flow correctly into RiskGate via Settings.model_copy."""

from src.config import Settings
from src.live.audit_journal import AuditJournal
from src.live.risk_gate import RiskGate
from src.risk_profiles import get_profile


def _apply_profile_to_settings(profile_name: str) -> Settings:
    profile = get_profile(profile_name)
    base = Settings(ALPACA_API_KEY="x", ALPACA_SECRET_KEY="x")
    return base.model_copy(
        update={
            "MAX_ORDER_NOTIONAL": profile.max_order_notional,
            "MAX_ORDER_QTY": profile.max_order_qty,
            "DAILY_LOSS_LIMIT_PCT": profile.daily_loss_limit_pct,
        }
    )


def test_conservative_caps_applied(tmp_path):
    settings = _apply_profile_to_settings("conservative")
    assert settings.MAX_ORDER_NOTIONAL == 5_000.0
    assert settings.MAX_ORDER_QTY == 500.0
    assert settings.DAILY_LOSS_LIMIT_PCT == 0.02


def test_speculative_caps_applied(tmp_path):
    settings = _apply_profile_to_settings("speculative")
    assert settings.MAX_ORDER_NOTIONAL == 50_000.0
    assert settings.MAX_ORDER_QTY == 5_000.0
    assert settings.DAILY_LOSS_LIMIT_PCT == 0.15


def test_conservative_gate_blocks_oversized_order(tmp_path):
    profile = get_profile("conservative")
    settings = Settings(
        ALPACA_API_KEY="x",
        ALPACA_SECRET_KEY="x",
        MAX_ORDER_NOTIONAL=profile.max_order_notional,
        MAX_ORDER_QTY=profile.max_order_qty,
        DAILY_LOSS_LIMIT_PCT=profile.daily_loss_limit_pct,
    )
    journal = AuditJournal(log_dir=str(tmp_path))
    gate = RiskGate(settings=settings, journal=journal)
    # 60 shares × $100 = $6_000 > conservative notional cap of $5_000
    allowed, reason = gate.check(ticker="AAPL", action="buy", side="buy", qty=60.0, price=100.0, account_equity=100_000.0, sod_equity=100_000.0)
    assert allowed is False
    assert reason == "notional_exceeded"


def test_balanced_matches_defaults(tmp_path):
    """balanced profile should not change Settings defaults."""
    settings = _apply_profile_to_settings("balanced")
    defaults = Settings(ALPACA_API_KEY="x", ALPACA_SECRET_KEY="x")
    assert settings.MAX_ORDER_NOTIONAL == defaults.MAX_ORDER_NOTIONAL
    assert settings.MAX_ORDER_QTY == defaults.MAX_ORDER_QTY
    assert settings.DAILY_LOSS_LIMIT_PCT == defaults.DAILY_LOSS_LIMIT_PCT
