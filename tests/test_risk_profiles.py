import pytest

from src.config import Settings
from src.risk_profiles import DEFAULT_PROFILE, RISK_PROFILES, get_profile


def test_five_profiles_exist():
    assert len(RISK_PROFILES) == 5
    assert set(RISK_PROFILES) == {"conservative", "cautious", "balanced", "aggressive", "speculative"}


def test_monotonic_order():
    order = ["conservative", "cautious", "balanced", "aggressive", "speculative"]
    limits = [RISK_PROFILES[n].base_limit for n in order]
    notionals = [RISK_PROFILES[n].max_order_notional for n in order]
    assert limits == sorted(limits), "base_limit should be non-decreasing conservative→speculative"
    assert notionals == sorted(notionals), "max_order_notional should be non-decreasing"


def test_balanced_matches_settings_defaults():
    p = get_profile("balanced")
    defaults = Settings()
    assert p.base_limit == 0.20
    assert p.max_order_notional == defaults.MAX_ORDER_NOTIONAL
    assert p.max_order_qty == defaults.MAX_ORDER_QTY
    assert p.daily_loss_limit_pct == defaults.DAILY_LOSS_LIMIT_PCT


def test_default_profile_is_balanced():
    assert DEFAULT_PROFILE == "balanced"


def test_get_profile_unknown_raises():
    with pytest.raises(ValueError, match="Unknown risk profile"):
        get_profile("moon_yolo_ultra")


def test_all_profiles_are_frozen():
    p = get_profile("aggressive")
    with pytest.raises(Exception):
        p.base_limit = 99.0  # type: ignore[misc]
