from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    name: str
    base_limit: float
    max_order_notional: float
    max_order_qty: float
    daily_loss_limit_pct: float


RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile("conservative", base_limit=0.10, max_order_notional=5_000.0, max_order_qty=500.0, daily_loss_limit_pct=0.02),
    "cautious": RiskProfile("cautious", base_limit=0.15, max_order_notional=7_500.0, max_order_qty=750.0, daily_loss_limit_pct=0.03),
    "balanced": RiskProfile("balanced", base_limit=0.20, max_order_notional=10_000.0, max_order_qty=1_000.0, daily_loss_limit_pct=0.05),
    "aggressive": RiskProfile("aggressive", base_limit=0.30, max_order_notional=20_000.0, max_order_qty=2_000.0, daily_loss_limit_pct=0.08),
    "speculative": RiskProfile("speculative", base_limit=0.50, max_order_notional=50_000.0, max_order_qty=5_000.0, daily_loss_limit_pct=0.15),
}

DEFAULT_PROFILE = "balanced"


def get_profile(name: str) -> RiskProfile:
    try:
        return RISK_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown risk profile {name!r}; choose from {sorted(RISK_PROFILES)}") from exc
