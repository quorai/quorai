from __future__ import annotations

from src.regime.classifier import MarketRegime
from src.utils.analysts import get_strategy_groups

# Which strategy groups are active per regime.
# sentiment_and_analytical always runs (it provides the analytical/fundamental baseline).
REGIME_GROUP_POLICY: dict[MarketRegime, list[str] | None] = {
    MarketRegime.BULL_TREND: [
        "growth_and_catalyst",
        "quant_systematic",
        "quality_compounders",
        "sentiment_and_analytical",
    ],
    MarketRegime.BEAR_TREND: [
        "deep_value",
        "macro_and_cycle",
        "quality_compounders",
        "sentiment_and_analytical",
    ],
    MarketRegime.RISK_OFF: [
        "macro_and_cycle",
        "deep_value",
        "sentiment_and_analytical",
    ],
    MarketRegime.NEUTRAL: None,  # None means all groups
}


def select_analysts_for_regime(regime: MarketRegime) -> list[str] | None:
    """Return the analyst keys to run for a given regime, or None to run all."""
    groups = REGIME_GROUP_POLICY.get(regime)
    if groups is None:
        return None

    group_map = get_strategy_groups()  # {group_key: [analyst_key, ...]}
    selected: list[str] = []
    for group in groups:
        selected.extend(group_map.get(group, []))
    return selected if selected else None
