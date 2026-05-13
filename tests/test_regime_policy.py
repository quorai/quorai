from __future__ import annotations

from src.regime.classifier import MarketRegime
from src.regime.policy import select_analysts_for_regime
from src.utils.analysts import ANALYST_CONFIG


def test_bull_trend_returns_growth_group_members():
    result = select_analysts_for_regime(MarketRegime.BULL_TREND)
    assert result is not None
    assert len(result) > 0
    # growth_and_catalyst members should be included
    assert "bill_ackman" in result or "peter_lynch" in result


def test_neutral_returns_none():
    assert select_analysts_for_regime(MarketRegime.NEUTRAL) is None


def test_all_returned_keys_are_valid():
    valid_keys = set(ANALYST_CONFIG.keys())
    for regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND, MarketRegime.RISK_OFF):
        result = select_analysts_for_regime(regime)
        assert result is not None
        for key in result:
            assert key in valid_keys, f"{key!r} not in ANALYST_CONFIG for regime {regime}"


def test_all_non_neutral_regimes_produce_non_empty_selections():
    for regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND, MarketRegime.RISK_OFF):
        result = select_analysts_for_regime(regime)
        assert result, f"Expected non-empty selection for {regime}"
