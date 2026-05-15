"""Tests for R35: growth_agent analyze_insider_conviction handles transaction_shares=None."""

from unittest.mock import MagicMock


def _trade(value=10_000.0, shares=100):
    t = MagicMock()
    t.transaction_value = value
    t.transaction_shares = shares
    return t


class TestInsiderConvictionNoneShares:
    def test_none_shares_does_not_raise(self):
        """transaction_shares=None must not cause TypeError when compared to 0."""
        from src.agents.growth_agent import analyze_insider_conviction

        trades = [
            _trade(value=5_000, shares=None),  # value-only SEC filing
            _trade(value=3_000, shares=50),  # normal buy
            _trade(value=-2_000, shares=-20),  # normal sell
        ]
        result = analyze_insider_conviction(trades)
        assert "score" in result

    def test_none_shares_treated_as_skip(self):
        """A trade with shares=None is not counted as either a buy or sell."""
        from src.agents.growth_agent import analyze_insider_conviction

        trades_with_none = [_trade(value=1_000, shares=None)]
        result_none = analyze_insider_conviction(trades_with_none)

        # Result should be the same as empty trades (shares=None → no net flow)
        trades_empty = []
        result_empty = analyze_insider_conviction(trades_empty)

        assert result_none["score"] == result_empty["score"]

    def test_normal_trades_still_work(self):
        """Regular trades (shares != None) are still counted correctly."""
        from src.agents.growth_agent import analyze_insider_conviction

        # 3 buys, 0 sells → bullish signal
        buys = [_trade(value=10_000, shares=100) for _ in range(3)]
        result = analyze_insider_conviction(buys)
        assert result["score"] >= 0
