"""Tests for sentiment_analyst_agent (insider-trades-only path)."""

from unittest.mock import patch

from src.agents.sentiment import sentiment_analyst_agent
from src.data.models import InsiderTrade


def _trade(ticker: str, shares: float) -> InsiderTrade:
    return InsiderTrade(
        ticker=ticker,
        issuer=None,
        name=None,
        title=None,
        is_board_director=None,
        transaction_date="2024-03-01",
        transaction_shares=shares,
        transaction_price_per_share=None,
        transaction_value=None,
        shares_owned_before_transaction=None,
        shares_owned_after_transaction=None,
        security_title=None,
        filing_date=None,
    )


def _state(tickers: list[str] | None = None) -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": tickers or ["AAPL"],
            "end_date": "2024-03-08",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False, "api_keys": {}},
    }


class TestSentimentAnalystAgent:
    def test_bullish_when_insider_buys_dominate(self):
        trades = [_trade("AAPL", 1000), _trade("AAPL", 500), _trade("AAPL", -100)]
        with (
            patch("src.agents.sentiment.get_insider_trades", return_value=trades),
            patch("src.agents.sentiment.progress"),
        ):
            result = sentiment_analyst_agent(_state())

        signals = result["data"]["analyst_signals"]["sentiment_analyst_agent"]
        assert signals["AAPL"]["signal"] == "bullish"
        assert signals["AAPL"]["confidence"] > 50

    def test_bearish_when_insider_sells_dominate(self):
        trades = [_trade("AAPL", -2000), _trade("AAPL", -500), _trade("AAPL", 100)]
        with (
            patch("src.agents.sentiment.get_insider_trades", return_value=trades),
            patch("src.agents.sentiment.progress"),
        ):
            result = sentiment_analyst_agent(_state())

        signals = result["data"]["analyst_signals"]["sentiment_analyst_agent"]
        assert signals["AAPL"]["signal"] == "bearish"

    def test_no_insider_trades_gives_neutral_zero_confidence(self):
        with (
            patch("src.agents.sentiment.get_insider_trades", return_value=[]),
            patch("src.agents.sentiment.progress"),
        ):
            result = sentiment_analyst_agent(_state())

        signals = result["data"]["analyst_signals"]["sentiment_analyst_agent"]
        assert signals["AAPL"]["signal"] == "neutral"
        assert signals["AAPL"]["confidence"] == 0

    def test_reasoning_has_no_news_sentiment_key(self):
        trades = [_trade("AAPL", 500)]
        with (
            patch("src.agents.sentiment.get_insider_trades", return_value=trades),
            patch("src.agents.sentiment.progress"),
        ):
            result = sentiment_analyst_agent(_state())

        reasoning = result["data"]["analyst_signals"]["sentiment_analyst_agent"]["AAPL"]["reasoning"]
        assert "news_sentiment" not in reasoning
        assert "insider_trading" in reasoning

    def test_zero_share_trades_not_counted_as_bullish(self):
        """
        R49: transaction_shares=0 is not a buy — must not inflate bullish_count.
        Before the fix: `>= 0` included 0-share rows as bullish.
        After the fix: `> 0` correctly treats 0 shares as neither bullish nor bearish.
        """
        trades = [_trade("AAPL", 0), _trade("AAPL", 0), _trade("AAPL", -100)]
        with (
            patch("src.agents.sentiment.get_insider_trades", return_value=trades),
            patch("src.agents.sentiment.progress"),
        ):
            result = sentiment_analyst_agent(_state())

        signals = result["data"]["analyst_signals"]["sentiment_analyst_agent"]["AAPL"]
        metrics = signals["reasoning"]["insider_trading"]["metrics"]
        assert metrics["bullish_trades"] == 0, f"Zero-share rows must not count as bullish, got {metrics['bullish_trades']}"
        assert signals["signal"] == "bearish", f"1 sell vs 0 real buys → bearish. Got: {signals['signal']}"

    def test_zero_share_trades_mixed_still_neutral_without_real_buys(self):
        """Only zero-share rows (no real buys or sells) → neutral with 0 confidence."""
        trades = [_trade("AAPL", 0), _trade("AAPL", 0)]
        with (
            patch("src.agents.sentiment.get_insider_trades", return_value=trades),
            patch("src.agents.sentiment.progress"),
        ):
            result = sentiment_analyst_agent(_state())

        signals = result["data"]["analyst_signals"]["sentiment_analyst_agent"]["AAPL"]
        assert signals["signal"] == "neutral"
