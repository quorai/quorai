"""Tests for news_sentiment_agent and its helpers."""

from unittest.mock import patch

import pytest

from src.agents.news_sentiment import (
    ArticleSentiment,
    BatchSentiment,
    _build_batch_prompt,
    _calculate_confidence_score,
    _dedupe_articles,
    news_sentiment_agent,
)
from src.data.models import CompanyNews


def _make_news(url: str, sentiment: str | None = None, title: str = "headline", date: str = "2024-03-08T10:00:00", summary: str | None = None) -> CompanyNews:
    return CompanyNews(ticker="AAPL", title=title, source="Reuters", date=date, url=url, sentiment=sentiment, summary=summary)


def _state(tickers: list[str] | None = None, end_date: str = "2024-03-08") -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": tickers or ["AAPL"],
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False, "api_keys": {}},
    }


def _batch(articles: list[tuple[int, str, int]]) -> BatchSentiment:
    """Build a BatchSentiment from (index, sentiment, confidence) tuples."""
    return BatchSentiment(articles=[ArticleSentiment(index=i, sentiment=s, confidence=c) for i, s, c in articles])


class TestDedupeArticles:
    def test_identical_title_keeps_first(self):
        a = _make_news("https://a.com/1", title="Apple Reports Record Q4 Earnings!")
        b = _make_news("https://a.com/2", title="apple reports record q4 earnings")
        result = _dedupe_articles([a, b])
        assert len(result) == 1
        assert result[0].url == "https://a.com/1"

    def test_different_titles_both_kept(self):
        a = _make_news("https://a.com/1", title="Apple beats earnings")
        b = _make_news("https://a.com/2", title="Tim Cook announces buyback")
        assert len(_dedupe_articles([a, b])) == 2

    def test_empty_list(self):
        assert _dedupe_articles([]) == []


class TestCalculateConfidenceScore:
    def test_empty_signals_returns_zero(self):
        result = _calculate_confidence_score(
            sentiment_confidences={},
            company_news=[],
            overall_signal="neutral",
            bullish_signals=0,
            bearish_signals=0,
            total_signals=0,
        )
        assert result == 0.0

    def test_all_bullish_with_llm_confidences(self):
        articles = [
            _make_news("https://a.com/1", sentiment="positive"),
            _make_news("https://a.com/2", sentiment="positive"),
        ]
        confidences = {a.url: 80 for a in articles}
        result = _calculate_confidence_score(
            sentiment_confidences=confidences,
            company_news=articles,
            overall_signal="bullish",
            bullish_signals=2,
            bearish_signals=0,
            total_signals=2,
        )
        # 70% of 80 + 30% of 100 = 56 + 30 = 86
        assert result == pytest.approx(86.0, abs=0.1)

    def test_fallback_to_proportion_when_no_llm_confidences(self):
        articles = [
            _make_news("https://a.com/1", sentiment="positive"),
            _make_news("https://a.com/2", sentiment="negative"),
            _make_news("https://a.com/3", sentiment="negative"),
        ]
        result = _calculate_confidence_score(
            sentiment_confidences={},
            company_news=articles,
            overall_signal="bearish",
            bullish_signals=1,
            bearish_signals=2,
            total_signals=3,
        )
        # proportion-only: 2/3 * 100 ≈ 66.67
        assert result == pytest.approx(66.67, abs=0.1)

    def test_tied_signals_returns_50(self):
        result = _calculate_confidence_score(
            sentiment_confidences={},
            company_news=[],
            overall_signal="neutral",
            bullish_signals=2,
            bearish_signals=2,
            total_signals=4,
        )
        assert result == pytest.approx(50.0, abs=0.1)


class TestNewsSentimentAgent:
    def test_bullish_signal_from_classified_articles(self):
        articles = [_make_news(f"https://a.com/{i}", title=f"positive news {i}") for i in range(3)]
        fake_batch = _batch([(1, "positive", 75), (2, "positive", 75), (3, "positive", 75)])

        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=articles),
            patch("src.agents.news_sentiment.call_llm", return_value=fake_batch),
            patch("src.agents.news_sentiment.progress"),
        ):
            result = news_sentiment_agent(_state())

        signals = result["data"]["analyst_signals"]["news_sentiment_agent"]
        assert signals["AAPL"]["signal"] == "bullish"
        assert signals["AAPL"]["confidence"] > 0

    def test_no_articles_gives_neutral_zero_confidence(self):
        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=[]),
            patch("src.agents.news_sentiment.progress"),
        ):
            result = news_sentiment_agent(_state())

        signals = result["data"]["analyst_signals"]["news_sentiment_agent"]
        assert signals["AAPL"]["signal"] == "neutral"
        assert signals["AAPL"]["confidence"] == 0.0

    def test_article_key_used_not_object_identity(self):
        """Confidence should survive even if articles were reconstructed (same URL, different id)."""
        article = _make_news("https://a.com/stable", title="good earnings")
        fake_batch = _batch([(1, "positive", 90)])

        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=[article]),
            patch("src.agents.news_sentiment.call_llm", return_value=fake_batch),
            patch("src.agents.news_sentiment.progress"),
        ):
            result = news_sentiment_agent(_state())

        signals = result["data"]["analyst_signals"]["news_sentiment_agent"]
        assert signals["AAPL"]["confidence"] > 0

    def test_single_batched_llm_call_per_ticker(self):
        """call_llm must be called exactly once per ticker, not once per article."""
        articles = [_make_news(f"https://a.com/{i}", title=f"headline {i}") for i in range(5)]
        fake_batch = _batch([(i + 1, "positive", 70) for i in range(5)])

        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=articles),
            patch("src.agents.news_sentiment.call_llm", return_value=fake_batch) as mock_llm,
            patch("src.agents.news_sentiment.progress"),
        ):
            news_sentiment_agent(_state())

        assert mock_llm.call_count == 1

    def test_recency_decay_old_bearish_does_not_flip_fresh_bullish(self):
        """Two fresh bullish articles should outweigh one very old bearish article."""
        fresh_bullish_1 = _make_news("https://a.com/1", title="great earnings today", date="2024-03-08T10:00:00")
        fresh_bullish_2 = _make_news("https://a.com/2", title="record revenue quarter", date="2024-03-07T10:00:00")
        old_bearish = _make_news("https://a.com/3", title="lawsuit filed last month", date="2024-01-01T10:00:00")

        fake_batch = _batch([(1, "positive", 80), (2, "positive", 80), (3, "negative", 80)])

        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=[fresh_bullish_1, fresh_bullish_2, old_bearish]),
            patch("src.agents.news_sentiment.call_llm", return_value=fake_batch),
            patch("src.agents.news_sentiment.progress"),
        ):
            result = news_sentiment_agent(_state(end_date="2024-03-08"))

        signals = result["data"]["analyst_signals"]["news_sentiment_agent"]
        assert signals["AAPL"]["signal"] == "bullish"

    def test_metrics_articles_payload_present(self):
        articles = [_make_news(f"https://a.com/{i}", title=f"news {i}") for i in range(2)]
        fake_batch = _batch([(1, "positive", 70), (2, "neutral", 50)])

        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=articles),
            patch("src.agents.news_sentiment.call_llm", return_value=fake_batch),
            patch("src.agents.news_sentiment.progress"),
        ):
            result = news_sentiment_agent(_state())

        metrics = result["data"]["analyst_signals"]["news_sentiment_agent"]["AAPL"]["reasoning"]["news_sentiment"]["metrics"]
        assert "articles" in metrics
        assert len(metrics["articles"]) > 0
        first = metrics["articles"][0]
        for key in ("title", "signal", "confidence", "age_days", "weight"):
            assert key in first

    def test_dedup_reduces_article_count(self):
        """Duplicate headlines should be collapsed before LLM classification."""
        a = _make_news("https://a.com/1", title="Apple Q4 Earnings Beat")
        b = _make_news("https://a.com/2", title="apple q4 earnings beat!")  # same normalized
        c = _make_news("https://a.com/3", title="CEO announces dividend increase")
        fake_batch = _batch([(1, "positive", 75), (2, "positive", 75)])

        with (
            patch("src.agents.news_sentiment.get_company_news", return_value=[a, b, c]),
            patch("src.agents.news_sentiment.call_llm", return_value=fake_batch) as mock_llm,
            patch("src.agents.news_sentiment.progress"),
        ):
            news_sentiment_agent(_state())

        # After dedup, only 2 unique articles → batch prompt should contain 2 numbered items
        prompt_arg = mock_llm.call_args[0][0]
        assert "1." in prompt_arg
        assert "2." in prompt_arg
        assert "3." not in prompt_arg


class TestBuildBatchPrompt:
    def test_contains_ticker(self):
        articles = [_make_news("https://a.com/1", title="AAPL earnings beat")]
        prompt = _build_batch_prompt("AAPL", articles)
        assert "AAPL" in prompt

    def test_contains_disambiguation_rule(self):
        articles = [_make_news("https://a.com/1")]
        prompt = _build_batch_prompt("AAPL", articles)
        assert "competitor" in prompt.lower()

    def test_contains_two_examples(self):
        articles = [_make_news("https://a.com/1")]
        prompt = _build_batch_prompt("AAPL", articles)
        assert '"index": 1' in prompt
        assert '"index": 2' in prompt

    def test_summary_included_when_present(self):
        articles = [_make_news("https://a.com/1", title="Earnings beat", summary="Revenue up 15% YoY")]
        prompt = _build_batch_prompt("AAPL", articles)
        assert "Revenue up 15% YoY" in prompt

    def test_summary_absent_when_none(self):
        articles = [_make_news("https://a.com/1", title="Earnings beat", summary=None)]
        prompt = _build_batch_prompt("AAPL", articles)
        # The " — " separator only appears when summary is present in the article line
        article_lines = [line for line in prompt.splitlines() if line.startswith("1.")]
        assert all(" — " not in line for line in article_lines)

    def test_article_date_included(self):
        articles = [_make_news("https://a.com/1", date="2024-03-08T10:00:00")]
        prompt = _build_batch_prompt("AAPL", articles)
        assert "2024-03-08" in prompt
