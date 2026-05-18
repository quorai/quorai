import datetime
import json
import re

from langchain_core.messages import HumanMessage
from pydantic import AliasChoices, BaseModel, Field
from typing_extensions import Literal

from src.data.models import CompanyNews
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.concurrency import parallel_per_ticker
from src.utils.llm import call_llm
from src.utils.progress import progress

NEWS_SENTIMENT_MAX_ARTICLES = 20
RECENCY_HALF_LIFE_DAYS = 3
FALLBACK_ARTICLE_AGE_DAYS = 7.0

_SIGNAL_TO_SENTIMENT = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}


def _dedupe_articles(articles: list[CompanyNews]) -> list[CompanyNews]:
    """Remove duplicate headlines, keeping the first (most recent) occurrence."""
    seen: set[str] = set()
    result = []
    for a in articles:
        norm = re.sub(r"\W+", " ", a.title.lower()).strip()
        if norm not in seen:
            seen.add(norm)
            result.append(a)
    return result


def _article_age_days(news: CompanyNews, ref_dt: datetime.datetime) -> float:
    try:
        article_dt = datetime.datetime.fromisoformat(news.date)
        if article_dt.tzinfo is None:
            article_dt = article_dt.replace(tzinfo=datetime.timezone.utc)
        return max(0.0, (ref_dt - article_dt).total_seconds() / 86400)
    except Exception:
        return FALLBACK_ARTICLE_AGE_DAYS


class ArticleSentiment(BaseModel):
    """Sentiment classification for one article in a batch."""

    index: int
    sentiment: Literal["neutral", "positive", "negative"]
    confidence: int = Field(description="Confidence 0-100", validation_alias=AliasChoices("confidence", "confidence_score"))


class BatchSentiment(BaseModel):
    """Batch sentiment response for multiple articles."""

    articles: list[ArticleSentiment]


# Kept for backwards compatibility with existing tests.
class Sentiment(BaseModel):
    """Represents the sentiment of a single news article."""

    sentiment: Literal["neutral", "positive", "negative"]
    confidence: int = Field(description="Confidence 0-100", validation_alias=AliasChoices("confidence", "confidence_score"))


def _build_batch_prompt(ticker: str, articles: list[CompanyNews]) -> str:
    """Build the batched sentiment classification prompt with disambiguation rules and examples."""
    lines = []
    for i, a in enumerate(articles, start=1):
        suffix = f" — {a.summary}" if a.summary else ""
        lines.append(f"{i}. [{a.date[:10]}] {a.title}{suffix}")
    numbered = "\n".join(lines)
    return (
        f"You are a financial news sentiment classifier for stock {ticker}.\n\n"
        f"Rules:\n"
        f"- Classify each article as positive, negative, or neutral for {ticker} ONLY.\n"
        f"- If the article is not primarily about {ticker} (e.g., it covers a competitor,\n"
        f"  broad market moves, or macro events that do not directly affect {ticker}),\n"
        f"  return neutral with confidence ≤ 30.\n"
        f"- Provide a confidence score 0-100 for each classification.\n\n"
        f"Examples (for {ticker}):\n"
        f'A. "{ticker} reports record quarterly revenue, beating estimates by 12%"\n'
        f'   → {{"index": 1, "sentiment": "positive", "confidence": 88}}\n'
        f'B. "Competitor announces aggressive price cuts in key market"\n'
        f'   → {{"index": 2, "sentiment": "neutral", "confidence": 12}}\n\n'
        f'Respond ONLY with JSON: {{"articles": [{{"index": 1, "sentiment": "...", "confidence": 0-100}}, ...]}}\n\n'
        f"{numbered}"
    )


def news_sentiment_agent(state: AgentState, agent_id: str = "news_sentiment_agent"):
    """
    Analyzes news sentiment for a list of tickers and generates trading signals.

    Fetches company news, deduplicates headlines, classifies all candidate articles
    in a single batched LLM call, then aggregates using recency-weighted sentiment
    mass to produce a signal and confidence score for each ticker.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    try:
        ref_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    except Exception:
        ref_dt = datetime.datetime.now(datetime.timezone.utc)

    def _analyze(ticker: str) -> dict:
        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker=ticker, end_date=end_date, limit=100, api_key=api_key) or []

        articles_classified: list[CompanyNews] = []
        sentiment_confidences: dict[str, int] = {}
        sentiments_classified_by_llm = 0

        if company_news:
            pool = _dedupe_articles(company_news[: NEWS_SENTIMENT_MAX_ARTICLES * 2])[:NEWS_SENTIMENT_MAX_ARTICLES]
            to_classify = [a for a in pool if a.sentiment is None]

            if to_classify:
                progress.update_status(agent_id, ticker, f"Classifying {len(to_classify)} articles (batched)")
                prompt = _build_batch_prompt(ticker, to_classify)
                batch = call_llm(prompt, BatchSentiment, agent_name=agent_id, state=state)
                by_index = {r.index: r for r in batch.articles} if batch else {}

                for i, article in enumerate(to_classify, start=1):
                    result = by_index.get(i)
                    if result:
                        article.sentiment = result.sentiment.lower()
                        sentiment_confidences[article.url] = result.confidence
                    else:
                        article.sentiment = "neutral"
                        sentiment_confidences[article.url] = 0
                sentiments_classified_by_llm = len(to_classify)

            articles_classified = [a for a in pool if a.sentiment is not None]

        progress.update_status(agent_id, ticker, "Aggregating signals")

        weighted_bullish = 0.0
        weighted_bearish = 0.0
        weighted_neutral = 0.0
        total_weight = 0.0
        bullish_raw = 0
        bearish_raw = 0
        neutral_raw = 0
        article_payloads: list[dict] = []

        for article in articles_classified:
            age = _article_age_days(article, ref_dt)
            weight = 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)
            if article.sentiment == "positive":
                signal_label = "bullish"
                weighted_bullish += weight
                bullish_raw += 1
            elif article.sentiment == "negative":
                signal_label = "bearish"
                weighted_bearish += weight
                bearish_raw += 1
            else:
                signal_label = "neutral"
                weighted_neutral += weight
                neutral_raw += 1
            total_weight += weight
            article_payloads.append(
                {
                    "title": article.title,
                    "signal": signal_label,
                    "confidence": sentiment_confidences.get(article.url, 0),
                    "age_days": round(age, 2),
                    "weight": round(weight, 4),
                }
            )

        article_payloads.sort(key=lambda x: x["weight"], reverse=True)
        article_payloads = article_payloads[:10]
        total_raw = bullish_raw + bearish_raw + neutral_raw

        if weighted_bullish > weighted_bearish and weighted_bullish > weighted_neutral:
            overall_signal = "bullish"
        elif weighted_bearish > weighted_bullish and weighted_bearish > weighted_neutral:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        confidence = _calculate_confidence_score(
            sentiment_confidences=sentiment_confidences,
            company_news=articles_classified,
            overall_signal=overall_signal,
            bullish_signals=weighted_bullish,
            bearish_signals=weighted_bearish,
            total_signals=total_weight,
        )

        reasoning = {
            "news_sentiment": {
                "signal": overall_signal,
                "confidence": confidence,
                "metrics": {
                    "total_articles": total_raw,
                    "bullish_articles": bullish_raw,
                    "bearish_articles": bearish_raw,
                    "neutral_articles": neutral_raw,
                    "articles_classified_by_llm": sentiments_classified_by_llm,
                    "articles": article_payloads,
                },
            }
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))
        return {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    sentiment_analysis = parallel_per_ticker(tickers, _analyze)

    message = HumanMessage(content=json.dumps(sentiment_analysis), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "News Sentiment Analysis Agent")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def _calculate_confidence_score(sentiment_confidences: dict, company_news: list, overall_signal: str, bullish_signals: float, bearish_signals: float, total_signals: float) -> float:
    """
    Calculate confidence score for a sentiment signal.

    Uses a weighted approach combining LLM confidence scores (70%) with
    signal proportion (30%) when LLM classifications are available.
    bullish_signals, bearish_signals, total_signals may be recency-weighted floats.
    """
    if total_signals == 0:
        return 0.0

    if sentiment_confidences:
        target_sentiment = _SIGNAL_TO_SENTIMENT[overall_signal]
        matching_articles = [news for news in company_news if news.sentiment == target_sentiment]
        llm_confidences = [sentiment_confidences[news.url] for news in matching_articles if news.url in sentiment_confidences]

        if llm_confidences:
            avg_llm_confidence = sum(llm_confidences) / len(llm_confidences)
            signal_proportion = (max(bullish_signals, bearish_signals) / total_signals) * 100
            return round(0.7 * avg_llm_confidence + 0.3 * signal_proportion, 2)

    return round((max(bullish_signals, bearish_signals) / total_signals) * 100, 2)
