import json

from langchain_core.messages import HumanMessage
import pandas as pd

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_insider_trades
from src.utils.api_key import get_api_key_from_state
from src.utils.concurrency import parallel_per_ticker
from src.utils.progress import progress


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes insider trading activity and generates trading signals for multiple tickers.

    News sentiment is handled separately by news_sentiment_agent.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    def _analyze(ticker: str) -> dict:
        progress.update_status(agent_id, ticker, "Fetching insider trades")

        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        bearish_count = int((transaction_shares < 0).sum())
        bullish_count = int((transaction_shares >= 0).sum())
        total_signals = len(transaction_shares)

        if bullish_count > bearish_count:
            overall_signal = "bullish"
        elif bearish_count > bullish_count:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        confidence = 0.0
        if total_signals > 0:
            confidence = round((max(bullish_count, bearish_count) / total_signals) * 100, 2)

        reasoning = {
            "insider_trading": {
                "signal": overall_signal,
                "confidence": confidence,
                "metrics": {
                    "total_trades": total_signals,
                    "bullish_trades": bullish_count,
                    "bearish_trades": bearish_count,
                },
            },
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))
        return {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    sentiment_analysis = parallel_per_ticker(tickers, _analyze)

    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
