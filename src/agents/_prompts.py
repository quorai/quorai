import json

from langchain_core.prompts import ChatPromptTemplate

_HUMAN = 'Ticker: {ticker}\nFacts:\n{facts}\n\nReturn exactly:\n{{\n  "signal": "bullish" | "bearish" | "neutral",\n  "confidence": int (0-100),\n  "reasoning": "short bullet list, 2–3 bullets preferred, max 5"\n}}'


def build_persona_prompt(persona_voice: str, facts: dict, ticker: str):
    """Build a compact LLM prompt from a stable system message + variable facts.

    The system message is stable across tickers (prompt-cache friendly).
    Facts are serialised as compact JSON.
    """
    template = ChatPromptTemplate.from_messages(
        [
            ("system", persona_voice),
            ("human", _HUMAN),
        ]
    )
    return template.invoke(
        {
            "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
            "ticker": ticker,
        }
    )
