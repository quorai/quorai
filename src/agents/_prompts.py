import json

from langchain_core.prompts import ChatPromptTemplate

_HUMAN = 'Ticker: {ticker}\nFacts:\n{facts}\n\nReturn exactly:\n{{\n  "signal": "bullish" | "bearish" | "neutral",\n  "confidence": int (0-100),\n  "reasoning": "short bullet list, 2–3 bullets preferred, max 5"\n}}'

# Appended to every agent system prompt.  Keeps agents from converting data
# gaps into fake-bearish signals that pollute the panel and trigger the gate.
_MISSING_DATA_GUARD = "\n\nData-absence rule: if the key metrics you need are predominantly None or missing, return signal='neutral' and confidence=0 — never emit 'bearish' solely because numbers are absent.  Reserve directional calls for when the data exists and is genuinely positive or negative."


def build_persona_prompt(persona_voice: str, facts: dict, ticker: str):
    """Build a compact LLM prompt from a stable system message + variable facts.

    The system message is stable across tickers (prompt-cache friendly).
    Facts are serialised as compact JSON.
    The missing-data guard is appended to every persona so agents never emit
    'bearish' solely because data fields are None.
    """
    template = ChatPromptTemplate.from_messages(
        [
            ("system", persona_voice + _MISSING_DATA_GUARD),
            ("human", _HUMAN),
        ]
    )
    return template.invoke(
        {
            "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
            "ticker": ticker,
        }
    )
