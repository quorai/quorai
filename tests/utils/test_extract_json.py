"""Tests for extract_json_from_response and _regex_field_extract."""

from src.agents._signals import BaseSignal
from src.utils.llm import _regex_field_extract, extract_json_from_response

# ---------------------------------------------------------------------------
# extract_json_from_response
# ---------------------------------------------------------------------------


def test_plain_json():
    raw = '{"signal": "bullish", "confidence": 75, "reasoning": "- Strong FCF"}'
    result = extract_json_from_response(raw)
    assert result == {"signal": "bullish", "confidence": 75, "reasoning": "- Strong FCF"}


def test_fenced_json_block():
    raw = '```json\n{"signal": "neutral", "confidence": 50, "reasoning": "- Mixed"}\n```'
    result = extract_json_from_response(raw)
    assert result is not None
    assert result["signal"] == "neutral"


def test_bullet_as_stray_key_truncated():
    """Local LLM truncates JSON mid-key after emitting bullet points as stray keys.

    The actual Mohnish Pabrai failure: reasoning ends at a semicolon, subsequent bullet
    lines start as new JSON keys, and the output is cut off before the closing brace.
    The regex fallback should recover signal, confidence, and the partial reasoning.
    """
    raw = (
        "{\n"
        '  "signal": "neutral",\n'
        '  "confidence": 40,\n'
        '  "reasoning": "- Negative FCF yield fails margin of safety;",\n'
        '  "- High valuation and asset-li'  # truncated mid-key — json.loads fails
    )
    result = extract_json_from_response(raw)
    assert result is not None
    assert result["signal"] == "neutral"
    assert result["confidence"] == 40
    assert "Negative FCF" in result["reasoning"]


def test_unparseable_returns_none():
    result = extract_json_from_response("This is pure prose with no JSON at all.")
    assert result is None


# ---------------------------------------------------------------------------
# _regex_field_extract
# ---------------------------------------------------------------------------


def test_regex_extract_full():
    content = '"signal": "bearish", "confidence": 30, "reasoning": "- Low moat;", "- Expensive": ""'
    result = _regex_field_extract(content)
    assert result["signal"] == "bearish"
    assert result["confidence"] == 30
    assert "Low moat" in result["reasoning"]
    assert "Expensive" in result["reasoning"]


def test_regex_extract_returns_none_with_too_few_fields():
    result = _regex_field_extract('"signal": "bullish"')
    assert result is None


# ---------------------------------------------------------------------------
# BaseSignal — missing reasoning field should not raise
# ---------------------------------------------------------------------------


def test_base_signal_missing_reasoning_defaults_to_empty():
    sig = BaseSignal(signal="neutral", confidence=50)
    assert sig.reasoning == ""


def test_base_signal_reasoning_list_coerced_to_str():
    sig = BaseSignal(signal="bullish", confidence=80, reasoning=["- Bullet one", "- Bullet two"])
    assert "Bullet one" in sig.reasoning
    assert "Bullet two" in sig.reasoning
