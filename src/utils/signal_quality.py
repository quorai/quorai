"""Detect analyst signals that are confidently directional despite missing input data."""

from __future__ import annotations

# Phrases are specific enough to avoid false positives on clean reasoning that
# happens to contain words like "missing" in a non-data-absence context.
# Match is always case-insensitive.
_MISSING_DATA_PHRASES = (
    "n/a",
    "insufficient data",
    "insufficient information",
    "missing data",
    "data missing",
    "data is missing",
    "data unavailable",
    "data is unavailable",
    "missing critical",
    "missing key",
    "cannot compute",
    "cannot be computed",
    "cannot calculate",
    "cannot be calculated",
    "cannot assess",
    "not provided",
    "lack of data",
    "lacking data",
    "impossible without",
    "incomplete data",
    "incomplete revenue",
    "incomplete information",
    "sparse data",
)


def _extract_reasoning_text(reasoning: object) -> str:
    """Recursively flatten a reasoning value into a single searchable string.

    Unlike _flatten_reasoning in src/graph/state.py this preserves `details`
    text nested inside dict-form reasoning (e.g. fundamentals_analyst_agent),
    where "N/A" / "unavailable" markers actually live.
    """
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, dict):
        return " ".join(_extract_reasoning_text(v) for v in reasoning.values())
    if isinstance(reasoning, (list, tuple)):
        return " ".join(_extract_reasoning_text(v) for v in reasoning)
    return str(reasoning)


def data_quality_multiplier(sig_data: dict) -> float:
    """Return a weight multiplier for a signal based on data-quality markers in its reasoning.

    Returns one of {0.0, 0.5, 1.0}:

    Rules applied in order:
    1. count >= 2 missing-data phrases anywhere in reasoning → 0.0.
       Catches analysts whose multiple sub-fields all report "N/A" — signal is
       fundamentally broken and should be excluded entirely.
    2. count >= 1 AND signal != "neutral" → 0.5 (partial discount).
       Directional call backed by a single "data missing" phrase is down-weighted
       but not erased — preserves its directional contribution at half strength.
    3. Else → 1.0 (clean reasoning, or a neutral self-flagging analyst that
       contributes zero stance-score regardless).
    """
    if not isinstance(sig_data, dict):
        return 1.0
    text = _extract_reasoning_text(sig_data.get("reasoning", "")).lower()
    if not text:
        return 1.0
    # Count total occurrences across all phrases (not just distinct phrase types).
    # "ROE: N/A, Net Margin: N/A" → 2 occurrences of "n/a" → count=2 → rule 1.
    # "Missing critical data for analysis." → 1 occurrence → count=1 → rule 2.
    count = sum(text.count(phrase) for phrase in _MISSING_DATA_PHRASES)
    if count >= 2:
        return 0.0
    if count >= 1 and str(sig_data.get("signal", "")).lower() != "neutral":
        return 0.5
    return 1.0
