from typing import Literal

from pydantic import BaseModel, Field


class BaseSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(ge=0, le=100)
    reasoning: str = Field(description=("Short hyphen-bulleted list (one '- ' bullet per line). Prefer 2-3 bullets, max 5. Each bullet under ~100 chars. No prose paragraphs."))
