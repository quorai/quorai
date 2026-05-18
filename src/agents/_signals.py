from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BaseSignal(BaseModel):
    signal: Literal["neutral", "bullish", "bearish"]
    confidence: int = Field(ge=0, le=100)
    reasoning: str = Field(description=("Short hyphen-bulleted list (one '- ' bullet per line). Prefer 2-3 bullets, max 5. Each bullet under ~100 chars. No prose paragraphs."))

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning_to_str(cls, v):
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)):
            return "\n".join(str(x) for x in v)
        return str(v)
