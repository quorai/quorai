from typing import Literal

from pydantic import BaseModel, Field


class BaseSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0, le=100)
    reasoning: str
