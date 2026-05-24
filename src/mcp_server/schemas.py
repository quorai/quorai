from __future__ import annotations

from pydantic import BaseModel


class AnalystInfo(BaseModel):
    key: str
    display_name: str
    description: str
    investing_style: str
    pull_quote: str
    strategy_group: str
    order: int


class Signal(BaseModel):
    signal: str
    confidence: float | int | str
    reasoning: str


class Decision(BaseModel):
    action: str
    quantity: float
    confidence: float | int | str
    reasoning: str


class GroupPosition(BaseModel):
    group: str
    stance: str
    key_argument: str | None = None


class DebateSummary(BaseModel):
    consensus_strength: str | None = None
    core_disagreement: str | None = None
    group_positions: list[GroupPosition] | None = None
    bull_case: str | None = None
    bear_case: str | None = None


class RiskAssessment(BaseModel):
    position_limit: float | None = None
    remaining_limit: float | None = None
    annualized_volatility: float | None = None
    correlation_multiplier: float | None = None


class PanelResult(BaseModel):
    tickers: list[str]
    window: dict[str, str]
    portfolio_decisions: dict[str, Decision]
    analyst_signals: dict[str, dict[str, Signal]]
    group_signals: dict
    debate_summaries: dict[str, DebateSummary]
    risk_assessments: dict[str, RiskAssessment]
    current_prices: dict[str, float]
    markdown_summary: str
