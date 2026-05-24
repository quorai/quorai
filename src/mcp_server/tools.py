from __future__ import annotations

from datetime import date, timedelta

from src.llm.request import RunRequest
from src.mcp_server.schemas import (
    AnalystInfo,
    DebateSummary,
    Decision,
    GroupPosition,
    PanelResult,
    RiskAssessment,
    Signal,
)
from src.utils.analysts import ANALYST_CONFIG


def _flatten_reasoning(reasoning) -> str:
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, dict):
        parts = [f"{k}={v['signal']}" for k, v in reasoning.items() if isinstance(v, dict) and "signal" in v]
        return ", ".join(parts) if parts else str(reasoning)
    return str(reasoning)


def parse_dates(start_date: str | None, end_date: str | None) -> tuple[str, str]:
    today = date.today()
    if end_date is None:
        end = today
    else:
        try:
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise ValueError(f"Invalid end_date {end_date!r}: must be YYYY-MM-DD") from exc
    if start_date is None:
        start = end - timedelta(days=30)
    else:
        try:
            start = date.fromisoformat(start_date)
        except ValueError as exc:
            raise ValueError(f"Invalid start_date {start_date!r}: must be YYYY-MM-DD") from exc
    if start > end:
        raise ValueError(f"start_date {start} must be <= end_date {end}")
    return str(start), str(end)


def validate_analyst_keys(keys: list[str]) -> None:
    invalid = [k for k in keys if k not in ANALYST_CONFIG]
    if invalid:
        raise ValueError(f"Unknown analyst key(s): {invalid}. Valid keys: {sorted(ANALYST_CONFIG)}")


def build_portfolio(tickers: list[str], initial_cash: float, margin_requirement: float = 0.25) -> dict:
    return {
        "cash": initial_cash,
        "margin_requirement": margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
    }


def build_request(agent_models: dict[str, list[str]] | None) -> RunRequest | None:
    """Convert MCP agent_models dict to a RunRequest.

    MCP keys are analyst keys (without _agent suffix) or '*'; values are [model, provider].
    RunRequest expects full node names (with _agent suffix) as keys.
    """
    if not agent_models:
        return None
    node_models: dict[str, tuple[str, str]] = {}
    for key, spec in agent_models.items():
        node_key = key if key == "*" else f"{key}_agent"
        node_models[node_key] = (str(spec[0]), str(spec[1]))
    return RunRequest(agent_models=node_models)


def _build_signal(payload: dict) -> Signal:
    return Signal(
        signal=payload.get("signal", "neutral"),
        confidence=payload.get("confidence", 0),
        reasoning=_flatten_reasoning(payload.get("reasoning", "")),
    )


def _build_decision(payload: dict) -> Decision:
    return Decision(
        action=payload.get("action", "hold"),
        quantity=float(payload.get("quantity", 0)),
        confidence=payload.get("confidence", 0),
        reasoning=str(payload.get("reasoning", "")),
    )


def _build_debate_summary(payload: dict) -> DebateSummary:
    group_positions = None
    if payload.get("group_positions") is not None:
        group_positions = [
            GroupPosition(
                group=gp.get("group", ""),
                stance=gp.get("stance", ""),
                key_argument=gp.get("key_argument"),
            )
            for gp in payload["group_positions"]
        ]
    return DebateSummary(
        consensus_strength=payload.get("consensus_strength"),
        core_disagreement=payload.get("core_disagreement"),
        group_positions=group_positions,
        bull_case=payload.get("bull_case"),
        bear_case=payload.get("bear_case"),
    )


def _build_risk_assessment(payload: dict) -> RiskAssessment:
    reasoning = payload.get("reasoning") or {}
    vol_metrics = payload.get("volatility_metrics") or {}
    return RiskAssessment(
        position_limit=reasoning.get("position_limit"),
        remaining_limit=reasoning.get("remaining_limit", payload.get("remaining_position_limit")),
        annualized_volatility=vol_metrics.get("annualized_volatility"),
        correlation_multiplier=reasoning.get("correlation_multiplier"),
    )


def _build_markdown_summary(
    tickers: list[str],
    window: dict[str, str],
    decisions: dict[str, Decision],
    signals: dict[str, dict[str, Signal]],
    debate_summaries: dict[str, DebateSummary],
) -> str:
    lines = [
        f"## Quorai Panel: {', '.join(tickers)}",
        f"**Period:** {window['start']} → {window['end']}",
        "",
        "### Portfolio Decisions",
    ]
    if decisions:
        for ticker, d in decisions.items():
            conf = f"{d.confidence:.0f}%" if isinstance(d.confidence, (int, float)) else str(d.confidence)
            lines.append(f"- **{ticker}**: {d.action.upper()} {d.quantity:.0f} shares (confidence: {conf})")
    else:
        lines.append("- No decisions available")
    lines.append("")

    for ticker in tickers:
        lines.append(f"### Analyst Signals — {ticker}")
        for agent_key, ticker_signals in signals.items():
            if ticker not in ticker_signals:
                continue
            s = ticker_signals[ticker]
            display = agent_key.replace("_agent", "").replace("_", " ").title()
            conf = f"{s.confidence:.0f}%" if isinstance(s.confidence, (int, float)) else str(s.confidence)
            lines.append(f"- **{display}**: {s.signal.upper()} ({conf})")
        if ticker in debate_summaries:
            db = debate_summaries[ticker]
            if db.consensus_strength:
                lines.append(f"\n**Consensus:** {db.consensus_strength.replace('_', ' ')}")
            if db.core_disagreement:
                lines.append(f"**Key disagreement:** {db.core_disagreement}")
        lines.append("")

    return "\n".join(lines)


def build_panel_result(raw: dict, tickers: list[str], start_date: str, end_date: str) -> PanelResult:
    decisions_raw = raw.get("decisions") or {}
    portfolio_decisions = {ticker: _build_decision(payload) for ticker, payload in decisions_raw.items() if isinstance(payload, dict)}

    raw_signals: dict = raw.get("analyst_signals", {})
    risk_raw = raw_signals.get("risk_management_agent", {})
    risk_assessments = {ticker: _build_risk_assessment(payload) for ticker, payload in risk_raw.items() if isinstance(payload, dict)}

    analyst_signals: dict[str, dict[str, Signal]] = {agent_id: {ticker: _build_signal(payload) for ticker, payload in ticker_signals.items() if isinstance(payload, dict)} for agent_id, ticker_signals in raw_signals.items() if agent_id != "risk_management_agent" and isinstance(ticker_signals, dict)}

    debate_summaries = {ticker: _build_debate_summary(payload) for ticker, payload in raw.get("debate_summaries", {}).items() if isinstance(payload, dict)}

    window = {"start": start_date, "end": end_date}
    markdown_summary = _build_markdown_summary(tickers, window, portfolio_decisions, analyst_signals, debate_summaries)

    return PanelResult(
        tickers=tickers,
        window=window,
        portfolio_decisions=portfolio_decisions,
        analyst_signals=analyst_signals,
        group_signals=raw.get("group_signals", {}),
        debate_summaries=debate_summaries,
        risk_assessments=risk_assessments,
        current_prices=raw.get("current_prices", {}),
        markdown_summary=markdown_summary,
    )


def list_analysts_impl() -> list[AnalystInfo]:
    return [
        AnalystInfo(
            key=key,
            display_name=cfg["display_name"],
            description=cfg["description"],
            investing_style=cfg["investing_style"],
            pull_quote=cfg["pull_quote"],
            strategy_group=cfg["strategy_group"],
            order=cfg["order"],
        )
        for key, cfg in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]


def get_analyst_info_impl(analyst_key: str) -> AnalystInfo:
    if analyst_key not in ANALYST_CONFIG:
        raise ValueError(f"Unknown analyst key {analyst_key!r}. Valid keys: {sorted(ANALYST_CONFIG)}")
    cfg = ANALYST_CONFIG[analyst_key]
    return AnalystInfo(
        key=analyst_key,
        display_name=cfg["display_name"],
        description=cfg["description"],
        investing_style=cfg["investing_style"],
        pull_quote=cfg["pull_quote"],
        strategy_group=cfg["strategy_group"],
        order=cfg["order"],
    )
