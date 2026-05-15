import json
import logging
import math

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, _flatten_reasoning, show_agent_reasoning
from src.utils.analysts import get_agent_to_group
from src.utils.concurrency import parallel_per_ticker
from src.utils.llm import call_llm
from src.utils.progress import progress

logger = logging.getLogger(__name__)

_STANCE_SCORE = {"bullish": 1, "bearish": -1, "neutral": 0}


def _aggregate_to_groups(
    analyst_signals: dict[str, dict],
    tickers: list[str],
    agent_weights: dict[str, float] | None = None,
) -> dict[str, dict[str, dict]]:
    """Deterministically collapse 25 individual signals into per-group stances.

    Returns: {ticker: {group: {signal, confidence, dissent, key_args}}}
    """
    agent_to_group = get_agent_to_group()

    # Accumulate per (ticker, group)
    raw: dict[str, dict[str, list[tuple[str, str, float, str]]]] = {t: {} for t in tickers}
    for agent, ticker_map in analyst_signals.items():
        if agent.startswith("risk_management_agent"):
            continue
        group = agent_to_group.get(agent)
        if group is None:
            continue
        agent_label = agent.replace("_agent", "")
        for ticker, sig_data in ticker_map.items():
            if ticker not in raw:
                continue
            signal = sig_data.get("signal", "neutral")
            raw_conf = float(sig_data.get("confidence") or 0.0)
            if math.isnan(raw_conf):
                raw_conf = 0.0
            raw_conf = max(0.0, min(100.0, raw_conf))
            w = (agent_weights or {}).get(agent_label, 1.0)
            confidence = raw_conf * max(0.0, w)
            reasoning = _flatten_reasoning(sig_data.get("reasoning", ""))
            raw[ticker].setdefault(group, []).append((agent_label, signal, confidence, reasoning))

    # Aggregate each group
    result: dict[str, dict[str, dict]] = {}
    for ticker in tickers:
        result[ticker] = {}
        for group, members in raw[ticker].items():
            total_weight = sum(conf for _, _, conf, _ in members)
            if total_weight == 0:
                weighted_stance = 0.0
                avg_conf = 0.0
            else:
                unknown = [sig for _, sig, _, _ in members if sig not in _STANCE_SCORE]
                if unknown:
                    logger.warning("Unknown signal value(s) in debate aggregation, treating as neutral: %s", unknown)
                weighted_stance = sum(_STANCE_SCORE.get(sig, 0) * conf for _, sig, conf, _ in members) / total_weight
                avg_conf = total_weight / len(members)

            if weighted_stance >= 0.25:
                group_signal = "bullish"
            elif weighted_stance <= -0.25:
                group_signal = "bearish"
            else:
                group_signal = "neutral"

            dissent = sum(1 for _, sig, _, _ in members if sig != group_signal and sig != "neutral")

            # Top-2 members by confidence for the moderator prompt
            sorted_members = sorted(members, key=lambda x: x[2], reverse=True)
            key_args = [f"{label}: {reasoning[:100]}" for label, _, _, reasoning in sorted_members[:2] if reasoning]

            result[ticker][group] = {
                "signal": group_signal,
                "confidence": round(avg_conf, 1),
                "dissent": dissent,
                "key_args": key_args,
            }

    return result


class GroupStance(BaseModel):
    group: str
    stance: Literal["bullish", "bearish", "neutral"]
    key_argument: str = Field(description="One-sentence summary of this group's strongest argument, ≤120 chars")


class DebateSummary(BaseModel):
    group_positions: list[GroupStance]
    core_disagreement: str = Field(description="Root structural disagreement across groups, ≤140 chars")
    consensus_strength: Literal["strong_agreement", "mixed", "structural_split"]


_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a neutral debate moderator for investment strategy groups. "
            "Each group represents a distinct school of thought. Given their aggregated stances and key arguments for a stock, "
            "summarise each group's position in one sentence, identify the root structural disagreement, "
            "and rate the overall consensus. Be concise.",
        ),
        (
            "human",
            (
                "Ticker: {ticker}\n\n"
                "Group stances:\n{group_table}\n\n"
                "Return ONLY a JSON object — no prose, no markdown fences.\n"
                'Keys: "group_positions" (array of {{"group":str,"stance":str,"key_argument":str≤120}}), '
                '"core_disagreement" (str≤140), "consensus_strength" ("strong_agreement"|"mixed"|"structural_split").\n'
                'Example: {{"group_positions":[{{"group":"deep_value","stance":"bullish","key_argument":"..."}}],'
                '"core_disagreement":"...","consensus_strength":"structural_split"}}'
            ),
        ),
    ]
)


def debate_node(state: AgentState) -> dict:
    """Aggregate analyst signals by strategy group, then synthesise a group-level debate per contested ticker."""
    analyst_signals = state["data"].get("analyst_signals", {})
    tickers = state["data"]["tickers"]
    conviction_weights = state.get("metadata", {}).get("conviction_weights") or {}

    group_signals = _aggregate_to_groups(analyst_signals, tickers, agent_weights=conviction_weights)

    # Only debate contested tickers (at least one bullish AND one bearish group)
    contested = [t for t in tickers if any(d["signal"] == "bullish" for d in group_signals.get(t, {}).values()) and any(d["signal"] == "bearish" for d in group_signals.get(t, {}).values())]

    def _debate_ticker(ticker: str) -> dict | None:
        groups = group_signals.get(ticker, {})
        progress.update_status("debate_node", ticker, "Synthesising group debate")

        rows = []
        for group, data in groups.items():
            dissent_note = f" (internal dissent: {data['dissent']})" if data["dissent"] else ""
            args_str = "; ".join(data["key_args"]) if data["key_args"] else "no data"
            rows.append(f"- {group} [{data['signal']}, conf {data['confidence']:.0f}{dissent_note}]: {args_str}")
        group_table = "\n".join(rows)

        prompt = _TEMPLATE.invoke({"ticker": ticker, "group_table": group_table})
        try:
            summary = call_llm(
                prompt=prompt,
                pydantic_model=DebateSummary,
                agent_name="debate_node",
                state=state,
            )
            return {
                "group_positions": [gp.model_dump() for gp in summary.group_positions],
                "core_disagreement": summary.core_disagreement,
                "consensus_strength": summary.consensus_strength,
            }
        except Exception:
            logger.exception("Debate synthesis failed for %s; skipping", ticker)
            return None

    raw_results = parallel_per_ticker(contested, _debate_ticker)
    debate_summaries: dict[str, dict] = {t: v for t, v in raw_results.items() if v is not None}

    if state["metadata"].get("show_reasoning") and debate_summaries:
        show_agent_reasoning(debate_summaries, "Debate Node")

    progress.update_status("debate_node", None, "Done")

    return {
        "messages": [HumanMessage(content=json.dumps(debate_summaries), name="debate_node")],
        "data": {"debate_summaries": debate_summaries, "group_signals": group_signals},
    }
