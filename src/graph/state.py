import json
import operator
import threading

from langchain_core.messages import BaseMessage
from rich.text import Text
from typing_extensions import Annotated, Sequence, TypedDict

_print_lock = threading.Lock()


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


_SIGNAL_STYLE = {"bullish": "bold green", "bearish": "bold red", "neutral": "bold yellow"}
_ACTION_STYLE = {
    "buy": "bold green",
    "cover": "bold green",
    "sell": "bold red",
    "short": "bold red",
    "hold": "bold yellow",
}


def _flatten_reasoning(reasoning) -> str:
    if isinstance(reasoning, str):
        return reasoning[:120]
    if isinstance(reasoning, dict):
        parts = [f"{k}={v['signal']}" for k, v in reasoning.items() if isinstance(v, dict) and "signal" in v]
        return ", ".join(parts) if parts else str(reasoning)[:120]
    return str(reasoning)[:120]


def _signal_block(data: dict) -> list[Text]:
    rows = []
    for ticker, payload in data.items():
        if not isinstance(payload, dict):
            continue
        signal = payload.get("signal", "")
        confidence = payload.get("confidence", "")
        reasoning = _flatten_reasoning(payload.get("reasoning", ""))
        conf_str = f"{confidence:.0f}%" if isinstance(confidence, (int, float)) else str(confidence)
        style = _SIGNAL_STYLE.get(signal, "")
        t = Text()
        t.append(f"  {ticker:<6}  ", style="default")
        t.append(f"{signal.upper():<8}", style=style)
        t.append(f"  {conf_str:>4}  {reasoning}")
        rows.append(t)
    return rows


def _decision_block(data: dict) -> list[Text]:
    rows = []
    for ticker, payload in data.items():
        if not isinstance(payload, dict):
            continue
        action = payload.get("action", "hold")
        qty = payload.get("quantity", 0)
        confidence = payload.get("confidence", "")
        reasoning = str(payload.get("reasoning", ""))[:100]
        conf_str = f"{confidence:.0f}%" if isinstance(confidence, (int, float)) else str(confidence)
        style = _ACTION_STYLE.get(action, "")
        t = Text()
        t.append(f"  {ticker:<6}  ", style="default")
        t.append(f"{action.upper():<6}", style=style)
        t.append(f"  {qty:>8.2f} shares  {conf_str:>4}  {reasoning}")
        rows.append(t)
    return rows


def _debate_block(data: dict) -> list[Text]:
    rows = []
    for ticker, payload in data.items():
        if not isinstance(payload, dict):
            continue
        rows.append(Text(f"  {ticker}", style="bold"))
        t_bull = Text()
        t_bull.append("    Bull: ", style="bold green")
        t_bull.append(payload.get("bull_case", ""))
        rows.append(t_bull)
        t_bear = Text()
        t_bear.append("    Bear: ", style="bold red")
        t_bear.append(payload.get("bear_case", ""))
        rows.append(t_bear)
        t_crux = Text()
        t_crux.append("    Crux: ", style="bold yellow")
        t_crux.append(payload.get("core_disagreement", ""))
        rows.append(t_crux)
    return rows


def _risk_block(data: dict) -> list[Text]:
    rows = []
    for ticker, payload in data.items():
        if not isinstance(payload, dict):
            continue
        reasoning = payload.get("reasoning", {}) or {}
        vol = (payload.get("volatility_metrics") or {}).get("annualized_volatility", 0.0)
        limit = reasoning.get("position_limit", 0.0)
        remain = reasoning.get("remaining_limit", payload.get("remaining_position_limit", 0.0))
        corr_mult = reasoning.get("correlation_multiplier", 1.0)
        remain_style = "bold red" if remain < 0 else "bold green"
        t = Text()
        t.append(f"  {ticker:<6}  ", style="default")
        t.append(f"LIMIT ${limit:>8,.0f}  ", style="default")
        t.append(f"REMAIN ${remain:>+8,.0f}  ", style=remain_style)
        t.append(f"VOL {vol:.0%}  CORR {corr_mult:.2f}x")
        rows.append(t)
    return rows


def _detect_block_type(data: dict) -> str:
    for payload in data.values():
        if not isinstance(payload, dict):
            return "raw"
        if "signal" in payload:
            return "signal"
        if "action" in payload:
            return "decision"
        if "bull_case" in payload:
            return "debate"
        if "remaining_position_limit" in payload or "volatility_metrics" in payload:
            return "risk"
        return "raw"
    return "raw"


def show_agent_reasoning(output, agent_name: str) -> None:
    # Import here to avoid circular imports at module load time
    from src.utils.progress import progress

    sep = "=" * 48
    header = f"{'=' * 10} {agent_name.center(28)} {'=' * 10}"

    rows: list[Text] = []
    if isinstance(output, dict) and output:
        kind = _detect_block_type(output)
        if kind == "signal":
            rows = _signal_block(output)
        elif kind == "decision":
            rows = _decision_block(output)
        elif kind == "debate":
            rows = _debate_block(output)
        elif kind == "risk":
            rows = _risk_block(output)

    with _print_lock:
        progress.print()
        progress.print(header, style="bold")

        if isinstance(output, dict) and output and not rows:
            progress.print(json.dumps(output, indent=2, default=str))
        elif isinstance(output, str):
            try:
                progress.print(json.dumps(json.loads(output), indent=2))
            except json.JSONDecodeError:
                progress.print(output)
        elif not isinstance(output, dict):
            progress.print(str(output))

        for row in rows:
            progress.print(row)

        progress.print(sep, style="bold")
