from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os

from colorama import init
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.debate_node import debate_node
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.cli.input import (
    parse_cli_inputs,
)
from src.graph.state import AgentState
from src.llm.request import RunRequest
from src.utils.analysts import ANALYST_CONFIG, get_analyst_nodes
from src.utils.display import print_trading_output
from src.utils.progress import progress

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)

logger = logging.getLogger(__name__)


def parse_quorai_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning("JSON decoding error: %s\nResponse: %r", e, response)
        return None
    except TypeError as e:
        logger.warning("Invalid response type (expected string, got %s): %s", type(response).__name__, e)
        return None
    except Exception:
        logger.exception("Unexpected error parsing Quorai response")
        raise


##### Run Quorai #####
def run_quorai(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] | None = None,
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
    llm_temperature: float | None = None,
    conviction_weights: dict[str, float] | None = None,
    request: RunRequest | None = None,
):
    if selected_analysts is None:
        selected_analysts = []
    # Start progress tracking
    progress.start()

    try:
        workflow = create_workflow()
        agent = workflow.compile()

        # selected_analysts is forwarded through metadata so parallel_analysts_node
        # can resolve the analyst list at runtime without it being baked into the graph.
        effective_analysts = selected_analysts if selected_analysts else list(ANALYST_CONFIG.keys())

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "llm_temperature": llm_temperature,
                    "conviction_weights": conviction_weights or {},
                    "request": request,
                    "selected_analysts": effective_analysts,
                },
            },
        )

        return {
            "decisions": parse_quorai_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def parallel_analysts_node(state: AgentState) -> dict:
    """Run all selected analyst agents, concurrently when QUORAI_PARALLEL_ANALYSTS > 1.

    Reads selected_analysts from state["metadata"]["selected_analysts"]. Each analyst
    function mutates state["data"]["analyst_signals"] in place (distinct keys per agent),
    so concurrent writes are safe under the CPython GIL.
    """
    selected = state["metadata"].get("selected_analysts") or list(ANALYST_CONFIG.keys())
    analyst_nodes_map = get_analyst_nodes()
    max_workers = int(os.environ.get("QUORAI_PARALLEL_ANALYSTS", "8"))

    funcs = [(key, analyst_nodes_map[key][1]) for key in selected if key in analyst_nodes_map]

    if max_workers <= 1 or len(funcs) <= 1:
        messages = []
        for _, fn in funcs:
            result = fn(state)
            messages.extend(result.get("messages", []))
    else:
        effective_workers = min(max_workers, len(funcs))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(fn, state) for _, fn in funcs]
        messages = []
        for fut in futures:
            messages.extend(fut.result().get("messages", []))

    return {"messages": messages, "data": state["data"]}


def create_workflow():
    """Create the workflow. Analyst selection is handled at runtime by parallel_analysts_node."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)
    workflow.add_node("parallel_analysts_node", parallel_analysts_node)
    workflow.add_node("debate_node", debate_node)
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    workflow.add_edge("start_node", "parallel_analysts_node")
    workflow.add_edge("parallel_analysts_node", "debate_node")
    workflow.add_edge("debate_node", "risk_management_agent")
    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the Quorai trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Construct portfolio here
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
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
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
            for ticker in tickers
        },
    }

    result = run_quorai(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
    )
    print_trading_output(result)
