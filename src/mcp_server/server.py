from __future__ import annotations

import asyncio
import logging

from mcp.server.fastmcp import Context, FastMCP

from src.mcp_server.schemas import AnalystInfo, PanelResult, Signal
from src.mcp_server.tools import (
    build_panel_result,
    build_portfolio,
    build_request,
    get_analyst_info_impl,
    list_analysts_impl,
    parse_dates,
    validate_analyst_keys,
)
from src.risk_profiles import get_profile

logger = logging.getLogger(__name__)

mcp = FastMCP("Quorai")

_panel_lock = asyncio.Lock()


@mcp.tool()
def list_analysts() -> list[AnalystInfo]:
    """List all 25 available analyst personas with investing styles and strategy groups."""
    return list_analysts_impl()


@mcp.tool()
def get_analyst_info(analyst_key: str) -> AnalystInfo:
    """Get full metadata for a single analyst persona.

    Args:
        analyst_key: Analyst identifier, e.g. 'warren_buffett'. Call list_analysts to see all valid keys.
    """
    return get_analyst_info_impl(analyst_key)


@mcp.tool()
async def run_panel(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    analysts: list[str] | None = None,
    agent_models: dict[str, list[str]] | None = None,
    initial_cash: float = 100000.0,
    risk_profile: str | None = None,
    show_reasoning: bool = False,
    ctx: Context = None,
) -> PanelResult:
    """Run the full Quorai analyst panel for the given tickers.

    Each of the 25 analysts produces per-ticker signals (bullish/bearish/neutral).
    A debate node aggregates signals by strategy group, a risk manager gates position
    sizes, and the portfolio manager issues the final buy/sell/hold decisions.

    This tool runs for 2–5 minutes depending on the number of tickers and analysts.
    Progress is reported at each pipeline stage.

    Args:
        tickers: Stock symbols to analyse, e.g. ['AAPL', 'MSFT'].
        start_date: Lookback start in YYYY-MM-DD format. Defaults to 30 days before end_date.
        end_date: Lookback end in YYYY-MM-DD format. Defaults to today.
        analysts: Optional subset of analyst keys to run (see list_analysts). Runs all 25 if omitted.
        agent_models: Per-agent model override. Keys are analyst keys or '*' (wildcard). Values are
            [model_slug, provider], e.g. {'*': ['nousresearch/hermes-4-70b', 'OpenRouter']}.
        initial_cash: Starting portfolio cash in USD. Default 100 000.
        risk_profile: Position-sizing profile. One of: conservative, cautious, balanced, aggressive, speculative.
        show_reasoning: Include per-analyst reasoning text in results.
    """
    if not tickers:
        raise ValueError("tickers must not be empty")

    start, end = parse_dates(start_date, end_date)

    if analysts:
        validate_analyst_keys(analysts)

    risk_profile_obj = get_profile(risk_profile) if risk_profile else None
    portfolio = build_portfolio(tickers, initial_cash)
    request = build_request(agent_models)

    if ctx:
        await ctx.report_progress(0, 4, "Queuing analyst panel…")

    async with _panel_lock:
        if ctx:
            await ctx.report_progress(1, 4, f"Running analyst panel for {', '.join(tickers)} ({start} → {end})…")

        from src.main import run_quorai  # noqa: PLC0415 — deferred so load_dotenv() runs first

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: run_quorai(
                tickers=tickers,
                start_date=start,
                end_date=end,
                portfolio=portfolio,
                show_reasoning=show_reasoning,
                selected_analysts=analysts or [],
                request=request,
                risk_profile=risk_profile_obj,
            ),
        )

    if ctx:
        await ctx.report_progress(4, 4, "Panel complete")

    return build_panel_result(raw, tickers, start, end)


@mcp.tool()
async def run_single_analyst(
    analyst_key: str,
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    agent_model: list[str] | None = None,
    ctx: Context = None,
) -> dict[str, Signal]:
    """Run a single analyst persona and return its per-ticker signals.

    Note: the full graph (debate, risk, portfolio) still runs internally because
    Quorai's pipeline cannot short-circuit at a single analyst. Only the requested
    analyst's signals are returned. Use run_panel for the complete output.

    Args:
        analyst_key: Analyst identifier, e.g. 'warren_buffett'. Call list_analysts to see all valid keys.
        tickers: Stock symbols to analyse.
        start_date: Lookback start in YYYY-MM-DD format.
        end_date: Lookback end in YYYY-MM-DD format.
        agent_model: Optional [model_slug, provider] for this analyst,
            e.g. ['nousresearch/hermes-4-405b', 'OpenRouter'].
    """
    agent_models = {analyst_key: agent_model} if agent_model else None
    result = await run_panel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        analysts=[analyst_key],
        agent_models=agent_models,
        ctx=ctx,
    )
    return result.analyst_signals.get(f"{analyst_key}_agent", {})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
