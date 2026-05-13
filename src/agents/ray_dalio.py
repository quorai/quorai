# SPDX-License-Identifier: MIT
import json

from langchain_core.messages import HumanMessage

from src.agents._data_bundle import AgentDataBundle
from src.agents._prompts import build_persona_prompt
from src.agents._signals import BaseSignal
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.concurrency import parallel_per_ticker
from src.utils.llm import call_llm
from src.utils.progress import progress

_PERSONA = (
    "You are Ray Dalio. Decide bullish, bearish, or neutral using only the provided facts.\n"
    "\n"
    "Framework:\n"
    "- Debt cycle: where is the company in its debt cycle? High leverage + rising = late cycle danger\n"
    "- Economic regime: growth × inflation quadrant — expansion or contraction?\n"
    "- Balance sheet resilience: can it survive the bad times without credit markets?\n"
    "- Valuation as cycle indicator: expensive = late cycle greed, cheap = early cycle fear\n"
    "\n"
    "Signal rules:\n"
    "- Bullish: early/mid debt cycle + expansion regime + resilient balance sheet + fair or cheap valuation\n"
    "- Bearish: late-cycle debt trap OR contraction regime OR fragile balance sheet with stretched valuation\n"
    "- Neutral: mixed signals or transitional cycle position\n"
    "\n"
    "Confidence scale:\n"
    "- 90-100%: Clear early/mid cycle, strong balance sheet, attractive valuation\n"
    "- 70-89%: Mostly positive cycle and balance sheet signals\n"
    "- 50-69%: Mixed or transitional\n"
    "- 30-49%: Late-cycle risks present but not severe\n"
    "- 10-29%: Debt trap, contraction regime, or fragile + expensive\n"
    "\n"
    "Use Dalio's vocabulary: debt cycle, deleveraging, beautiful/ugly deleveraging, economic machine, risk-parity, all-weather, regime.\n"
    'Write reasoning as a short bullet list (2–3 bullets preferred, max 5). Each bullet: "- " + one fact or judgment under ~100 chars. Stay in Dalio\'s economic-machine voice. Do not invent data. Return JSON only.'
)


def ray_dalio_agent(state: AgentState, agent_id: str = "ray_dalio_agent"):
    """Analyzes stocks using Dalio's debt-cycle, economic-regime, and risk-parity framework."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    def _analyze(ticker: str) -> dict:
        progress.update_status(agent_id, ticker, "Fetching financial data")
        bundle = AgentDataBundle.fetch(
            ticker,
            end_date,
            line_item_names=[
                "total_debt",
                "total_assets",
                "shareholders_equity",
                "net_income",
                "revenue",
                "operating_income",
                "interest_expense",
                "free_cash_flow",
            ],
            line_item_period="ttm",
            line_item_limit=5,
            metrics_period="ttm",
            metrics_limit=5,
            api_key=api_key,
        )
        metrics = bundle.financial_metrics
        line_items = bundle.line_items
        market_cap = bundle.market_cap

        progress.update_status(agent_id, ticker, "Analyzing debt cycle")
        debt_cycle = analyze_debt_cycle(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing economic regime")
        economic_regime = analyze_economic_regime(metrics)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet resilience")
        balance_sheet = analyze_balance_sheet_resilience(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing valuation as cycle indicator")
        valuation_cycle = analyze_valuation_cycle(metrics, market_cap)

        total_score = debt_cycle["score"] + economic_regime["score"] + balance_sheet["score"] + valuation_cycle["score"]
        max_possible_score = debt_cycle["max_score"] + economic_regime["max_score"] + balance_sheet["max_score"] + valuation_cycle["max_score"]

        ticker_analysis = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "debt_cycle": debt_cycle,
            "economic_regime": economic_regime,
            "balance_sheet": balance_sheet,
            "valuation_cycle": valuation_cycle,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Ray Dalio analysis")
        dalio_output = generate_dalio_output(
            ticker=ticker,
            analysis_data=ticker_analysis,
            state=state,
            agent_id=agent_id,
        )

        progress.update_status(agent_id, ticker, "Done", analysis=dalio_output.reasoning)
        return {
            "signal": dalio_output.signal,
            "confidence": dalio_output.confidence,
            "reasoning": dalio_output.reasoning,
        }

    dalio_analysis = parallel_per_ticker(tickers, _analyze)

    message = HumanMessage(content=json.dumps(dalio_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(dalio_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = dalio_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_debt_cycle(metrics: list, line_items: list) -> dict:
    """Score debt-cycle position: expanding debt = late cycle risk, deleveraging = recovery opportunity."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # Interest coverage — Dalio: low coverage = debt trap / end of cycle
    if latest.interest_coverage:
        if latest.interest_coverage > 5:
            score += 2
            details.append(f"Strong interest coverage {latest.interest_coverage:.1f}x — well clear of debt trap")
        elif latest.interest_coverage > 2:
            score += 1
            details.append(f"Adequate interest coverage {latest.interest_coverage:.1f}x")
        else:
            details.append(f"Weak interest coverage {latest.interest_coverage:.1f}x — debt-cycle risk")
    else:
        details.append("Interest coverage unavailable")

    # Leverage trend: is debt growing faster than assets? Increasing leverage = late cycle
    debt_vals = [li.total_debt for li in line_items if li.total_debt is not None]
    asset_vals = [li.total_assets for li in line_items if li.total_assets is not None]
    if len(debt_vals) >= 2 and len(asset_vals) >= 2:
        debt_ratio_now = debt_vals[0] / asset_vals[0] if asset_vals[0] else None
        debt_ratio_prev = debt_vals[-1] / asset_vals[-1] if asset_vals[-1] else None
        if debt_ratio_now is not None and debt_ratio_prev is not None:
            if debt_ratio_now < debt_ratio_prev:
                score += 2
                details.append(f"Leverage declining ({debt_ratio_prev:.2f} → {debt_ratio_now:.2f}) — early/mid cycle")
            elif debt_ratio_now > debt_ratio_prev * 1.1:
                details.append(f"Leverage rising ({debt_ratio_prev:.2f} → {debt_ratio_now:.2f}) — late-cycle warning")
            else:
                score += 1
                details.append(f"Leverage stable ({debt_ratio_now:.2f})")
    else:
        details.append("Insufficient leverage trend data")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_economic_regime(metrics: list) -> dict:
    """Proxy for macro regime: revenue/earnings growth and margin trends."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # Revenue growth proxy (growth > 0 = expansion regime)
    if latest.revenue_growth is not None:
        if latest.revenue_growth > 0.10:
            score += 2
            details.append(f"Strong revenue growth {latest.revenue_growth:.1%} — expansion regime")
        elif latest.revenue_growth > 0:
            score += 1
            details.append(f"Modest revenue growth {latest.revenue_growth:.1%}")
        else:
            details.append(f"Revenue contraction {latest.revenue_growth:.1%} — contraction regime")
    else:
        details.append("Revenue growth data unavailable")

    # Operating margin trend as inflation/pricing proxy
    if latest.operating_margin is not None:
        if latest.operating_margin > 0.15:
            score += 2
            details.append(f"Healthy operating margins {latest.operating_margin:.1%} — pricing power intact")
        elif latest.operating_margin > 0.05:
            score += 1
            details.append(f"Modest operating margins {latest.operating_margin:.1%}")
        else:
            details.append(f"Compressed margins {latest.operating_margin:.1%} — cost/inflation pressure")
    else:
        details.append("Operating margin data unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_balance_sheet_resilience(metrics: list, line_items: list) -> dict:
    """Dalio: survive the bad times. Cash and equity buffer against debt deflation."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # Debt-to-equity: Dalio warns heavily leveraged balance sheets blow up in downturns
    if latest.debt_to_equity is not None:
        if latest.debt_to_equity < 0.5:
            score += 2
            details.append(f"Low leverage {latest.debt_to_equity:.2f}x D/E — resilient balance sheet")
        elif latest.debt_to_equity < 1.5:
            score += 1
            details.append(f"Moderate leverage {latest.debt_to_equity:.2f}x D/E")
        else:
            details.append(f"High leverage {latest.debt_to_equity:.2f}x D/E — fragile in downturn")
    else:
        details.append("Debt-to-equity unavailable")

    # FCF generation: Dalio looks for self-funding businesses that don't need credit markets
    if latest.free_cash_flow_yield is not None:
        if latest.free_cash_flow_yield > 0.05:
            score += 2
            details.append(f"Strong FCF yield {latest.free_cash_flow_yield:.1%} — self-funding")
        elif latest.free_cash_flow_yield > 0:
            score += 1
            details.append(f"Positive FCF yield {latest.free_cash_flow_yield:.1%}")
        else:
            details.append(f"Negative FCF yield {latest.free_cash_flow_yield:.1%} — dependent on credit markets")
    else:
        details.append("FCF yield unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_valuation_cycle(metrics: list, market_cap: float | None) -> dict:
    """Valuation extremes as cycle indicators: expensive = late cycle / greed, cheap = early cycle / fear."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # P/E as cycle barometer
    if latest.price_to_earnings_ratio is not None and latest.price_to_earnings_ratio > 0:
        pe = latest.price_to_earnings_ratio
        if pe < 15:
            score += 2
            details.append(f"Low P/E {pe:.1f} — early/mid cycle pricing")
        elif pe < 25:
            score += 1
            details.append(f"Fair P/E {pe:.1f}")
        else:
            details.append(f"High P/E {pe:.1f} — late-cycle or bubble territory")
    else:
        details.append("P/E ratio unavailable")

    # EV/EBITDA as cycle barometer
    if latest.enterprise_value_to_ebitda_ratio is not None and latest.enterprise_value_to_ebitda_ratio > 0:
        ev_ebitda = latest.enterprise_value_to_ebitda_ratio
        if ev_ebitda < 10:
            score += 2
            details.append(f"Attractive EV/EBITDA {ev_ebitda:.1f} — cycle-adjusted value")
        elif ev_ebitda < 18:
            score += 1
            details.append(f"Fair EV/EBITDA {ev_ebitda:.1f}")
        else:
            details.append(f"Stretched EV/EBITDA {ev_ebitda:.1f} — cycle peak risk")
    else:
        details.append("EV/EBITDA unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def generate_dalio_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "ray_dalio_agent",
) -> BaseSignal:
    """Get investment decision from LLM in Ray Dalio's voice."""
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "debt_cycle": analysis_data.get("debt_cycle", {}).get("details"),
        "economic_regime": analysis_data.get("economic_regime", {}).get("details"),
        "balance_sheet": analysis_data.get("balance_sheet", {}).get("details"),
        "valuation_cycle": analysis_data.get("valuation_cycle", {}).get("details"),
        "market_cap": analysis_data.get("market_cap"),
    }

    prompt = build_persona_prompt(_PERSONA, facts, ticker)

    def _default():
        return BaseSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=BaseSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
