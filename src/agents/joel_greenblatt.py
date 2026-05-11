# SPDX-License-Identifier: MIT
from datetime import datetime, timedelta
import json

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents._data_bundle import AgentDataBundle
from src.agents._signals import BaseSignal
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class JoelGreenblattSignal(BaseSignal):
    pass


def joel_greenblatt_agent(state: AgentState, agent_id: str = "joel_greenblatt_agent"):
    """Analyzes stocks using Greenblatt's Magic Formula (ROIC × earnings yield) and special-situations lens."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data = {}
    greenblatt_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial data")
        bundle = AgentDataBundle.fetch(
            ticker,
            end_date,
            start_date=start_date,
            line_item_names=[
                "operating_income",
                "total_assets",
                "total_debt",
                "cash_and_equivalents",
                "net_income",
                "revenue",
                "shareholders_equity",
                "capital_expenditure",
            ],
            line_item_period="ttm",
            line_item_limit=3,
            metrics_period="ttm",
            metrics_limit=3,
            insider_limit=50,
            api_key=api_key,
        )
        metrics = bundle.financial_metrics
        line_items = bundle.line_items
        insider_trades = bundle.insider_trades
        market_cap = bundle.market_cap

        progress.update_status(agent_id, ticker, "Scoring Magic Formula")
        magic_formula = score_magic_formula(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "Checking special situations")
        special_situations = score_special_situations(metrics, line_items, insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing capital returns")
        capital_returns = score_capital_returns(metrics, line_items)

        total_score = magic_formula["score"] + special_situations["score"] + capital_returns["score"]
        max_possible_score = magic_formula["max_score"] + special_situations["max_score"] + capital_returns["max_score"]

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "magic_formula": magic_formula,
            "special_situations": special_situations,
            "capital_returns": capital_returns,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Joel Greenblatt analysis")
        greenblatt_output = generate_greenblatt_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        greenblatt_analysis[ticker] = {
            "signal": greenblatt_output.signal,
            "confidence": greenblatt_output.confidence,
            "reasoning": greenblatt_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=greenblatt_output.reasoning)

    message = HumanMessage(content=json.dumps(greenblatt_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(greenblatt_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = greenblatt_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def score_magic_formula(metrics: list, line_items: list, market_cap: float | None) -> dict:
    """Greenblatt's Magic Formula: high ROIC (quality) × high earnings yield (cheap). Both must be high."""
    score = 0
    max_score = 8
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics for Magic Formula"}

    m = metrics[0]

    # Earnings yield = EBIT / Enterprise Value (inverse of EV/EBIT)
    # Proxy: use operating_income from line items + EV from market_cap + debt - cash
    ebit = None
    if line_items:
        latest_li = line_items[0]
        ebit = getattr(latest_li, "operating_income", None)

    debt = getattr(line_items[0], "total_debt", None) if line_items else None
    cash = getattr(line_items[0], "cash_and_equivalents", None) if line_items else None

    ev = None
    if market_cap and market_cap > 0:
        ev = market_cap + (debt or 0) - (cash or 0)

    if ebit and ev and ev > 0:
        earnings_yield = ebit / ev
        if earnings_yield > 0.12:
            score += 4
            details.append(f"Earnings yield {earnings_yield:.1%} — top Magic Formula rank (cheap)")
        elif earnings_yield > 0.07:
            score += 3
            details.append(f"Earnings yield {earnings_yield:.1%} — good value signal")
        elif earnings_yield > 0.04:
            score += 2
            details.append(f"Earnings yield {earnings_yield:.1%} — fair")
        elif earnings_yield > 0:
            score += 1
            details.append(f"Earnings yield {earnings_yield:.1%} — thin")
        else:
            details.append("Negative earnings yield — Magic Formula penalises")
    elif m.enterprise_value_to_ebitda_ratio and m.enterprise_value_to_ebitda_ratio > 0:
        # Fallback: use EV/EBITDA as proxy
        ey_proxy = 1 / m.enterprise_value_to_ebitda_ratio
        if ey_proxy > 0.10:
            score += 3
            details.append(f"EV/EBITDA-implied yield {ey_proxy:.1%} — cheap")
        elif ey_proxy > 0.06:
            score += 2
            details.append(f"EV/EBITDA-implied yield {ey_proxy:.1%} — fair")
        else:
            score += 1
            details.append(f"EV/EBITDA-implied yield {ey_proxy:.1%} — expensive")
    else:
        details.append("Cannot compute earnings yield — no EV or EBIT data")

    # ROIC — the quality leg
    if m.return_on_invested_capital is not None:
        if m.return_on_invested_capital > 0.25:
            score += 4
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — top Magic Formula rank (quality)")
        elif m.return_on_invested_capital > 0.15:
            score += 3
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — good quality")
        elif m.return_on_invested_capital > 0.08:
            score += 2
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — adequate")
        else:
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — low quality, Magic Formula penalises")
    else:
        details.append("ROIC unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def score_special_situations(metrics: list, line_items: list, insider_trades: list) -> dict:
    """Greenblatt loves spin-offs, restructurings, and catalysts. Insider buying is a key signal."""
    score = 0
    max_score = 6
    details = []

    # Insider buying as catalyst signal
    if insider_trades:
        buys = sum(1 for t in insider_trades if t.transaction_shares and t.transaction_shares > 0)
        sells = sum(1 for t in insider_trades if t.transaction_shares and t.transaction_shares < 0)
        total = buys + sells
        if total > 0:
            buy_ratio = buys / total
            if buy_ratio > 0.7 and buys >= 3:
                score += 3
                details.append(f"Strong insider buying ({buys}/{total}) — classic catalyst signal for Greenblatt")
            elif buy_ratio > 0.5:
                score += 2
                details.append(f"Moderate insider buying ({buys}/{total})")
            elif buy_ratio < 0.3 and sells >= 5:
                details.append(f"Heavy insider selling ({sells}/{total}) — negative catalyst signal")
            else:
                score += 1
                details.append(f"Mixed insider activity ({buys} buys, {sells} sells)")
        else:
            details.append("No insider transactions")
    else:
        details.append("No insider data available")

    # Earnings acceleration as restructuring / improving operations signal
    if metrics and len(metrics) >= 2:
        curr_growth = metrics[0].earnings_growth
        if curr_growth is not None and curr_growth > 0.15:
            score += 2
            details.append(f"Earnings growth {curr_growth:.1%} — business improving (potential catalyst)")
        elif curr_growth is not None and curr_growth > 0:
            score += 1
            details.append(f"Modest earnings growth {curr_growth:.1%}")
        elif curr_growth is not None:
            details.append(f"Earnings declining {curr_growth:.1%} — no catalyst visible")
        else:
            details.append("Earnings growth data unavailable")
    else:
        details.append("Insufficient metrics for earnings trend")

    # Low debt = less distress, more capacity for shareholder-friendly actions
    if metrics:
        m = metrics[0]
        if m.debt_to_equity is not None and m.debt_to_equity < 0.5:
            score += 1
            details.append(f"Low D/E {m.debt_to_equity:.2f} — balance sheet supports buybacks / special divs")
        elif m.debt_to_equity is not None:
            details.append(f"D/E {m.debt_to_equity:.2f} — limits special-situation optionality")
        else:
            details.append("D/E unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def score_capital_returns(metrics: list, line_items: list) -> dict:
    """Greenblatt: capital returns to shareholders amplify the Magic Formula thesis."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics for capital returns"}

    m = metrics[0]

    # FCF yield indicates ability to return capital
    if m.free_cash_flow_yield is not None:
        if m.free_cash_flow_yield > 0.08:
            score += 2
            details.append(f"FCF yield {m.free_cash_flow_yield:.1%} — strong buyback / dividend capacity")
        elif m.free_cash_flow_yield > 0.04:
            score += 1
            details.append(f"FCF yield {m.free_cash_flow_yield:.1%} — adequate returns")
        else:
            details.append(f"Low FCF yield {m.free_cash_flow_yield:.1%} — limited capital return potential")
    else:
        details.append("FCF yield unavailable")

    # Asset-light (low capex relative to earnings) = more free cash flow to return
    if line_items:
        li = line_items[0]
        capex = getattr(li, "capital_expenditure", None)
        net_income = getattr(li, "net_income", None)
        if capex is not None and net_income and net_income > 0:
            # capex is often negative in datasets
            capex_abs = abs(capex)
            capex_ratio = capex_abs / net_income
            if capex_ratio < 0.20:
                score += 2
                details.append(f"Asset-light: capex/net-income {capex_ratio:.0%} — high FCF conversion")
            elif capex_ratio < 0.50:
                score += 1
                details.append(f"Moderate capex intensity {capex_ratio:.0%}")
            else:
                details.append(f"Capital-intensive {capex_ratio:.0%} — limits return potential")
        else:
            details.append("Capex/earnings ratio unavailable")
    else:
        details.append("No line items for capex analysis")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def generate_greenblatt_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "joel_greenblatt_agent",
) -> JoelGreenblattSignal:
    """Get investment decision from LLM in Joel Greenblatt's voice."""
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "magic_formula": analysis_data.get("magic_formula", {}).get("details"),
        "special_situations": analysis_data.get("special_situations", {}).get("details"),
        "capital_returns": analysis_data.get("capital_returns", {}).get("details"),
        "market_cap": analysis_data.get("market_cap"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Joel Greenblatt. Decide bullish, bearish, or neutral using only the provided facts.\n"
                "\n"
                "Framework:\n"
                "- Magic Formula: high ROIC (quality) AND high earnings yield (cheap) = rank high → buy\n"
                "- Special situations: insider buying, earnings acceleration, restructuring = catalyst\n"
                "- Capital returns: strong FCF yield and asset-light = ability to buy back stock or pay dividends\n"
                "\n"
                "Signal rules:\n"
                "- Bullish: top Magic Formula rank (high ROIC + cheap) AND at least one catalyst\n"
                "- Bearish: low ROIC OR expensive OR heavy insider selling with declining earnings\n"
                "- Neutral: good one leg of formula but not the other, or no catalyst\n"
                "\n"
                "Confidence scale:\n"
                "- 90-100%: Top Magic Formula rank + insider buying + strong FCF return\n"
                "- 70-89%: Good Magic Formula score with one catalyst\n"
                "- 50-69%: Average formula rank, mixed catalysts\n"
                "- 30-49%: Weak formula (cheap but low ROIC, or high ROIC but expensive)\n"
                "- 10-29%: Poor formula rank + negative catalysts\n"
                "\n"
                "Use Greenblatt's vocabulary: Magic Formula, earnings yield, ROIC, special situation, spin-off, catalyst, good business at a cheap price.\n"
                "Keep reasoning under 150 characters. Do not invent data. Return JSON only.",
            ),
            (
                "human",
                'Ticker: {ticker}\nFacts:\n{facts}\n\nReturn exactly:\n{{\n  "signal": "bullish" | "bearish" | "neutral",\n  "confidence": int,\n  "reasoning": "short justification"\n}}',
            ),
        ]
    )

    prompt = template.invoke(
        {
            "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
            "ticker": ticker,
        }
    )

    def _default():
        return JoelGreenblattSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=JoelGreenblattSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
