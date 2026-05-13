# SPDX-License-Identifier: MIT
from datetime import datetime, timedelta
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
    "You are Howard Marks. Decide bullish, bearish, or neutral using only the provided facts.\n"
    "\n"
    "Framework:\n"
    "- Credit quality: avoid losses above all else; weak coverage = first-level trap\n"
    "- Cycle position: where is the market/stock in the fear-greed cycle?\n"
    "- Risk premium: are you being paid adequately for the risk you're taking?\n"
    "- Second-level thinking: what does the price imply? What does the crowd miss?\n"
    "\n"
    "Signal rules:\n"
    "- Bullish: strong credit quality + fear/early-cycle pricing + adequate risk premium + second-level insight\n"
    "- Bearish: stressed credit OR late-cycle greed pricing OR inadequate risk premium\n"
    "- Neutral: average credit and mid-cycle with fair compensation — nothing exceptional\n"
    "\n"
    "Confidence scale:\n"
    "- 90-100%: Distressed-asset-level opportunity — high quality, fear pricing, insider buying\n"
    "- 70-89%: Good credit, below-average enthusiasm, decent compensation\n"
    "- 50-69%: Average quality and average pricing\n"
    "- 30-49%: Some credit stress or late-cycle signals\n"
    "- 10-29%: Stressed credit, greed pricing, thin risk premium — avoid\n"
    "\n"
    "Use Marks's vocabulary: first-level thinking, second-level thinking, cycle, risk premium, wall of worry, tree doesn't grow to the sky.\n"
    'Write reasoning as a short bullet list (2–3 bullets preferred, max 5). Each bullet: "- " + one fact or judgment under ~100 chars. Stay in Howard Marks\'s second-level-thinking voice. Do not invent data. Return JSON only.'
)


def howard_marks_agent(state: AgentState, agent_id: str = "howard_marks_agent"):
    """Analyzes stocks using Marks's market-cycle, second-level thinking, and credit-quality framework."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    def _analyze(ticker: str) -> dict:
        progress.update_status(agent_id, ticker, "Fetching financial data")
        bundle = AgentDataBundle.fetch(
            ticker,
            end_date,
            start_date=start_date,
            line_item_names=[
                "total_debt",
                "total_assets",
                "net_income",
                "revenue",
                "operating_income",
                "interest_expense",
                "free_cash_flow",
                "cash_and_equivalents",
            ],
            line_item_period="ttm",
            line_item_limit=5,
            metrics_period="ttm",
            metrics_limit=5,
            insider_limit=50,
            api_key=api_key,
        )
        metrics = bundle.financial_metrics
        line_items = bundle.line_items
        insider_trades = bundle.insider_trades
        market_cap = bundle.market_cap

        progress.update_status(agent_id, ticker, "Analyzing credit quality")
        credit_quality = analyze_credit_quality(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing cycle position")
        cycle_position = analyze_cycle_position(metrics)

        progress.update_status(agent_id, ticker, "Analyzing risk premium")
        risk_premium = analyze_risk_premium(metrics)

        progress.update_status(agent_id, ticker, "Applying second-level thinking")
        second_level = analyze_second_level(metrics, insider_trades)

        total_score = credit_quality["score"] + cycle_position["score"] + risk_premium["score"] + second_level["score"]
        max_possible_score = credit_quality["max_score"] + cycle_position["max_score"] + risk_premium["max_score"] + second_level["max_score"]

        ticker_analysis = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "credit_quality": credit_quality,
            "cycle_position": cycle_position,
            "risk_premium": risk_premium,
            "second_level": second_level,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Howard Marks analysis")
        marks_output = generate_marks_output(
            ticker=ticker,
            analysis_data=ticker_analysis,
            state=state,
            agent_id=agent_id,
        )

        progress.update_status(agent_id, ticker, "Done", analysis=marks_output.reasoning)
        return {
            "signal": marks_output.signal,
            "confidence": marks_output.confidence,
            "reasoning": marks_output.reasoning,
        }

    marks_analysis = parallel_per_ticker(tickers, _analyze)

    message = HumanMessage(content=json.dumps(marks_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(marks_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = marks_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_credit_quality(metrics: list, line_items: list) -> dict:
    """Marks prioritizes credit quality — avoiding loss matters more than achieving gain."""
    score = 0
    max_score = 6
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # Interest coverage — below 2x is distress territory
    if latest.interest_coverage is not None:
        if latest.interest_coverage > 5:
            score += 2
            details.append(f"Robust interest coverage {latest.interest_coverage:.1f}x — investment-grade quality")
        elif latest.interest_coverage > 2:
            score += 1
            details.append(f"Adequate interest coverage {latest.interest_coverage:.1f}x")
        else:
            details.append(f"Stressed interest coverage {latest.interest_coverage:.1f}x — high-yield / distress")
    else:
        details.append("Interest coverage unavailable")

    # Debt-to-equity
    if latest.debt_to_equity is not None:
        if latest.debt_to_equity < 0.5:
            score += 2
            details.append(f"Conservative leverage {latest.debt_to_equity:.2f}x D/E — ample cushion")
        elif latest.debt_to_equity < 1.5:
            score += 1
            details.append(f"Moderate leverage {latest.debt_to_equity:.2f}x D/E")
        else:
            details.append(f"High leverage {latest.debt_to_equity:.2f}x D/E — Marks would demand a margin of safety on credit")
    else:
        details.append("D/E unavailable")

    # Debt trend: is the company borrowing more or paying down?
    debt_vals = [li.total_debt for li in line_items if li.total_debt is not None]
    if len(debt_vals) >= 2:
        if debt_vals[0] < debt_vals[-1]:
            score += 2
            details.append("Debt declining — deleveraging trend, improving credit profile")
        elif debt_vals[0] > debt_vals[-1] * 1.2:
            details.append("Debt rising significantly — deteriorating credit quality")
        else:
            score += 1
            details.append("Debt roughly stable")
    else:
        details.append("Insufficient debt trend data")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_cycle_position(metrics: list) -> dict:
    """Where are we in the cycle? Expensive = greed / late cycle. Cheap = fear / early cycle."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # P/E as greed/fear barometer
    if latest.price_to_earnings_ratio is not None and latest.price_to_earnings_ratio > 0:
        pe = latest.price_to_earnings_ratio
        if pe < 12:
            score += 2
            details.append(f"P/E {pe:.1f} — fear/pessimism zone, cycle may be early/turning")
        elif pe < 22:
            score += 1
            details.append(f"P/E {pe:.1f} — mid-cycle, reasonable")
        else:
            details.append(f"P/E {pe:.1f} — optimism/greed zone, late cycle caution")
    else:
        details.append("P/E unavailable")

    # Price-to-book as cycle indicator (extremely high = speculative bubble)
    if latest.price_to_book_ratio is not None and latest.price_to_book_ratio > 0:
        pb = latest.price_to_book_ratio
        if pb < 2:
            score += 2
            details.append(f"P/B {pb:.1f} — below average optimism, room for re-rating")
        elif pb < 5:
            score += 1
            details.append(f"P/B {pb:.1f} — moderate market enthusiasm")
        else:
            details.append(f"P/B {pb:.1f} — elevated expectations; consensus already optimistic")
    else:
        details.append("P/B unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_risk_premium(metrics: list) -> dict:
    """Marks: you are paid to take risk only when the price compensates you adequately."""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # FCF yield as risk-premium proxy
    if latest.free_cash_flow_yield is not None:
        if latest.free_cash_flow_yield > 0.07:
            score += 2
            details.append(f"High FCF yield {latest.free_cash_flow_yield:.1%} — ample compensation for risk")
        elif latest.free_cash_flow_yield > 0.03:
            score += 1
            details.append(f"Adequate FCF yield {latest.free_cash_flow_yield:.1%}")
        else:
            details.append(f"Low FCF yield {latest.free_cash_flow_yield:.1%} — thin risk premium")
    else:
        details.append("FCF yield unavailable")

    # ROIC as quality premium indicator
    if latest.return_on_invested_capital is not None:
        if latest.return_on_invested_capital > 0.15:
            score += 2
            details.append(f"Strong ROIC {latest.return_on_invested_capital:.1%} — quality business justifies premium")
        elif latest.return_on_invested_capital > 0.08:
            score += 1
            details.append(f"Adequate ROIC {latest.return_on_invested_capital:.1%}")
        else:
            details.append(f"Weak ROIC {latest.return_on_invested_capital:.1%} — insufficient return on capital")
    else:
        details.append("ROIC unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_second_level(metrics: list, insider_trades: list) -> dict:
    """Second-level thinking: what does the price imply vs what is actually happening?"""
    score = 0
    max_score = 4
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics available"}

    latest = metrics[0]

    # Earnings vs revenue growth divergence: margins expanding while sentiment is negative = second-level opportunity
    if latest.earnings_growth is not None and latest.revenue_growth is not None:
        if latest.earnings_growth > latest.revenue_growth + 0.05:
            score += 2
            details.append(f"Earnings growing faster than revenue ({latest.earnings_growth:.1%} vs {latest.revenue_growth:.1%}) — operating leverage / margin expansion")
        elif latest.earnings_growth > 0 and latest.revenue_growth > 0:
            score += 1
            details.append(f"Both earnings and revenue growing ({latest.earnings_growth:.1%} / {latest.revenue_growth:.1%})")
        elif latest.earnings_growth < 0:
            details.append(f"Earnings declining {latest.earnings_growth:.1%} — consensus may be correctly pessimistic")
    else:
        details.append("Growth data unavailable")

    # Insider activity as second-level signal: insiders buying = they see more than consensus
    if insider_trades:
        buys = sum(1 for t in insider_trades if t.transaction_shares and t.transaction_shares > 0)
        sells = sum(1 for t in insider_trades if t.transaction_shares and t.transaction_shares < 0)
        total = buys + sells
        if total > 0:
            buy_ratio = buys / total
            if buy_ratio > 0.6:
                score += 2
                details.append(f"Insider buying dominates ({buys}/{total}) — second-level signal: they know something consensus doesn't")
            elif buy_ratio < 0.3 and total >= 5:
                details.append(f"Heavy insider selling ({sells}/{total}) — insiders disagree with any optimistic consensus")
            else:
                score += 1
                details.append(f"Mixed insider activity ({buys} buys, {sells} sells)")
        else:
            details.append("No insider transactions")
    else:
        details.append("No insider data")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def generate_marks_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "howard_marks_agent",
) -> BaseSignal:
    """Get investment decision from LLM in Howard Marks's voice."""
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "credit_quality": analysis_data.get("credit_quality", {}).get("details"),
        "cycle_position": analysis_data.get("cycle_position", {}).get("details"),
        "risk_premium": analysis_data.get("risk_premium", {}).get("details"),
        "second_level": analysis_data.get("second_level", {}).get("details"),
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
