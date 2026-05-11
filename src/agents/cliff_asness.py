# SPDX-License-Identifier: MIT
from datetime import datetime, timedelta
import json
import math

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents._data_bundle import AgentDataBundle
from src.agents._signals import BaseSignal
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class CliffAsnessSignal(BaseSignal):
    pass


def cliff_asness_agent(state: AgentState, agent_id: str = "cliff_asness_agent"):
    """Analyzes stocks using AQR's multi-factor framework: value, momentum, quality, low-vol."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    # 14 months to compute 12-1 momentum (skip most recent month)
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=430)).date().isoformat()

    analysis_data = {}
    asness_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial data")
        bundle = AgentDataBundle.fetch(
            ticker,
            end_date,
            start_date=start_date,
            line_item_names=[
                "net_income",
                "revenue",
                "operating_income",
            ],
            line_item_period="annual",
            line_item_limit=5,
            metrics_period="ttm",
            metrics_limit=5,
            include_prices=True,
            api_key=api_key,
        )
        metrics = bundle.financial_metrics
        line_items = bundle.line_items
        prices_df = prices_to_df(bundle.prices) if bundle.prices else None
        market_cap = bundle.market_cap

        progress.update_status(agent_id, ticker, "Scoring value factor")
        value_factor = score_value_factor(metrics)

        progress.update_status(agent_id, ticker, "Scoring momentum factor")
        momentum_factor = score_momentum_factor(prices_df)

        progress.update_status(agent_id, ticker, "Scoring quality factor")
        quality_factor = score_quality_factor(metrics, line_items)

        progress.update_status(agent_id, ticker, "Scoring low-vol factor")
        low_vol_factor = score_low_vol_factor(prices_df)

        total_score = value_factor["score"] + momentum_factor["score"] + quality_factor["score"] + low_vol_factor["score"]
        max_possible_score = value_factor["max_score"] + momentum_factor["max_score"] + quality_factor["max_score"] + low_vol_factor["max_score"]

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "value_factor": value_factor,
            "momentum_factor": momentum_factor,
            "quality_factor": quality_factor,
            "low_vol_factor": low_vol_factor,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Cliff Asness analysis")
        asness_output = generate_asness_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        asness_analysis[ticker] = {
            "signal": asness_output.signal,
            "confidence": asness_output.confidence,
            "reasoning": asness_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=asness_output.reasoning)

    message = HumanMessage(content=json.dumps(asness_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(asness_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = asness_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def score_value_factor(metrics: list) -> dict:
    """AQR value: composite of P/E, P/B, FCF yield, EV/EBITDA. Lower multiples = higher score."""
    score = 0
    max_score = 8
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics for value factor"}

    m = metrics[0]
    hits = 0

    if m.price_to_earnings_ratio and m.price_to_earnings_ratio > 0:
        hits += 1
        if m.price_to_earnings_ratio < 15:
            score += 2
            details.append(f"P/E {m.price_to_earnings_ratio:.1f} — cheap value signal")
        elif m.price_to_earnings_ratio < 25:
            score += 1
            details.append(f"P/E {m.price_to_earnings_ratio:.1f} — fair")
        else:
            details.append(f"P/E {m.price_to_earnings_ratio:.1f} — expensive")

    if m.price_to_book_ratio and m.price_to_book_ratio > 0:
        hits += 1
        if m.price_to_book_ratio < 2:
            score += 2
            details.append(f"P/B {m.price_to_book_ratio:.1f} — deep value")
        elif m.price_to_book_ratio < 4:
            score += 1
            details.append(f"P/B {m.price_to_book_ratio:.1f} — reasonable")
        else:
            details.append(f"P/B {m.price_to_book_ratio:.1f} — growth premium")

    if m.free_cash_flow_yield is not None:
        hits += 1
        if m.free_cash_flow_yield > 0.06:
            score += 2
            details.append(f"FCF yield {m.free_cash_flow_yield:.1%} — high value score")
        elif m.free_cash_flow_yield > 0.02:
            score += 1
            details.append(f"FCF yield {m.free_cash_flow_yield:.1%}")
        else:
            details.append(f"FCF yield {m.free_cash_flow_yield:.1%} — low value score")

    if m.enterprise_value_to_ebitda_ratio and m.enterprise_value_to_ebitda_ratio > 0:
        hits += 1
        if m.enterprise_value_to_ebitda_ratio < 10:
            score += 2
            details.append(f"EV/EBITDA {m.enterprise_value_to_ebitda_ratio:.1f} — value")
        elif m.enterprise_value_to_ebitda_ratio < 18:
            score += 1
            details.append(f"EV/EBITDA {m.enterprise_value_to_ebitda_ratio:.1f} — fair")
        else:
            details.append(f"EV/EBITDA {m.enterprise_value_to_ebitda_ratio:.1f} — expensive")

    if hits == 0:
        details.append("No valuation metrics available")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def score_momentum_factor(prices_df) -> dict:
    """AQR 12-1 momentum: 12-month return excluding the most recent month."""
    max_score = 4
    if prices_df is None or len(prices_df) < 252:
        return {"score": 0, "max_score": max_score, "details": "Insufficient price history for momentum"}

    closes = prices_df["close"]
    # 12-1: return from 252 days ago to 21 days ago (skip the most recent month)
    price_12m_ago = closes.iloc[-252] if len(closes) >= 252 else closes.iloc[0]
    price_1m_ago = closes.iloc[-21] if len(closes) >= 21 else closes.iloc[-1]

    if price_12m_ago <= 0:
        return {"score": 0, "max_score": max_score, "details": "Invalid price data"}

    mom_12_1 = (price_1m_ago - price_12m_ago) / price_12m_ago

    # Also compute 1-month (to skip — high short-term reversal is actually negative in AQR)
    price_now = closes.iloc[-1]
    mom_1m = (price_now - price_1m_ago) / price_1m_ago if price_1m_ago > 0 else 0

    score = 0
    details = []

    if mom_12_1 > 0.20:
        score += 3
        details.append(f"12-1 momentum {mom_12_1:.1%} — strong positive signal")
    elif mom_12_1 > 0.05:
        score += 2
        details.append(f"12-1 momentum {mom_12_1:.1%} — moderate positive")
    elif mom_12_1 > -0.05:
        score += 1
        details.append(f"12-1 momentum {mom_12_1:.1%} — flat/neutral")
    else:
        details.append(f"12-1 momentum {mom_12_1:.1%} — negative, avoid per momentum factor")

    # AQR warns about short-term reversal: very strong recent 1-month may reverse
    if mom_1m > 0.15:
        score = max(0, score - 1)
        details.append(f"Short-term reversal risk: 1m gain {mom_1m:.1%} may mean-revert")

    return {"score": min(score, max_score), "max_score": max_score, "details": "; ".join(details)}


def score_quality_factor(metrics: list, line_items: list) -> dict:
    """AQR quality: high profitability + stability of earnings = premium multiple justified."""
    score = 0
    max_score = 6
    details = []

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics for quality factor"}

    m = metrics[0]

    # Profitability: ROIC
    if m.return_on_invested_capital is not None:
        if m.return_on_invested_capital > 0.20:
            score += 2
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — high quality")
        elif m.return_on_invested_capital > 0.10:
            score += 1
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — adequate")
        else:
            details.append(f"ROIC {m.return_on_invested_capital:.1%} — low quality")
    else:
        details.append("ROIC unavailable")

    # Gross margin as pricing power / quality proxy
    if m.gross_margin is not None:
        if m.gross_margin > 0.40:
            score += 2
            details.append(f"Gross margin {m.gross_margin:.1%} — strong competitive position")
        elif m.gross_margin > 0.20:
            score += 1
            details.append(f"Gross margin {m.gross_margin:.1%} — adequate")
        else:
            details.append(f"Gross margin {m.gross_margin:.1%} — commoditised business")
    else:
        details.append("Gross margin unavailable")

    # Earnings stability over time
    earnings_vals = [li.net_income for li in line_items if li.net_income is not None]
    if len(earnings_vals) >= 3:
        positive_years = sum(1 for e in earnings_vals if e > 0)
        if positive_years == len(earnings_vals):
            score += 2
            details.append(f"Earnings positive all {len(earnings_vals)} periods — stable quality")
        elif positive_years >= len(earnings_vals) * 0.75:
            score += 1
            details.append(f"Earnings positive {positive_years}/{len(earnings_vals)} periods")
        else:
            details.append(f"Earnings unstable: positive only {positive_years}/{len(earnings_vals)} periods")
    else:
        details.append("Insufficient earnings history for stability check")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def score_low_vol_factor(prices_df) -> dict:
    """AQR low-vol: lower realized volatility stocks tend to outperform on risk-adjusted basis."""
    max_score = 4
    if prices_df is None or len(prices_df) < 63:
        return {"score": 0, "max_score": max_score, "details": "Insufficient price history for vol factor"}

    returns = prices_df["close"].pct_change().dropna()
    ann_vol = returns.iloc[-63:].std() * math.sqrt(252)

    score = 0
    details = []

    if ann_vol < 0.20:
        score += 4
        details.append(f"Annualised vol {ann_vol:.1%} — low-vol premium candidate")
    elif ann_vol < 0.30:
        score += 3
        details.append(f"Annualised vol {ann_vol:.1%} — below-average volatility")
    elif ann_vol < 0.45:
        score += 2
        details.append(f"Annualised vol {ann_vol:.1%} — average volatility")
    elif ann_vol < 0.60:
        score += 1
        details.append(f"Annualised vol {ann_vol:.1%} — elevated volatility, lower factor score")
    else:
        details.append(f"Annualised vol {ann_vol:.1%} — high-vol, penalised by low-vol factor")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def generate_asness_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "cliff_asness_agent",
) -> CliffAsnessSignal:
    """Get investment decision from LLM in Cliff Asness's AQR voice."""
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "value_factor": analysis_data.get("value_factor", {}).get("details"),
        "momentum_factor": analysis_data.get("momentum_factor", {}).get("details"),
        "quality_factor": analysis_data.get("quality_factor", {}).get("details"),
        "low_vol_factor": analysis_data.get("low_vol_factor", {}).get("details"),
        "market_cap": analysis_data.get("market_cap"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Cliff Asness of AQR Capital. Decide bullish, bearish, or neutral using only the provided factor scores.\n"
                "\n"
                "AQR factor framework (all four matter equally):\n"
                "- Value: cheap on P/E, P/B, FCF yield, EV/EBITDA → bullish tilt\n"
                "- Momentum (12-1): positive 12-month price momentum, skip last month → bullish tilt\n"
                "- Quality: high ROIC, strong margins, stable earnings → bullish tilt\n"
                "- Low-vol: lower realized volatility → higher factor score\n"
                "\n"
                "Signal rules:\n"
                "- Bullish: majority of factors are positive (value + momentum + quality + low-vol)\n"
                "- Bearish: majority of factors are negative (expensive + negative momentum + low quality + high vol)\n"
                "- Neutral: mixed factor signals\n"
                "\n"
                "Confidence scale:\n"
                "- 90-100%: All four factors strongly aligned\n"
                "- 70-89%: Three factors positive\n"
                "- 50-69%: Two factors positive (or all weak)\n"
                "- 30-49%: Only one factor positive\n"
                "- 10-29%: All factors negative\n"
                "\n"
                "Use AQR vocabulary: factor premium, value spread, momentum signal, quality tilt, low-vol anomaly, factor exposure.\n"
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
        return CliffAsnessSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=CliffAsnessSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
