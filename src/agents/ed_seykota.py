# SPDX-License-Identifier: MIT
from datetime import datetime, timedelta
import json
import math

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents._data_bundle import AgentDataBundle
from src.agents._signals import BaseSignal
from src.agents.technicals import calculate_atr, calculate_ema, safe_float
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class EdSeykotaSignal(BaseSignal):
    pass


def ed_seykota_agent(state: AgentState, agent_id: str = "ed_seykota_agent"):
    """Analyzes stocks using Seykota's systematic trend-following: ride trends, cut losers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    # 14 months for reliable 200-day MA and Donchian 52-week window
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=430)).date().isoformat()

    analysis_data = {}
    seykota_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price data")
        bundle = AgentDataBundle.fetch(
            ticker,
            end_date,
            start_date=start_date,
            include_prices=True,
            include_market_cap=False,
            api_key=api_key,
        )
        prices_df = prices_to_df(bundle.prices) if bundle.prices else None

        progress.update_status(agent_id, ticker, "Analyzing trend direction")
        trend_direction = analyze_trend_direction(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing trend strength")
        trend_strength = analyze_trend_strength(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing Donchian breakout")
        donchian = analyze_donchian_breakout(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing volatility sizing")
        vol_sizing = analyze_volatility_sizing(prices_df)

        total_score = trend_direction["score"] + trend_strength["score"] + donchian["score"] + vol_sizing["score"]
        max_possible_score = trend_direction["max_score"] + trend_strength["max_score"] + donchian["max_score"] + vol_sizing["max_score"]

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "donchian": donchian,
            "vol_sizing": vol_sizing,
        }

        progress.update_status(agent_id, ticker, "Generating Ed Seykota analysis")
        seykota_output = generate_seykota_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        seykota_analysis[ticker] = {
            "signal": seykota_output.signal,
            "confidence": seykota_output.confidence,
            "reasoning": seykota_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=seykota_output.reasoning)

    message = HumanMessage(content=json.dumps(seykota_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(seykota_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = seykota_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_trend_direction(prices_df) -> dict:
    """Seykota: 'The trend is your friend.' Price vs 200-day MA is the primary signal."""
    max_score = 4
    if prices_df is None or len(prices_df) < 50:
        return {"score": 0, "max_score": max_score, "details": "Insufficient price data"}

    details = []
    score = 0
    closes = prices_df["close"]

    # 200-day MA (use 50-day if less data)
    ma_period = min(200, len(closes) - 1)
    ma200 = closes.rolling(ma_period).mean()
    current_price = closes.iloc[-1]
    ma200_val = safe_float(ma200.iloc[-1])

    if ma200_val > 0:
        pct_above = (current_price - ma200_val) / ma200_val
        if pct_above > 0.05:
            score += 2
            details.append(f"Price {pct_above:.1%} above {ma_period}-MA — uptrend confirmed")
        elif pct_above > 0:
            score += 1
            details.append(f"Price slightly above {ma_period}-MA (+{pct_above:.1%}) — weak uptrend")
        elif pct_above > -0.05:
            details.append(f"Price near {ma_period}-MA ({pct_above:.1%}) — no trend")
        else:
            details.append(f"Price {pct_above:.1%} below {ma_period}-MA — downtrend")
    else:
        details.append(f"{ma_period}-MA unavailable")

    # 50-day vs 200-day golden/death cross
    if len(closes) >= 200:
        ma50 = closes.rolling(50).mean()
        ma50_val = safe_float(ma50.iloc[-1])
        if ma50_val > 0 and ma200_val > 0:
            if ma50_val > ma200_val:
                score += 2
                details.append("Golden cross: 50-MA > 200-MA — strong bull trend")
            else:
                details.append("Death cross: 50-MA < 200-MA — bear trend")
    else:
        # Use 21 vs 55 EMA when not enough history
        ema21 = calculate_ema(prices_df, 21)
        ema55 = calculate_ema(prices_df, 55)
        if safe_float(ema21.iloc[-1]) > safe_float(ema55.iloc[-1]):
            score += 2
            details.append("EMA21 > EMA55 — upward trend structure")
        else:
            details.append("EMA21 < EMA55 — downward trend structure")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_trend_strength(prices_df) -> dict:
    """Seykota: only ride strong trends. Weak or choppy markets = wait."""
    max_score = 4
    if prices_df is None or len(prices_df) < 30:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data for trend strength"}

    details = []
    score = 0

    # 6-month and 3-month momentum as trend-strength proxies
    closes = prices_df["close"]
    if len(closes) >= 126:
        mom_6m = (closes.iloc[-1] - closes.iloc[-126]) / closes.iloc[-126]
        if mom_6m > 0.20:
            score += 2
            details.append(f"6-month momentum {mom_6m:.1%} — strongly trending")
        elif mom_6m > 0.05:
            score += 1
            details.append(f"6-month momentum {mom_6m:.1%} — modest uptrend")
        elif mom_6m < -0.10:
            details.append(f"6-month momentum {mom_6m:.1%} — downtrend")
        else:
            details.append(f"6-month momentum {mom_6m:.1%} — flat/choppy")
    else:
        details.append("Insufficient history for 6-month momentum")

    if len(closes) >= 63:
        mom_3m = (closes.iloc[-1] - closes.iloc[-63]) / closes.iloc[-63]
        if mom_3m > 0.10:
            score += 2
            details.append(f"3-month momentum {mom_3m:.1%} — acceleration")
        elif mom_3m > 0:
            score += 1
            details.append(f"3-month momentum {mom_3m:.1%} — positive but mild")
        else:
            details.append(f"3-month momentum {mom_3m:.1%} — weakening")
    else:
        details.append("Insufficient history for 3-month momentum")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_donchian_breakout(prices_df) -> dict:
    """Donchian channel: new 52-week high = breakout (bullish); new 52-week low = breakdown."""
    max_score = 4
    if prices_df is None or len(prices_df) < 50:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data for Donchian"}

    closes = prices_df["close"]
    window = min(252, len(closes) - 1)
    high_52w = closes.iloc[-window:].max()
    low_52w = closes.iloc[-window:].min()
    current = closes.iloc[-1]

    if high_52w == low_52w:
        return {"score": 2, "max_score": max_score, "details": "No price range — inconclusive Donchian"}

    range_pct = (current - low_52w) / (high_52w - low_52w)

    details = []
    score = 0

    if range_pct > 0.90:
        score += 4
        details.append(f"Price at {range_pct:.0%} of {window}d range — near breakout high, Seykota BUY")
    elif range_pct > 0.70:
        score += 3
        details.append(f"Price at {range_pct:.0%} of {window}d range — upper quartile, bullish")
    elif range_pct > 0.40:
        score += 2
        details.append(f"Price at {range_pct:.0%} of {window}d range — middle, no breakout signal")
    elif range_pct > 0.20:
        score += 1
        details.append(f"Price at {range_pct:.0%} of {window}d range — lower half, weak")
    else:
        details.append(f"Price at {range_pct:.0%} of {window}d range — near breakout low, Seykota EXIT/SHORT")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def analyze_volatility_sizing(prices_df) -> dict:
    """ATR-based risk: Seykota sizes positions by volatility. Low ATR = larger position = higher conviction."""
    max_score = 4
    if prices_df is None or len(prices_df) < 20:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data for ATR"}

    atr = calculate_atr(prices_df)
    current_price = safe_float(prices_df["close"].iloc[-1])
    atr_val = safe_float(atr.iloc[-1])

    if current_price <= 0 or atr_val <= 0:
        return {"score": 2, "max_score": max_score, "details": "Cannot compute ATR ratio"}

    atr_pct = atr_val / current_price
    ann_vol = atr_pct * math.sqrt(252)

    details = []
    score = 0

    if ann_vol < 0.20:
        score += 4
        details.append(f"ATR-implied vol {ann_vol:.1%} — calm market, Seykota can size large")
    elif ann_vol < 0.35:
        score += 3
        details.append(f"ATR-implied vol {ann_vol:.1%} — moderate, normal sizing")
    elif ann_vol < 0.55:
        score += 2
        details.append(f"ATR-implied vol {ann_vol:.1%} — elevated, smaller position")
    elif ann_vol < 0.80:
        score += 1
        details.append(f"ATR-implied vol {ann_vol:.1%} — high, tight stops required")
    else:
        details.append(f"ATR-implied vol {ann_vol:.1%} — extreme, Seykota would stand aside")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def generate_seykota_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "ed_seykota_agent",
) -> EdSeykotaSignal:
    """Get investment decision from LLM in Ed Seykota's voice."""
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "trend_direction": analysis_data.get("trend_direction", {}).get("details"),
        "trend_strength": analysis_data.get("trend_strength", {}).get("details"),
        "donchian": analysis_data.get("donchian", {}).get("details"),
        "vol_sizing": analysis_data.get("vol_sizing", {}).get("details"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Ed Seykota, pioneer of systematic trend-following. Decide bullish, bearish, or neutral using only the provided facts.\n"
                "\n"
                "Your rules:\n"
                "- 'The trend is your friend until the bend at the end.' — price above 200-MA = ride it\n"
                "- Donchian breakout to new highs = buy signal; breakdown to new lows = exit/short\n"
                "- Trend strength (momentum) must confirm direction — no trend = no trade\n"
                "- Size by volatility: calm markets allow larger positions; choppy = stand aside\n"
                "- Cut losers short; let winners run — never average down\n"
                "\n"
                "Signal rules:\n"
                "- Bullish: uptrend confirmed (price > MA) + Donchian upper zone + positive momentum\n"
                "- Bearish: downtrend (price < MA) + Donchian lower zone + negative momentum\n"
                "- Neutral: no clear trend, choppy range, or mixed signals\n"
                "\n"
                "Confidence scale:\n"
                "- 90-100%: All trend signals aligned, breakout confirmed, low vol\n"
                "- 70-89%: Strong trend with confirmation\n"
                "- 50-69%: Trend present but weak or consolidating\n"
                "- 30-49%: Mixed or early-stage signals\n"
                "- 10-29%: Downtrend or breakdown confirmed\n"
                "\n"
                "Use Seykota's vocabulary: trend, breakout, cut your losses, ride the trend, whipsaw, position sizing.\n"
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
        return EdSeykotaSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=EdSeykotaSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
