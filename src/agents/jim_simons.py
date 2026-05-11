# SPDX-License-Identifier: MIT
from datetime import datetime, timedelta
import json

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import numpy as np

from src.agents._data_bundle import AgentDataBundle
from src.agents._signals import BaseSignal
from src.agents.technicals import calculate_hurst_exponent, safe_float
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class JimSimonsSignal(BaseSignal):
    pass


def jim_simons_agent(state: AgentState, agent_id: str = "jim_simons_agent"):
    """Analyzes stocks using Renaissance-style statistical signals: mean reversion, autocorrelation, microstructure."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINNHUB_API_KEY")

    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data = {}
    simons_analysis = {}

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

        progress.update_status(agent_id, ticker, "Computing mean reversion signal")
        mean_reversion = compute_mean_reversion(prices_df)

        progress.update_status(agent_id, ticker, "Computing autocorrelation signal")
        autocorrelation = compute_autocorrelation(prices_df)

        progress.update_status(agent_id, ticker, "Computing microstructure signal")
        microstructure = compute_microstructure(prices_df)

        progress.update_status(agent_id, ticker, "Computing Hurst regime")
        hurst_regime = compute_hurst_regime(prices_df)

        total_score = mean_reversion["score"] + autocorrelation["score"] + microstructure["score"] + hurst_regime["score"]
        max_possible_score = mean_reversion["max_score"] + autocorrelation["max_score"] + microstructure["max_score"] + hurst_regime["max_score"]

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "mean_reversion": mean_reversion,
            "autocorrelation": autocorrelation,
            "microstructure": microstructure,
            "hurst_regime": hurst_regime,
        }

        progress.update_status(agent_id, ticker, "Generating Jim Simons analysis")
        simons_output = generate_simons_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        simons_analysis[ticker] = {
            "signal": simons_output.signal,
            "confidence": simons_output.confidence,
            "reasoning": simons_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=simons_output.reasoning)

    message = HumanMessage(content=json.dumps(simons_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(simons_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = simons_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def compute_mean_reversion(prices_df) -> dict:
    """Z-score of price vs moving average: extreme negative z-score = oversold = stat-arb buy signal."""
    max_score = 4
    if prices_df is None or len(prices_df) < 30:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data"}

    closes = prices_df["close"]
    window = min(63, len(closes) - 1)
    ma = closes.rolling(window).mean()
    std = closes.rolling(window).std()
    z = (closes - ma) / (std + 1e-10)
    z_val = safe_float(z.iloc[-1])

    score = 0
    details = []

    if z_val < -2.0:
        score += 4
        details.append(f"Z-score {z_val:.2f} — strong mean-reversion buy signal")
    elif z_val < -1.0:
        score += 3
        details.append(f"Z-score {z_val:.2f} — moderate oversold")
    elif z_val < 0:
        score += 2
        details.append(f"Z-score {z_val:.2f} — mildly below average")
    elif z_val < 1.0:
        score += 1
        details.append(f"Z-score {z_val:.2f} — near average, neutral")
    else:
        details.append(f"Z-score {z_val:.2f} — above average / overbought, mean-reversion sell")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def compute_autocorrelation(prices_df) -> dict:
    """Lag-1 autocorrelation of returns: negative = mean-reverting (buy dips), positive = momentum."""
    max_score = 4
    if prices_df is None or len(prices_df) < 30:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data"}

    returns = prices_df["close"].pct_change().dropna()
    window = min(63, len(returns))
    recent_returns = returns.iloc[-window:]

    if len(recent_returns) < 10:
        return {"score": 0, "max_score": max_score, "details": "Too few return observations"}

    acf1 = float(np.corrcoef(recent_returns.iloc[:-1].values, recent_returns.iloc[1:].values)[0, 1])

    score = 0
    details = []

    # For stat-arb mean reversion: strong negative autocorrelation is bullish (after recent loss)
    recent_return = safe_float(returns.iloc[-1])
    if acf1 < -0.10 and recent_return < -0.01:
        score += 4
        details.append(f"ACF(1)={acf1:.2f} + recent loss {recent_return:.1%} — mean-reversion buy setup")
    elif acf1 < -0.10 and recent_return > 0.01:
        score += 2
        details.append(f"ACF(1)={acf1:.2f} (mean-reverting) but recent gain — neutral stat-arb")
    elif acf1 < 0:
        score += 2
        details.append(f"ACF(1)={acf1:.2f} — mild mean reversion in returns")
    elif acf1 > 0.10:
        score += 1
        details.append(f"ACF(1)={acf1:.2f} — momentum regime, not stat-arb territory")
    else:
        score += 2
        details.append(f"ACF(1)={acf1:.2f} — near-zero autocorrelation, weak signal")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def compute_microstructure(prices_df) -> dict:
    """Volume anomalies and price-volume relationship as microstructure signals."""
    max_score = 4
    if prices_df is None or len(prices_df) < 21:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data"}

    closes = prices_df["close"]
    volumes = prices_df["volume"]

    vol_ma = volumes.rolling(21).mean()
    vol_ratio = (volumes.iloc[-1] / safe_float(vol_ma.iloc[-1])) if safe_float(vol_ma.iloc[-1]) > 0 else 1.0

    recent_return = (closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5] if closes.iloc[-5] > 0 else 0

    score = 0
    details = []

    # High volume on down move: washout / capitulation = stat-arb buy
    if vol_ratio > 2.0 and recent_return < -0.03:
        score += 4
        details.append(f"Volume spike {vol_ratio:.1f}x on {recent_return:.1%} drop — capitulation / washout")
    # High volume on up move: accumulation signal
    elif vol_ratio > 1.5 and recent_return > 0.02:
        score += 3
        details.append(f"Volume spike {vol_ratio:.1f}x on {recent_return:.1%} gain — accumulation")
    # Normal volume, small move
    elif 0.7 < vol_ratio < 1.3:
        score += 2
        details.append(f"Volume in-line {vol_ratio:.1f}x — no microstructure edge")
    # Low volume decline: not a capitulation, can continue lower
    elif vol_ratio < 0.7 and recent_return < -0.02:
        score += 1
        details.append(f"Low-volume decline {vol_ratio:.1f}x — not a washout signal")
    else:
        score += 2
        details.append(f"Volume ratio {vol_ratio:.1f}x, move {recent_return:.1%} — inconclusive")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def compute_hurst_regime(prices_df) -> dict:
    """Hurst exponent: H < 0.5 = mean-reverting (Renaissance trades), H > 0.5 = trending (sit out)."""
    max_score = 4
    if prices_df is None or len(prices_df) < 40:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data for Hurst"}

    try:
        hurst = calculate_hurst_exponent(prices_df["close"])
    except Exception:
        return {"score": 2, "max_score": max_score, "details": "Hurst calculation error — neutral"}

    score = 0
    details = []

    if hurst < 0.35:
        score += 4
        details.append(f"Hurst={hurst:.2f} — strongly mean-reverting, high stat-arb opportunity")
    elif hurst < 0.45:
        score += 3
        details.append(f"Hurst={hurst:.2f} — mean-reverting regime, favorable for stat-arb")
    elif hurst < 0.55:
        score += 2
        details.append(f"Hurst={hurst:.2f} — near random walk, moderate signal")
    elif hurst < 0.65:
        score += 1
        details.append(f"Hurst={hurst:.2f} — trending regime, reduced stat-arb edge")
    else:
        details.append(f"Hurst={hurst:.2f} — strongly trending, stat-arb likely to lose")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def generate_simons_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "jim_simons_agent",
) -> JimSimonsSignal:
    """Get investment decision from LLM in Jim Simons / Renaissance style: data only, no story."""
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "mean_reversion": analysis_data.get("mean_reversion", {}).get("details"),
        "autocorrelation": analysis_data.get("autocorrelation", {}).get("details"),
        "microstructure": analysis_data.get("microstructure", {}).get("details"),
        "hurst_regime": analysis_data.get("hurst_regime", {}).get("details"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Medallion Fund-style statistical arbitrage system inspired by Jim Simons. Decide bullish, bearish, or neutral using only the quantitative signals provided.\n"
                "\n"
                "Signal framework (no narratives, only data):\n"
                "- Mean reversion: low Z-score = oversold → buy; high Z-score = overbought → sell\n"
                "- Autocorrelation: negative ACF(1) + recent loss = mean-reversion buy setup\n"
                "- Microstructure: high-volume washout = capitulation buy; high-volume accumulation = follow\n"
                "- Hurst regime: H < 0.5 = mean-reverting (our edge); H > 0.5 = trending (no edge here)\n"
                "\n"
                "Signal rules:\n"
                "- Bullish: mean-reverting regime (Hurst < 0.5) + oversold Z-score + negative ACF on loss + capitulation volume\n"
                "- Bearish: overbought Z-score + positive ACF + distribution volume + mean-reverting regime\n"
                "- Neutral: trending regime (no stat-arb edge), mixed signals, or insufficient data\n"
                "\n"
                "Confidence scale:\n"
                "- 90-100%: All four signals strongly aligned in mean-reverting regime\n"
                "- 70-89%: Three signals aligned\n"
                "- 50-69%: Two signals or weak alignment\n"
                "- 30-49%: One signal, rest neutral\n"
                "- 10-29%: Trending regime or all signals bearish\n"
                "\n"
                "Use purely statistical language: Z-score, autocorrelation, Hurst, mean reversion, regime, signal, edge.\n"
                "No fundamental commentary. Keep reasoning under 150 characters. Return JSON only.",
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
        return JimSimonsSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=JimSimonsSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
