import json

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

# Minimum absolute panel tilt to block trading against the consensus directionally.
# tilt = (bull_conf - bear_conf) / (bull_conf + bear_conf), range [-1, 1].
# 0.34 ≈ roughly 2:1 directional imbalance (e.g. 8 bearish vs 4 bullish at equal weight).
# Applies independently of regime classification.
_PANEL_BLOCK_TILT: float = 0.34


class PortfolioDecision(BaseModel):
    action: Literal["hold", "buy", "sell", "short", "cover"]
    quantity: float = Field(description="Number of shares to trade. Fractional quantities are supported (e.g. 1.5).")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """Makes final trading decisions and generates orders for multiple tickers"""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    short_position_limits = {}
    current_prices = {}
    max_shares = {}
    max_short_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")

        # Find the corresponding risk manager for this portfolio manager
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split("_")[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # Fallback for CLI

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        short_position_limits[ticker] = risk_data.get("max_short_position_size", position_limits[ticker])
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = position_limits[ticker] / current_prices[ticker]
            max_short_shares[ticker] = short_position_limits[ticker] / current_prices[ticker]
        else:
            max_shares[ticker] = 0
            max_short_shares[ticker] = 0

        # Compress group-aggregated signals to {group: {sig, conf, dissent}}
        group_signals = state["data"].get("group_signals", {})
        ticker_signals = {}
        for group, gdata in group_signals.get(ticker, {}).items():
            sig = gdata.get("signal")
            conf = gdata.get("confidence")
            if sig is not None and conf is not None:
                ticker_signals[group] = {"sig": sig, "conf": conf, "dissent": gdata.get("dissent", 0)}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Generating trading decisions")

    regime = state.get("metadata", {}).get("regime")
    panel_stats = state["data"].get("panel_stats") or {}
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        max_short_shares=max_short_shares,
        portfolio=portfolio,
        agent_id=agent_id,
        state=state,
        regime=regime,
        panel_stats=panel_stats,
    )
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
    }


def compute_allowed_actions(
    tickers: list[str],
    current_prices: dict[str, float],
    max_shares: dict[str, float],
    portfolio: dict[str, float],
    regime: str | None = None,
    group_signals: dict[str, dict] | None = None,
    max_short_shares: dict[str, float] | None = None,
    panel_stats: dict[str, dict] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute allowed actions and max quantities for each ticker deterministically."""
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}
    margin_requirement = float(portfolio.get("margin_requirement", 0.5))
    margin_used = float(portfolio.get("margin_used", 0.0))
    equity = float(portfolio.get("equity", cash))

    # Running totals prevent double-spending across tickers in the same cycle.
    # remaining_short_capacity is in notional dollars: (equity − margin_used) / margin_req.
    remaining_cash = cash
    remaining_short_capacity = max(0.0, (equity - margin_used) / margin_requirement) if margin_requirement > 0 else 0.0

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(
            ticker,
            {"long": 0.0, "long_cost_basis": 0.0, "short": 0.0, "short_cost_basis": 0.0},
        )
        long_shares = float(pos.get("long", 0) or 0)
        short_shares = float(pos.get("short", 0) or 0)
        max_qty = float(max_shares.get(ticker, 0) or 0)
        # Use a separate short cap if provided; fall back to the long cap.
        max_short_qty = float((max_short_shares or {}).get(ticker, 0) or 0) if max_short_shares is not None else max_qty

        # Start with zeros
        actions: dict[str, float] = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}

        # Long side
        if long_shares > 0:
            actions["sell"] = long_shares
        if remaining_cash > 0 and price > 0:
            max_buy_cash = remaining_cash / price
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy
                remaining_cash -= max_buy * price  # reserve so later tickers see the residual

        # Short side
        if short_shares > 0:
            actions["cover"] = short_shares
        if price > 0 and max_short_qty > 0:
            if margin_requirement <= 0.0:
                # If margin requirement is zero or unset, only cap by max_short_qty
                max_short = max_short_qty
            else:
                max_short_margin = remaining_short_capacity / price
                max_short = max(0, min(max_short_qty, max_short_margin))
            if max_short > 0:
                actions["short"] = max_short
                remaining_short_capacity -= max_short * price  # consume notional capacity

        # Hold always valid
        actions["hold"] = 0

        # Prune zero-capacity actions to reduce tokens, keep hold
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v

        # Regime gate — deterministically block actions that fight the prevailing trend.
        # "cover" is always allowed (closing a short is not fighting the trend).
        # "sell" is always allowed (reducing a long is not fighting the trend).
        if regime is not None and group_signals is not None:
            ticker_groups = group_signals.get(ticker, {})
            if regime == "bull_trend":
                quant_bull = ticker_groups.get("quant_systematic", {}).get("signal") == "bullish"
                growth_bull = ticker_groups.get("growth_and_catalyst", {}).get("signal") == "bullish"
                if quant_bull or growth_bull:
                    pruned.pop("short", None)
            elif regime == "bear_trend":
                quant_bear = ticker_groups.get("quant_systematic", {}).get("signal") == "bearish"
                quality_bear = ticker_groups.get("quality_compounders", {}).get("signal") == "bearish"
                if quant_bear or quality_bear:
                    pruned.pop("buy", None)
            elif regime == "risk_off":
                pruned.pop("buy", None)
                pruned.pop("short", None)

        # Panel-tilt guard — blocks new longs/shorts that fight a clearly net-directional
        # analyst panel.  Operates independently of regime classification so it fires even
        # when regime detection is disabled.  Reducing positions (sell/cover) is always
        # allowed regardless of tilt.
        if panel_stats is not None:
            tilt = float((panel_stats.get(ticker) or {}).get("tilt", 0.0))
            if tilt <= -_PANEL_BLOCK_TILT:
                pruned.pop("buy", None)
            elif tilt >= _PANEL_BLOCK_TILT:
                pruned.pop("short", None)

        allowed[ticker] = pruned

    return allowed


def _compact_signals(signals_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """Keep only {agent: {sig, conf}} and drop empty agents."""
    out = {}
    for t, agents in signals_by_ticker.items():
        if not agents:
            out[t] = {}
            continue
        compact = {}
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal")
            conf = payload.get("conf") if "conf" in payload else payload.get("confidence")
            if sig is not None and conf is not None:
                compact[agent] = {"sig": sig, "conf": conf}
        out[t] = compact
    return out


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, float],
    portfolio: dict[str, float],
    agent_id: str,
    state: AgentState,
    regime: str | None = None,
    max_short_shares: dict[str, float] | None = None,
    panel_stats: dict[str, dict] | None = None,
) -> PortfolioManagerOutput:
    """Get decisions from the LLM with deterministic constraints and a minimal prompt."""

    group_signals = state["data"].get("group_signals", {})
    # Resolve panel_stats: explicit arg takes precedence; fall back to state for direct callers.
    if panel_stats is None:
        panel_stats = state["data"].get("panel_stats") or {}
    # Deterministic constraints (regime gate + panel-tilt guard applied here)
    allowed_actions_full = compute_allowed_actions(
        tickers,
        current_prices,
        max_shares,
        portfolio,
        regime=regime,
        group_signals=group_signals,
        max_short_shares=max_short_shares,
        panel_stats=panel_stats,
    )

    # Pre-fill pure holds to avoid sending them to the LLM at all
    prefilled_decisions: dict[str, PortfolioDecision] = {}
    tickers_for_llm: list[str] = []
    for t in tickers:
        aa = allowed_actions_full.get(t, {"hold": 0})
        # If only 'hold' key exists, there is no trade possible
        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=100.0, reasoning="No valid trade available")
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return PortfolioManagerOutput(decisions=prefilled_decisions)

    # Build compact payloads only for tickers sent to LLM
    compact_signals = _compact_signals({t: signals_by_ticker.get(t, {}) for t in tickers_for_llm})
    compact_allowed = {t: allowed_actions_full[t] for t in tickers_for_llm}

    # Forward actual position shape so the LLM doesn't hallucinate which side is held
    positions = portfolio.get("positions", {}) or {}
    current_positions = {}
    for t in tickers_for_llm:
        pos = positions.get(t, {}) or {}
        current_positions[t] = {"long": int(pos.get("long", 0) or 0), "short": int(pos.get("short", 0) or 0)}

    # Build per-ticker position context: cost basis, unrealized P&L, and recent trade history.
    # This anchors the LLM to its own prior decisions so it doesn't flip on unchanged signals.
    recent_trades_data = state["data"].get("recent_trades") or {}
    pm_context: dict[str, dict] = {}
    for t in tickers_for_llm:
        pos = positions.get(t, {}) or {}
        price = float(current_prices.get(t, 0.0))
        long_shares = float(pos.get("long", 0) or 0)
        short_shares = float(pos.get("short", 0) or 0)
        long_cb = float(pos.get("long_cost_basis", 0) or 0)
        short_cb = float(pos.get("short_cost_basis", 0) or 0)
        entry: dict = {}
        if long_shares > 0 and long_cb > 0 and price > 0:
            entry["long_shares"] = round(long_shares, 4)
            entry["long_cost_basis"] = round(long_cb, 2)
            entry["unrealized_pnl_pct"] = round((price - long_cb) / long_cb * 100, 2)
        if short_shares > 0 and short_cb > 0 and price > 0:
            entry["short_shares"] = round(short_shares, 4)
            entry["short_cost_basis"] = round(short_cb, 2)
            entry["unrealized_pnl_pct"] = round((short_cb - price) / short_cb * 100, 2)
        rt = (recent_trades_data.get(t) or [])[-5:]
        if rt:
            entry["recent_trades"] = rt
        pm_context[t] = entry

    # Include group debate summaries for contested tickers if available
    raw_debates = state["data"].get("debate_summaries", {})
    relevant_debates = {}
    for t in tickers_for_llm:
        if t not in raw_debates:
            continue
        d = raw_debates[t]
        # Compact to essential fields only
        relevant_debates[t] = {
            "consensus_strength": d.get("consensus_strength"),
            "core_disagreement": d.get("core_disagreement"),
            "group_positions": [{"group": gp["group"], "stance": gp["stance"]} for gp in d.get("group_positions", [])],
        }
    debate_section = "\nGroup debate (contested tickers):\n" + json.dumps(relevant_debates, separators=(",", ":"), ensure_ascii=False) + "\n" if relevant_debates else ""

    # Compact panel summary for each ticker: bull/bear/neutral counts and confidence-weighted tilt.
    # tilt ∈ [-1, 1]: negative = net-bearish panel, positive = net-bullish panel.
    compact_panel: dict[str, dict] = {}
    for t in tickers_for_llm:
        stats = (panel_stats or {}).get(t)
        if stats:
            compact_panel[t] = {
                "bull": stats["bullish"],
                "bear": stats["bearish"],
                "neut": stats["neutral"],
                "tilt": stats["tilt"],
            }
    panel_section = "\nPanel:\n" + json.dumps(compact_panel, separators=(",", ":"), ensure_ascii=False) + "\n" if compact_panel else ""

    regime_instruction = ""
    if regime == "bull_trend":
        regime_instruction = "\nMarket regime: BULL_TREND. Avoid new shorts — the allowed actions already block them when momentum/growth groups are bullish. Lean long; hold existing shorts only if quant and growth are both decisively bearish."
    elif regime == "bear_trend":
        regime_instruction = "\nMarket regime: BEAR_TREND. Avoid new longs — the allowed actions already block them when quant/quality groups are bearish. Lean short or cash; hold existing longs only if quant and quality are both decisively bullish."
    elif regime == "risk_off":
        regime_instruction = "\nMarket regime: RISK_OFF. Reduce exposure — new buys and new shorts are blocked. Prefer sell/cover/hold to protect capital."

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a portfolio manager. For each ticker pick exactly one allowed action and a quantity ≤ the listed max (fractional allowed).\n"
                "\n"
                "Inputs per ticker:\n"
                "- strategy-group signals (each group aggregates multiple analysts) and optional group debate context\n"
                "- Positions: current long/short shares (integer view)\n"
                "- Position context: cost basis, unrealized P&L %, and the last up-to-5 executed trades for this ticker\n"
                "- Allowed actions with max qty (already validated for cash/margin/regime)\n"
                "\n"
                "Decision rules — apply in order:\n"
                "1) Every round-trip costs ~10-20 bps in slippage. Do not trade unless expected edge clearly exceeds that cost.\n"
                "2) Anti-flip: if recent_trades shows you opened a position in the last 1-3 cycles, reversing it (sell after a recent buy; cover after a recent short) is almost never correct. Only flip if a specific group's signal changed materially since that trade — name which group in your reasoning.\n"
                '3) When debate consensus_strength == "structural_split" and an existing position aligns with one of the camps, prefer hold over flipping. A split debate is not new information.\n'
                "4) Do not chase: closing a losing position only to re-open it at a similar price is forbidden. If you held through a drawdown, keep holding unless a group signal materially changed.\n"
                "5) Reference the actual current position and unrealized_pnl_pct in your reasoning. Do not invent positions that don't exist.\n"
                "6) Panel tilt: the Panel block shows bull/bear/neutral counts and a confidence-weighted tilt. "
                "Do not open a new long when the panel is net-bearish (tilt < −0.3), or a new short when net-bullish (tilt > +0.3), "
                "unless a specific high-confidence group clearly justifies the exception — name that group.\n"
                "\n"
                "Reasoning: max 120 chars, must cite either (a) the group whose signal drives the action or (b) the rule above that justifies holding. Return JSON only."
                "{regime_instruction}",
            ),
            ("human", 'Signals:\n{signals}\n{debate_section}{panel_section}Positions:\n{positions}\nPosition context:\n{pm_context}\nAllowed:\n{allowed}\n\nFormat:\n{{\n  "decisions": {{\n    "TICKER": {{"action":"...","quantity":float,"confidence":int,"reasoning":"..."}}\n  }}\n}}'),
        ]
    )

    prompt_data = {
        "signals": json.dumps(compact_signals, separators=(",", ":"), ensure_ascii=False),
        "debate_section": debate_section,
        "panel_section": panel_section,
        "positions": json.dumps(current_positions, separators=(",", ":"), ensure_ascii=False),
        "pm_context": json.dumps(pm_context, separators=(",", ":"), ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, separators=(",", ":"), ensure_ascii=False),
        "regime_instruction": regime_instruction,
    }
    prompt = template.invoke(prompt_data)

    # Default factory fills remaining tickers as hold if the LLM fails
    def create_default_portfolio_output():
        # start from prefilled
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Default decision: hold")
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    # Merge prefilled holds with LLM results
    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)
    return PortfolioManagerOutput(decisions=merged)
