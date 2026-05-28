"""Tests for portfolio manager decision logic: allowed actions and pre-fill/LLM split."""

from unittest.mock import patch

import pytest

from src.agents.portfolio_manager import (
    _PANEL_BLOCK_TILT,
    PortfolioDecision,
    PortfolioManagerOutput,
    compute_allowed_actions,
    generate_trading_decision,
)


def _portfolio(cash: float = 10_000.0, equity: float | None = None, positions: dict | None = None) -> dict:
    p: dict = {
        "cash": cash,
        "positions": positions or {},
        "margin_requirement": 0.5,
        "margin_used": 0.0,
    }
    if equity is not None:
        p["equity"] = equity
    return p


def _state() -> dict:
    return {"messages": [], "data": {}, "metadata": {"show_reasoning": False}}


class TestComputeAllowedActions:
    def test_zero_max_shares_results_in_hold_only(self):
        """max_shares == 0 → no buy/short possible; only hold is returned."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 0},
            portfolio=_portfolio(cash=10_000.0, equity=10_000.0),
        )
        assert result["AAPL"] == {"hold": 0}

    def test_fractional_share_capacity_allows_actions(self):
        """max_shares = 0.5 (fractional) → buy and short actions are present (not hold-only)."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 0.5},
            portfolio=_portfolio(cash=10_000.0, equity=10_000.0),
        )
        assert "buy" in result["AAPL"]
        assert result["AAPL"]["buy"] == pytest.approx(0.5)

    def test_existing_long_adds_sell_action(self):
        """Holding long shares exposes the sell action equal to held quantity."""
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=_portfolio(
                cash=5_000.0,
                equity=6_000.0,
                positions={"AAPL": {"long": 5.0, "long_cost_basis": 100.0, "short": 0.0, "short_cost_basis": 0.0}},
            ),
        )
        assert "sell" in result["AAPL"]
        assert result["AAPL"]["sell"] == pytest.approx(5.0)

    def test_max_short_shares_allows_larger_short_than_long(self):
        """
        Regression for F2: when cash is low (low long cap) but margin capacity is high,
        providing a separate max_short_shares allows a larger short than the long cap.
        """
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 5.0},  # long cap: 5 shares ($500 from cash)
            max_short_shares={"AAPL": 20.0},  # short cap: 20 shares (margin-derived)
            portfolio=_portfolio(
                cash=500.0,
                equity=10_000.0,
                positions={},
            ),
        )
        long_cap = result["AAPL"].get("buy", 0)
        short_cap = result["AAPL"].get("short", 0)
        assert long_cap == pytest.approx(5.0), f"Expected buy=5.0, got {long_cap}"
        assert short_cap > long_cap, f"Short capacity ({short_cap}) should exceed long capacity ({long_cap}) when max_short_shares is higher than max_shares. F2 bug: max_short_shares ignored."

    def test_equity_fallback_uses_cash_when_equity_missing(self):
        """portfolio without 'equity' key falls back to cash for margin calculation (M13 known behaviour)."""
        portfolio_no_equity = {
            "cash": 5_000.0,
            "positions": {},
            "margin_requirement": 0.5,
            "margin_used": 0.0,
            # no 'equity' key
        }
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio=portfolio_no_equity,
        )
        # available_margin = (cash / margin_requirement) - margin_used = (5000 / 0.5) - 0 = 10000
        # max_short_margin = 10000 / 100 = 100; capped to max_qty=10
        assert "short" in result["AAPL"]
        assert result["AAPL"]["short"] == pytest.approx(10.0)


class TestGenerateTradingDecision:
    def test_zero_capacity_ticker_prefilled_not_sent_to_llm(self):
        """All-zero max_shares → all tickers pre-filled as hold; LLM is never called."""
        result = generate_trading_decision(
            tickers=["AAPL", "MSFT"],
            signals_by_ticker={"AAPL": {}, "MSFT": {}},
            current_prices={"AAPL": 150.0, "MSFT": 300.0},
            max_shares={"AAPL": 0, "MSFT": 0},
            portfolio=_portfolio(cash=0.0, equity=0.0),
            agent_id="portfolio_manager",
            state=_state(),
        )
        assert result.decisions["AAPL"].action == "hold"
        assert result.decisions["MSFT"].action == "hold"

    def test_nonzero_capacity_ticker_sent_to_llm(self):
        """Ticker with max_shares > 0 is forwarded to the LLM; mock LLM result is used."""
        mock_output = PortfolioManagerOutput(decisions={"AAPL": PortfolioDecision(action="buy", quantity=1.0, confidence=80, reasoning="bullish")})
        with patch("src.agents.portfolio_manager.call_llm", return_value=mock_output):
            result = generate_trading_decision(
                tickers=["AAPL"],
                signals_by_ticker={"AAPL": {"agent_x": {"sig": "bullish", "conf": 75}}},
                current_prices={"AAPL": 150.0},
                max_shares={"AAPL": 5.0},
                portfolio=_portfolio(cash=10_000.0, equity=10_000.0),
                agent_id="portfolio_manager",
                state=_state(),
            )

        assert result.decisions["AAPL"].action == "buy"
        assert result.decisions["AAPL"].quantity == pytest.approx(1.0)


class TestPMContextBlock:
    """Tests that position context (cost basis, unrealized P&L, recent trades) reaches the LLM prompt."""

    def _call_and_capture_prompt(self, state: dict, portfolio: dict, current_prices: dict) -> str:
        """Run generate_trading_decision with a mocked LLM and return the rendered human prompt text."""
        captured_prompts = []
        mock_output = PortfolioManagerOutput(decisions={"AAPL": PortfolioDecision(action="hold", quantity=0, confidence=50, reasoning="test")})

        def _capture(prompt, **kwargs):
            captured_prompts.append(str(prompt))
            return mock_output

        with patch("src.agents.portfolio_manager.call_llm", side_effect=_capture):
            generate_trading_decision(
                tickers=["AAPL"],
                signals_by_ticker={"AAPL": {"agent_x": {"sig": "bearish", "conf": 60}}},
                current_prices=current_prices,
                max_shares={"AAPL": 5.0},
                portfolio=portfolio,
                agent_id="portfolio_manager",
                state=state,
            )

        assert captured_prompts, "call_llm was never called"
        return captured_prompts[0]

    def test_position_context_includes_cost_basis_and_unrealized_pnl(self):
        """When holding a long at cost 100 and price 90, prompt includes long_cost_basis and unrealized_pnl_pct=-10.0."""
        portfolio = _portfolio(
            cash=5_000.0,
            equity=6_000.0,
            positions={"AAPL": {"long": 10.0, "long_cost_basis": 100.0, "short": 0.0, "short_cost_basis": 0.0}},
        )
        prompt_text = self._call_and_capture_prompt(_state(), portfolio, {"AAPL": 90.0})
        assert '"long_cost_basis":100' in prompt_text, "cost basis missing from prompt"
        assert '"unrealized_pnl_pct":-10.0' in prompt_text, "unrealized P&L missing from prompt"

    def test_recent_trades_forwarded_and_capped_at_five(self):
        """8 recent trades → only the last 5 appear in the prompt."""
        trades = [{"date": f"2026-05-{18 + i:02d}", "action": "buy", "qty": 1.0, "price": 100.0 + i} for i in range(8)]
        state = {"messages": [], "data": {"recent_trades": {"AAPL": trades}}, "metadata": {"show_reasoning": False}}
        portfolio = _portfolio(cash=10_000.0, equity=10_000.0)
        prompt_text = self._call_and_capture_prompt(state, portfolio, {"AAPL": 110.0})
        assert '"recent_trades"' in prompt_text, "recent_trades key missing from prompt"
        # Last 5 are indices 3-7 → dates 05-21 through 05-25
        assert "2026-05-21" in prompt_text, "oldest of the capped 5 missing"
        assert "2026-05-25" in prompt_text, "newest missing"
        assert "2026-05-18" not in prompt_text, "entry beyond cap-5 should be excluded"

    def test_empty_position_context_when_no_position_and_no_history(self):
        """No position and no recent trades → pm_context renders as {} for the ticker."""
        portfolio = _portfolio(cash=10_000.0, equity=10_000.0)
        prompt_text = self._call_and_capture_prompt(_state(), portfolio, {"AAPL": 150.0})
        assert '"AAPL":{}' in prompt_text, "expected empty context entry for AAPL"

    def test_system_prompt_contains_anti_flip_rules(self):
        """System prompt includes anti-flip, structural_split, and slippage language."""
        portfolio = _portfolio(cash=10_000.0, equity=10_000.0)
        prompt_text = self._call_and_capture_prompt(_state(), portfolio, {"AAPL": 150.0})
        assert "Anti-flip" in prompt_text, "Anti-flip rule missing from system prompt"
        assert "structural_split" in prompt_text, "structural_split rule missing from system prompt"
        assert "slippage" in prompt_text, "slippage cost warning missing from system prompt"

    def test_panel_section_appears_in_prompt_when_panel_stats_in_state(self):
        """Panel stats from state['data']['panel_stats'] are rendered into the human prompt."""
        state = {
            "messages": [],
            "data": {"panel_stats": {"AAPL": {"bullish": 2, "bearish": 8, "neutral": 15, "n": 25, "tilt": -0.6}}},
            "metadata": {"show_reasoning": False},
        }
        portfolio = _portfolio(cash=10_000.0, equity=10_000.0)
        prompt_text = self._call_and_capture_prompt(state, portfolio, {"AAPL": 150.0})
        assert '"tilt"' in prompt_text, "tilt key missing from rendered panel section"
        assert '"bull"' in prompt_text, "bull count missing from rendered panel section"


class TestPanelTiltGuard:
    """Tests for the deterministic panel-tilt guard in compute_allowed_actions."""

    def _base_portfolio(self) -> dict:
        return _portfolio(cash=10_000.0, equity=10_000.0)

    def test_negative_tilt_below_threshold_removes_buy(self):
        """tilt <= -BLOCK_TILT with no regime → buy is removed from allowed actions."""
        tilt = -(_PANEL_BLOCK_TILT + 0.1)  # clearly below threshold
        panel_stats = {"AAPL": {"bullish": 2, "bearish": 8, "neutral": 15, "n": 25, "tilt": tilt}}
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 10.0},
            portfolio=self._base_portfolio(),
            panel_stats=panel_stats,
        )
        assert "buy" not in result["AAPL"], f"buy should be blocked when tilt={tilt:.2f}, got {result['AAPL']}"
        assert "hold" in result["AAPL"], "hold must always be present"

    def test_positive_tilt_above_threshold_removes_short(self):
        """tilt >= +BLOCK_TILT with no regime → short is removed from allowed actions."""
        tilt = _PANEL_BLOCK_TILT + 0.1
        panel_stats = {"AAPL": {"bullish": 8, "bearish": 2, "neutral": 15, "n": 25, "tilt": tilt}}
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 10.0},
            portfolio=self._base_portfolio(),
            panel_stats=panel_stats,
        )
        assert "short" not in result["AAPL"], f"short should be blocked when tilt={tilt:.2f}, got {result['AAPL']}"

    def test_tilt_within_threshold_does_not_block(self):
        """tilt between -BLOCK_TILT and +BLOCK_TILT → buy and short both available."""
        tilt = 0.0
        panel_stats = {"AAPL": {"bullish": 5, "bearish": 5, "neutral": 15, "n": 25, "tilt": tilt}}
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 10.0},
            portfolio=self._base_portfolio(),
            panel_stats=panel_stats,
        )
        assert "buy" in result["AAPL"], "buy should not be blocked at tilt=0.0"

    def test_sell_never_blocked_by_tilt(self):
        """Reducing a long position (sell) is always allowed regardless of tilt direction."""
        tilt = -0.9  # strongly bearish panel
        panel_stats = {"AAPL": {"bullish": 1, "bearish": 20, "neutral": 4, "n": 25, "tilt": tilt}}
        portfolio = _portfolio(
            cash=1_000.0,
            equity=11_000.0,
            positions={"AAPL": {"long": 10.0, "long_cost_basis": 100.0, "short": 0.0, "short_cost_basis": 0.0}},
        )
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 5.0},
            portfolio=portfolio,
            panel_stats=panel_stats,
        )
        assert "sell" in result["AAPL"], f"sell should never be blocked by tilt guard, got {result['AAPL']}"

    def test_cover_never_blocked_by_tilt(self):
        """Covering a short position is always allowed regardless of tilt direction."""
        tilt = 0.9  # strongly bullish panel
        panel_stats = {"AAPL": {"bullish": 20, "bearish": 1, "neutral": 4, "n": 25, "tilt": tilt}}
        portfolio = _portfolio(
            cash=1_000.0,
            equity=11_000.0,
            positions={"AAPL": {"long": 0.0, "long_cost_basis": 0.0, "short": 5.0, "short_cost_basis": 100.0}},
        )
        result = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 5.0},
            portfolio=portfolio,
            panel_stats=panel_stats,
        )
        assert "cover" in result["AAPL"], f"cover should never be blocked by tilt guard, got {result['AAPL']}"

    def test_no_panel_stats_does_not_change_actions(self):
        """When panel_stats is None, allowed actions are unchanged (guard is a no-op)."""
        result_no_stats = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 10.0},
            portfolio=self._base_portfolio(),
            panel_stats=None,
        )
        result_empty_stats = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 150.0},
            max_shares={"AAPL": 10.0},
            portfolio=self._base_portfolio(),
            panel_stats={},
        )
        assert "buy" in result_no_stats["AAPL"]
        assert result_no_stats["AAPL"].keys() == result_empty_stats["AAPL"].keys()

    def test_msft_day1_bearish_panel_blocks_buy(self):
        """Regression: MSFT day-1 scenario (8 bearish/15 neutral/2 bullish, tilt≈-0.6) must block buy."""
        panel_stats = {"MSFT": {"bullish": 2, "bearish": 8, "neutral": 15, "n": 25, "tilt": -0.6}}
        result = compute_allowed_actions(
            tickers=["MSFT"],
            current_prices={"MSFT": 428.97},
            max_shares={"MSFT": 37.0},
            portfolio=_portfolio(cash=100_000.0, equity=100_000.0),
            panel_stats=panel_stats,
        )
        assert "buy" not in result["MSFT"], f"MSFT day-1 buy must be blocked by panel tilt guard, got {result['MSFT']}"
