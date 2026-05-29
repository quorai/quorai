"""Tests for src/backtesting/attribution.py.

Uses a synthetic labeled JSONL signal log so no real data / LLM calls are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.backtesting.attribution import (
    _conf_weighted_mean,
    _directional_spread,
    compute_attribution,
)

# ---------------------------------------------------------------------------
# Helper to write a synthetic labeled signal log
# ---------------------------------------------------------------------------


def _write_log(records: list[dict], tmp_path: Path) -> str:
    log = tmp_path / "signals.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records))
    return str(log)


def _record(
    agent_id: str,
    signal: str,
    return_5d: float | None,
    confidence: float = 80.0,
    ticker: str = "AAPL",
    date: str = "2025-01-01",
) -> dict:
    return {
        "agent_id": agent_id,
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "date": date,
        "return_1d": return_5d,
        "return_5d": return_5d,
        "return_20d": return_5d,
    }


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestDirectionalSpread:
    def test_positive_spread(self):
        spread = _directional_spread([0.05, 0.03], [-0.04, -0.02])
        assert spread is not None
        assert spread > 0

    def test_negative_spread(self):
        spread = _directional_spread([-0.05], [0.05])
        assert spread is not None
        assert spread < 0

    def test_none_when_no_bull(self):
        assert _directional_spread([], [-0.01]) is None

    def test_none_when_no_bear(self):
        assert _directional_spread([0.01], []) is None


class TestConfWeightedMean:
    def test_simple_case(self):
        records = [
            {"signal": "bullish", "confidence": 100.0, "return_5d": 0.1},
            {"signal": "bearish", "confidence": 100.0, "return_5d": -0.1},
        ]
        cwm = _conf_weighted_mean(records, "return_5d")
        # Both contribute positively (bull: +1 * 100 * 0.1; bear: -1 * 100 * -0.1)
        assert cwm is not None
        assert cwm > 0

    def test_neutral_signals_excluded(self):
        records = [
            {"signal": "neutral", "confidence": 100.0, "return_5d": 0.5},
        ]
        assert _conf_weighted_mean(records, "return_5d") is None

    def test_none_return_excluded(self):
        records = [
            {"signal": "bullish", "confidence": 100.0, "return_5d": None},
        ]
        assert _conf_weighted_mean(records, "return_5d") is None


# ---------------------------------------------------------------------------
# compute_attribution integration tests
# ---------------------------------------------------------------------------


class TestComputeAttribution:
    def test_basic_hit_rate_and_spread(self, tmp_path):
        """An analyst who is always right should have hit_rate=1 and positive spread."""
        records = [
            _record("warren_buffett_agent", "bullish", 0.05, date="2025-01-01"),
            _record("warren_buffett_agent", "bullish", 0.03, date="2025-01-02"),
            _record("warren_buffett_agent", "bearish", -0.04, date="2025-01-03"),
            _record("warren_buffett_agent", "bearish", -0.02, date="2025-01-04"),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)

        analysts = [e for e in report if not e["agent_id"].startswith("[GROUP]")]
        assert len(analysts) == 1
        entry = analysts[0]
        assert entry["hit_rate"] == 1.0
        assert entry["directional_spread"] is not None
        assert entry["directional_spread"] > 0
        assert entry["sample_count"] == 4

    def test_bad_analyst_has_low_hit_rate(self, tmp_path):
        """Analyst whose signals are always wrong should have hit_rate=0."""
        records = [
            _record("cathie_wood_agent", "bullish", -0.05, date="2025-01-01"),
            _record("cathie_wood_agent", "bearish", 0.05, date="2025-01-02"),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)
        analysts = [e for e in report if not e["agent_id"].startswith("[GROUP]")]
        assert analysts[0]["hit_rate"] == 0.0
        assert analysts[0]["directional_spread"] is not None
        assert analysts[0]["directional_spread"] < 0

    def test_neutral_signals_excluded(self, tmp_path):
        records = [
            _record("warren_buffett_agent", "neutral", 0.05),
            _record("warren_buffett_agent", "neutral", -0.03),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)
        assert report == []

    def test_empty_log_returns_empty(self, tmp_path):
        log = _write_log([], tmp_path)
        assert compute_attribution(log, horizon=5) == []

    def test_none_returns_excluded_from_scoring(self, tmp_path):
        records = [
            _record("warren_buffett_agent", "bullish", None),
            _record("warren_buffett_agent", "bullish", 0.05),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)
        analysts = [e for e in report if not e["agent_id"].startswith("[GROUP]")]
        # Only the non-None record should be counted.
        assert analysts[0]["sample_count"] == 1

    def test_group_rollup_present(self, tmp_path):
        """Group-level entries should appear in the output when analysts belong to a group."""
        records = [
            _record("warren_buffett_agent", "bullish", 0.05),
            _record("warren_buffett_agent", "bearish", -0.03),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)
        group_entries = [e for e in report if e["agent_id"].startswith("[GROUP]")]
        # warren_buffett belongs to deep_value group.
        assert len(group_entries) >= 1
        group_names = [e["group"] for e in group_entries]
        # warren_buffett belongs to quality_compounders group (see ANALYST_CONFIG).
        assert "quality_compounders" in group_names

    def test_output_written_to_file(self, tmp_path):
        records = [
            _record("warren_buffett_agent", "bullish", 0.05),
            _record("warren_buffett_agent", "bearish", -0.03),
        ]
        log = _write_log(records, tmp_path)
        out = str(tmp_path / "attr.json")
        compute_attribution(log, horizon=5, output_path=out)
        data = json.loads(Path(out).read_text())
        assert "analysts" in data
        assert data["horizon_days"] == 5
        assert "baseline_mean_return" in data

    def test_sorted_by_spread_descending(self, tmp_path):
        """Best analyst (highest spread) should appear first."""
        records = [
            # good analyst: consistently right
            _record("warren_buffett_agent", "bullish", 0.10, date="2025-01-01"),
            _record("warren_buffett_agent", "bearish", -0.10, date="2025-01-02"),
            # bad analyst: consistently wrong
            _record("cathie_wood_agent", "bullish", -0.10, date="2025-01-03"),
            _record("cathie_wood_agent", "bearish", 0.10, date="2025-01-04"),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)
        analysts = [e for e in report if not e["agent_id"].startswith("[GROUP]")]
        assert len(analysts) == 2
        assert analysts[0]["directional_spread"] > analysts[1]["directional_spread"]
        assert analysts[0]["agent_id"] == "warren_buffett"

    def test_multiple_tickers(self, tmp_path):
        """Attribution aggregates across all tickers for each analyst."""
        records = [
            _record("warren_buffett_agent", "bullish", 0.05, ticker="AAPL"),
            _record("warren_buffett_agent", "bullish", 0.03, ticker="MSFT"),
            _record("warren_buffett_agent", "bearish", -0.04, ticker="AAPL"),
        ]
        log = _write_log(records, tmp_path)
        report = compute_attribution(log, horizon=5)
        analysts = [e for e in report if not e["agent_id"].startswith("[GROUP]")]
        assert analysts[0]["sample_count"] == 3


# ---------------------------------------------------------------------------
# Gate-toggle tests (B2)
# ---------------------------------------------------------------------------


class TestPMGateToggles:
    """Verify that QUORAI_GATE_* env vars disable individual PM gates."""

    def test_regime_gate_off_allows_short_in_bull_trend(self, monkeypatch):
        monkeypatch.setenv("QUORAI_GATE_REGIME", "0")
        from src.agents.portfolio_manager import compute_allowed_actions

        group_signals = {
            "AAPL": {
                "quant_systematic": {"signal": "bullish"},
                "growth_and_catalyst": {"signal": "bullish"},
            }
        }
        allowed = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio={"cash": 10_000, "equity": 10_000, "margin_used": 0, "margin_requirement": 0.5, "positions": {}},
            regime="bull_trend",
            group_signals=group_signals,
            max_short_shares={"AAPL": 5.0},
        )
        # With gate OFF, short should remain available even in bull_trend.
        assert "short" in allowed["AAPL"]

    def test_regime_gate_on_removes_short_in_bull_trend(self, monkeypatch):
        monkeypatch.delenv("QUORAI_GATE_REGIME", raising=False)
        from src.agents.portfolio_manager import compute_allowed_actions

        group_signals = {
            "AAPL": {
                "quant_systematic": {"signal": "bullish"},
            }
        }
        allowed = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio={"cash": 10_000, "equity": 10_000, "margin_used": 0, "margin_requirement": 0.5, "positions": {}},
            regime="bull_trend",
            group_signals=group_signals,
            max_short_shares={"AAPL": 5.0},
        )
        assert "short" not in allowed["AAPL"]

    def test_panel_gate_off_allows_buy_against_bearish_panel(self, monkeypatch):
        monkeypatch.setenv("QUORAI_GATE_PANEL", "0")
        from src.agents.portfolio_manager import compute_allowed_actions

        panel_stats = {"AAPL": {"bullish": 2, "bearish": 18, "neutral": 5, "tilt": -0.80}}
        allowed = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio={"cash": 10_000, "equity": 10_000, "margin_used": 0, "margin_requirement": 0.5, "positions": {}},
            panel_stats=panel_stats,
        )
        assert "buy" in allowed["AAPL"]

    def test_panel_gate_on_blocks_buy_against_bearish_panel(self, monkeypatch):
        monkeypatch.delenv("QUORAI_GATE_PANEL", raising=False)
        from src.agents.portfolio_manager import compute_allowed_actions

        panel_stats = {"AAPL": {"bullish": 2, "bearish": 18, "neutral": 5, "tilt": -0.80}}
        allowed = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 100.0},
            max_shares={"AAPL": 10.0},
            portfolio={"cash": 10_000, "equity": 10_000, "margin_used": 0, "margin_requirement": 0.5, "positions": {}},
            panel_stats=panel_stats,
        )
        assert "buy" not in allowed["AAPL"]

    def test_min_hold_gate_off_allows_immediate_sell(self, monkeypatch):
        monkeypatch.setenv("QUORAI_GATE_MIN_HOLD", "0")
        from src.agents.portfolio_manager import compute_allowed_actions

        recent_trades = {"AAPL": [{"action": "buy", "qty": 5, "price": 100.0, "date": "2025-01-01"}]}
        portfolio = {
            "cash": 5_000,
            "equity": 10_000,
            "margin_used": 0,
            "margin_requirement": 0.5,
            "positions": {"AAPL": {"long": 5.0, "short": 0.0, "long_cost_basis": 100.0, "short_cost_basis": 0.0}},
        }
        allowed = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 105.0},
            max_shares={"AAPL": 50.0},
            portfolio=portfolio,
            recent_trades=recent_trades,
        )
        assert "sell" in allowed["AAPL"]

    def test_min_hold_gate_on_blocks_immediate_sell(self, monkeypatch):
        monkeypatch.delenv("QUORAI_GATE_MIN_HOLD", raising=False)
        from src.agents.portfolio_manager import compute_allowed_actions

        recent_trades = {"AAPL": [{"action": "buy", "qty": 5, "price": 100.0, "date": "2025-01-01"}]}
        portfolio = {
            "cash": 5_000,
            "equity": 10_000,
            "margin_used": 0,
            "margin_requirement": 0.5,
            "positions": {"AAPL": {"long": 5.0, "short": 0.0, "long_cost_basis": 100.0, "short_cost_basis": 0.0}},
        }
        allowed = compute_allowed_actions(
            tickers=["AAPL"],
            current_prices={"AAPL": 105.0},
            max_shares={"AAPL": 50.0},
            portfolio=portfolio,
            recent_trades=recent_trades,
        )
        assert "sell" not in allowed["AAPL"]
