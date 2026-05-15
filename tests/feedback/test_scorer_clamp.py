"""Tests that compute_weights clamps per-agent conviction weights to [min, max]."""

import json
import os

import pytest


def _write_labeled_log(tmp_path, records: list[dict]) -> str:
    p = tmp_path / "labeled.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(p)


def _make_record(agent: str, signal: str, ret: float, date: str = "2024-01-01") -> dict:
    return {"agent_id": f"{agent}_agent", "ticker": "AAPL", "signal": signal, "date": date, "return_5d": ret}


class TestWeightClamping:
    def test_weight_not_above_max(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUORAI_AGENT_WEIGHT_MAX", "3.0")
        monkeypatch.setenv("QUORAI_AGENT_WEIGHT_MIN", "0.1")
        # Reload module-level constants from fresh env
        import importlib
        import src.feedback.scorer as scorer_mod

        importlib.reload(scorer_mod)

        # perfect agent: 10 hits out of 10
        records = [_make_record("great_agent", "bullish", 0.01, f"2024-01-{i:02d}") for i in range(1, 11)]
        # mediocre agents: 5 hits out of 10 each (to lower the mean)
        for j in range(8):
            for i in range(1, 11):
                records.append(_make_record(f"avg_{j}", "bullish" if i <= 5 else "bearish", 0.01, f"2024-01-{i:02d}"))

        log_path = _write_labeled_log(tmp_path, records)
        weights_path = str(tmp_path / "weights.json")
        weights = scorer_mod.compute_weights(log_path, horizon=5, weights_path=weights_path, report_path=str(tmp_path / "report.json"))

        assert all(v <= 3.0 for v in weights.values()), weights
        assert all(v >= 0.1 for v in weights.values()), weights

    def test_weight_not_below_min(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUORAI_AGENT_WEIGHT_MAX", "3.0")
        monkeypatch.setenv("QUORAI_AGENT_WEIGHT_MIN", "0.1")
        import importlib
        import src.feedback.scorer as scorer_mod

        importlib.reload(scorer_mod)

        # terrible agent: 0 hits out of 10
        records = [_make_record("bad_agent", "bullish", -0.01, f"2024-01-{i:02d}") for i in range(1, 11)]
        # perfect agents: 10/10 each
        for j in range(8):
            for i in range(1, 11):
                records.append(_make_record(f"great_{j}", "bullish", 0.01, f"2024-01-{i:02d}"))

        log_path = _write_labeled_log(tmp_path, records)
        weights_path = str(tmp_path / "weights.json")
        weights = scorer_mod.compute_weights(log_path, horizon=5, weights_path=weights_path, report_path=str(tmp_path / "report.json"))

        assert weights["bad_agent"] >= 0.1

    def test_env_override_max(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUORAI_AGENT_WEIGHT_MAX", "2.0")
        monkeypatch.setenv("QUORAI_AGENT_WEIGHT_MIN", "0.1")
        import importlib
        import src.feedback.scorer as scorer_mod

        importlib.reload(scorer_mod)

        records = [_make_record("great_agent", "bullish", 0.01, f"2024-01-{i:02d}") for i in range(1, 11)]
        for j in range(8):
            for i in range(1, 11):
                records.append(_make_record(f"avg_{j}", "bullish" if i <= 5 else "bearish", 0.01, f"2024-01-{i:02d}"))

        log_path = _write_labeled_log(tmp_path, records)
        weights_path = str(tmp_path / "weights.json")
        weights = scorer_mod.compute_weights(log_path, horizon=5, weights_path=weights_path, report_path=str(tmp_path / "report.json"))

        assert all(v <= 2.0 for v in weights.values())
