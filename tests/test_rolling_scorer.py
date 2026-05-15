from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.feedback.loader import load_weights
from src.feedback.scorer import compute_weights


def _make_labeled_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _build_records(agent_id: str, n_hits: int, n_misses: int, start_date: int = 20250101) -> list[dict]:
    recs = []
    for i in range(n_hits):
        recs.append({"agent_id": agent_id, "date": f"{start_date + i}", "signal": "bullish", "return_5d": 0.01})
    for i in range(n_misses):
        recs.append({"agent_id": agent_id, "date": f"{start_date + n_hits + i}", "signal": "bullish", "return_5d": -0.01})
    return recs


def test_weights_direction_and_mean():
    with tempfile.TemporaryDirectory() as tmpdir:
        labeled = Path(tmpdir) / "labeled.jsonl"
        weights_path = Path(tmpdir) / "weights.json"
        report_path = Path(tmpdir) / "report.json"

        recs = _build_records("agent_a", n_hits=15, n_misses=5)
        recs += _build_records("agent_b", n_hits=5, n_misses=15, start_date=20250201)
        _make_labeled_jsonl(labeled, recs)

        weights = compute_weights(
            labeled_path=str(labeled),
            horizon=5,
            window_days=60,
            weights_path=str(weights_path),
            report_path=str(report_path),
            min_samples=5,
        )

        assert weights["agent_a"] > 1.0
        assert weights["agent_b"] < 1.0
        mean_w = sum(weights.values()) / len(weights)
        assert abs(mean_w - 1.0) < 1e-9


def test_output_files_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        labeled = Path(tmpdir) / "labeled.jsonl"
        weights_path = Path(tmpdir) / "weights.json"
        report_path = Path(tmpdir) / "report.json"

        recs = _build_records("agent_a", n_hits=10, n_misses=10)
        _make_labeled_jsonl(labeled, recs)

        compute_weights(
            labeled_path=str(labeled),
            weights_path=str(weights_path),
            report_path=str(report_path),
        )

        assert weights_path.exists()
        assert report_path.exists()

        w = json.loads(weights_path.read_text())
        assert "agent_a" in w

        r = json.loads(report_path.read_text())
        assert "agent_a" in r
        assert {"hit_rate", "samples", "weight"} == set(r["agent_a"].keys())


def test_low_sample_agent_gets_neutral_weight():
    with tempfile.TemporaryDirectory() as tmpdir:
        labeled = Path(tmpdir) / "labeled.jsonl"
        weights_path = Path(tmpdir) / "weights.json"
        report_path = Path(tmpdir) / "report.json"

        # agent_a: only 3 samples (< min_samples=5) → should get weight 1.0
        recs = _build_records("agent_a", n_hits=3, n_misses=0)
        _make_labeled_jsonl(labeled, recs)

        weights = compute_weights(
            labeled_path=str(labeled),
            weights_path=str(weights_path),
            report_path=str(report_path),
            min_samples=5,
        )

        # Only one agent, falls back to prior → normalized weight = 1.0
        assert abs(weights["agent_a"] - 1.0) < 1e-9


def test_empty_file_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        labeled = Path(tmpdir) / "labeled.jsonl"
        labeled.write_text("")
        weights_path = Path(tmpdir) / "weights.json"
        report_path = Path(tmpdir) / "report.json"

        weights = compute_weights(
            labeled_path=str(labeled),
            weights_path=str(weights_path),
            report_path=str(report_path),
        )

        assert weights == {}


def test_scorer_keys_match_debate_node_convention():
    """scorer → weights.json keys must not have _agent suffix; loader must strip legacy suffix.

    The debate_node strips '_agent' before looking up agent_weights. If weights.json
    is keyed with the suffix the lookup silently falls back to 1.0 and conviction
    weights are never applied (regression test for the suffix-mismatch bug).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        labeled = Path(tmpdir) / "labeled.jsonl"
        weights_path = Path(tmpdir) / "weights.json"
        report_path = Path(tmpdir) / "report.json"

        # signal_log writes agent_id with _agent suffix (LangGraph node name convention)
        recs = _build_records("agent_a_agent", n_hits=15, n_misses=5)
        recs += _build_records("agent_b_agent", n_hits=5, n_misses=15, start_date=20250201)
        _make_labeled_jsonl(labeled, recs)

        compute_weights(
            labeled_path=str(labeled),
            horizon=5,
            window_days=60,
            weights_path=str(weights_path),
            report_path=str(report_path),
            min_samples=5,
        )

        # weights.json should be keyed without _agent suffix
        raw = json.loads(weights_path.read_text())
        assert "agent_a" in raw, "weights.json must use bare agent names (no _agent suffix)"
        assert "agent_a_agent" not in raw, "weights.json must not contain _agent suffix"

        # load_weights must also normalise legacy files written with the old suffix
        weights_path.write_text(json.dumps({"agent_a_agent": 1.5, "agent_b_agent": 0.7}))
        loaded = load_weights(str(weights_path))
        assert "agent_a" in loaded
        assert "agent_a_agent" not in loaded
        assert loaded["agent_a"] == 1.5

        # Keys returned by load_weights must be usable directly by the debate node's
        # stripped lookup (agent.replace("_agent", "") → same key as in loaded dict)
        for key in loaded:
            assert not key.endswith("_agent"), f"loaded key {key!r} still has suffix"
