from __future__ import annotations

import json
from pathlib import Path
import tempfile

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
