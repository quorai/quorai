"""End-to-end smoke test: SignalLogger output → label_signals → compute_weights → load_weights."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.feedback import compute_weights, label_signals, load_weights


def _price_df(start: str = "2025-01-01", periods: int = 60, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="B")
    return pd.DataFrame({"close": [base + i * 0.5 for i in range(periods)]}, index=idx)


def _write_signal_log(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _build_signal_records(price_df: pd.DataFrame, agent_ids: list[str], n_records_per_agent: int = 10) -> list[dict]:
    """Create synthetic bullish signals on early dates (all expected to be hits since price trends up)."""
    records = []
    dates = price_df.index[:n_records_per_agent]
    for agent_id in agent_ids:
        for i, ts in enumerate(dates):
            records.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "agent_id": agent_id,
                    "ticker": "AAPL",
                    "signal": "bullish",
                    "confidence": 70.0,
                    "price_at_signal": float(price_df["close"].iloc[i]),
                }
            )
    return records


def test_full_pipeline_produces_weights(tmp_path):
    """label_signals → compute_weights → load_weights round-trip produces non-empty weights."""
    price_df = _price_df()
    agents = ["buffett_agent", "dalio_agent", "wood_agent", "lynch_agent", "munger_agent"]

    signal_log = tmp_path / "signals-test.jsonl"
    _write_signal_log(signal_log, _build_signal_records(price_df, agents))

    labeled_path = label_signals(str(signal_log), {"AAPL": price_df})

    weights_path = tmp_path / "weights.json"
    report_path = tmp_path / "accuracy_report.json"

    weights = compute_weights(
        labeled_path,
        horizon=5,
        window_days=60,
        weights_path=str(weights_path),
        report_path=str(report_path),
    )

    assert weights, "weights must be non-empty after scoring"
    assert set(weights.keys()) == set(agents)

    for agent_id, w in weights.items():
        assert w > 0, f"weight for {agent_id} should be positive"

    # Normalised: mean weight ≈ 1.0
    mean_w = sum(weights.values()) / len(weights)
    assert abs(mean_w - 1.0) < 1e-6, f"mean weight should be 1.0, got {mean_w}"

    # Weights file was written
    assert weights_path.exists()
    assert json.loads(weights_path.read_text()) == {k: pytest.approx(v, rel=1e-4) for k, v in weights.items()}

    # Report file was written
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert set(report.keys()) == set(agents)
    for entry in report.values():
        assert "hit_rate" in entry
        assert "samples" in entry
        assert "weight" in entry


def test_load_weights_reads_written_file(tmp_path):
    """load_weights reads the JSON produced by compute_weights."""
    price_df = _price_df()
    agents = ["a_agent", "b_agent", "c_agent", "d_agent", "e_agent"]

    signal_log = tmp_path / "signals.jsonl"
    _write_signal_log(signal_log, _build_signal_records(price_df, agents))

    labeled_path = label_signals(str(signal_log), {"AAPL": price_df})
    weights_path = tmp_path / "weights.json"
    compute_weights(labeled_path, horizon=5, weights_path=str(weights_path), report_path=str(tmp_path / "report.json"))

    loaded = load_weights(str(weights_path))
    assert set(loaded.keys()) == set(agents)
    for v in loaded.values():
        assert isinstance(v, float)


def test_load_weights_returns_empty_when_missing():
    """load_weights returns {} when the file doesn't exist."""
    result = load_weights("/nonexistent/path/weights.json")
    assert result == {}


def test_pipeline_with_insufficient_data_falls_back_to_prior(tmp_path):
    """Agents with < min_samples records get a neutral weight (prior = 0.5 hit rate)."""
    price_df = _price_df()
    # Only 3 records per agent — below the default min_samples=5
    records = _build_signal_records(price_df, ["sparse_agent"], n_records_per_agent=3)

    signal_log = tmp_path / "signals-sparse.jsonl"
    _write_signal_log(signal_log, records)

    labeled_path = label_signals(str(signal_log), {"AAPL": price_df})

    weights_path = tmp_path / "weights.json"
    weights = compute_weights(labeled_path, horizon=5, weights_path=str(weights_path), report_path=str(tmp_path / "report.json"))

    # sparse_agent should get weight=1.0 (neutral prior: 0.5 / mean=0.5)
    assert "sparse_agent" in weights
    assert abs(weights["sparse_agent"] - 1.0) < 1e-6
