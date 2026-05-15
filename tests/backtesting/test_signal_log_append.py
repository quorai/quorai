"""Tests that SignalLogger appends rather than truncates on re-open."""

from src.backtesting.signal_log import SignalLogger


def test_second_open_appends_not_truncates(tmp_path):
    run_id = "test-2024-01-01"
    signals = {"agent_a": {"AAPL": {"signal": "bullish", "confidence": 0.8}}}
    prices = {"AAPL": 150.0}

    logger1 = SignalLogger(run_id=run_id, log_dir=str(tmp_path))
    logger1.log_day("2024-01-01", signals, prices)
    logger1.close()

    logger2 = SignalLogger(run_id=run_id, log_dir=str(tmp_path))
    logger2.log_day("2024-01-02", signals, prices)
    logger2.close()

    lines = [ln for ln in (tmp_path / f"signals-{run_id}.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 2, f"Expected 2 records (one per cycle), got {len(lines)}"
    import json

    assert json.loads(lines[0])["date"] == "2024-01-01"
    assert json.loads(lines[1])["date"] == "2024-01-02"
