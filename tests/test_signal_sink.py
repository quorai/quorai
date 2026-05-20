from __future__ import annotations

import json

import pytest

from src.backtesting.signal_log import SignalLogger


def test_log_day_writes_correct_records(tmp_path):
    logger = SignalLogger("test-run", log_dir=str(tmp_path))

    analyst_signals = {
        "fundamentals_agent": {
            "AAPL": {"signal": "bullish", "confidence": 0.8},
            "MSFT": {"signal": "neutral", "confidence": 0.5},
        },
        "sentiment_agent": {
            "AAPL": {"signal": "bearish", "confidence": 0.6},
        },
    }
    signal_prices = {"AAPL": 150.0, "MSFT": 300.0}

    logger.log_day("2024-01-15", analyst_signals, signal_prices)
    logger.close()

    records = [json.loads(line) for line in (tmp_path / "signals" / "signals-test-run.jsonl").read_text().splitlines()]
    assert len(records) == 3

    by_key = {(r["agent_id"], r["ticker"]): r for r in records}
    aapl_fund = by_key[("fundamentals_agent", "AAPL")]
    assert aapl_fund["date"] == "2024-01-15"
    assert aapl_fund["signal"] == "bullish"
    assert aapl_fund["confidence"] == pytest.approx(0.8)
    assert aapl_fund["price_at_signal"] == pytest.approx(150.0)

    msft_fund = by_key[("fundamentals_agent", "MSFT")]
    assert msft_fund["signal"] == "neutral"
    assert msft_fund["price_at_signal"] == pytest.approx(300.0)

    aapl_sent = by_key[("sentiment_agent", "AAPL")]
    assert aapl_sent["signal"] == "bearish"


def test_log_day_missing_price_records_none(tmp_path):
    logger = SignalLogger("test-run2", log_dir=str(tmp_path))
    analyst_signals = {"agent": {"UNKNOWN": {"signal": "bullish", "confidence": 0.9}}}
    logger.log_day("2024-01-16", analyst_signals, {})
    logger.close()

    records = [json.loads(line) for line in (tmp_path / "signals" / "signals-test-run2.jsonl").read_text().splitlines()]
    assert records[0]["price_at_signal"] is None


def test_log_day_missing_confidence_defaults_to_zero(tmp_path):
    logger = SignalLogger("test-run3", log_dir=str(tmp_path))
    analyst_signals = {"agent": {"AAPL": {"signal": "neutral"}}}
    logger.log_day("2024-01-17", analyst_signals, {"AAPL": 100.0})
    logger.close()

    records = [json.loads(line) for line in (tmp_path / "signals" / "signals-test-run3.jsonl").read_text().splitlines()]
    assert records[0]["confidence"] == pytest.approx(0.0)


def test_close_makes_file_unwritable(tmp_path):
    logger = SignalLogger("test-run4", log_dir=str(tmp_path))
    logger.close()
    assert logger._file.closed
