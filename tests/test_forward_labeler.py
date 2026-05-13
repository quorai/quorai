from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.feedback.labeler import label_signals


def _make_price_df(start: str, periods: int, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="B")
    closes = [base + i for i in range(periods)]
    return pd.DataFrame({"close": closes}, index=idx)


@pytest.fixture
def price_data():
    return {"AAPL": _make_price_df("2024-01-01", 30)}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_forward_returns_populated(price_data, tmp_path):
    df = price_data["AAPL"]
    day5 = df.index[5]
    day10 = df.index[10]

    signals_path = tmp_path / "signals.jsonl"
    _write_jsonl(
        signals_path,
        [
            {"date": str(day5.date()), "ticker": "AAPL", "agent_id": "a1", "signal": "bullish", "confidence": 0.8, "price_at_signal": float(df["close"].iloc[5])},
            {"date": str(day10.date()), "ticker": "AAPL", "agent_id": "a1", "signal": "bearish", "confidence": 0.6, "price_at_signal": float(df["close"].iloc[10])},
        ],
    )

    out_path = label_signals(str(signals_path), price_data)
    records = [json.loads(line) for line in Path(out_path).read_text().splitlines()]

    assert records[0]["return_1d"] is not None
    assert records[0]["return_5d"] is not None
    assert records[1]["return_1d"] is not None
    assert records[1]["return_5d"] is not None


def test_return_20d_none_near_end(price_data, tmp_path):
    df = price_data["AAPL"]
    day25 = df.index[25]

    signals_path = tmp_path / "signals.jsonl"
    _write_jsonl(
        signals_path,
        [{"date": str(day25.date()), "ticker": "AAPL", "agent_id": "a1", "signal": "neutral", "confidence": 0.5, "price_at_signal": float(df["close"].iloc[25])}],
    )

    out_path = label_signals(str(signals_path), price_data)
    record = json.loads(Path(out_path).read_text().splitlines()[0])

    assert record["return_20d"] is None


def test_return_arithmetic(price_data, tmp_path):
    df = price_data["AAPL"]
    day0 = df.index[0]
    price_at_signal = float(df["close"].iloc[0])

    signals_path = tmp_path / "signals.jsonl"
    _write_jsonl(
        signals_path,
        [{"date": str(day0.date()), "ticker": "AAPL", "agent_id": "a1", "signal": "bullish", "confidence": 0.9, "price_at_signal": price_at_signal}],
    )

    out_path = label_signals(str(signals_path), price_data)
    record = json.loads(Path(out_path).read_text().splitlines()[0])

    expected_1d = (float(df["close"].iloc[1]) - price_at_signal) / price_at_signal
    assert record["return_1d"] == pytest.approx(expected_1d)

    expected_5d = (float(df["close"].iloc[5]) - price_at_signal) / price_at_signal
    assert record["return_5d"] == pytest.approx(expected_5d)


def test_missing_ticker_returns_none(tmp_path):
    signals_path = tmp_path / "signals.jsonl"
    _write_jsonl(
        signals_path,
        [{"date": "2024-01-05", "ticker": "MSFT", "agent_id": "a1", "signal": "bullish", "confidence": 0.7, "price_at_signal": 300.0}],
    )

    out_path = label_signals(str(signals_path), {})
    record = json.loads(Path(out_path).read_text().splitlines()[0])

    assert record["return_1d"] is None
    assert record["return_5d"] is None
    assert record["return_20d"] is None


def test_default_output_path(price_data, tmp_path):
    df = price_data["AAPL"]
    signals_path = tmp_path / "signals.jsonl"
    _write_jsonl(
        signals_path,
        [{"date": str(df.index[0].date()), "ticker": "AAPL", "agent_id": "a1", "signal": "bullish", "confidence": 0.5, "price_at_signal": float(df["close"].iloc[0])}],
    )

    out_path = label_signals(str(signals_path), price_data)
    assert Path(out_path).name == "labeled_signals.jsonl"
    assert Path(out_path).exists()
