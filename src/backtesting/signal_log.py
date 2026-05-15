from __future__ import annotations

import json
from pathlib import Path

from src.backtesting.types import AgentSignals


class SignalLogger:
    """Appends per-agent per-ticker signal records to a JSONL file."""

    def __init__(self, run_id: str, log_dir: str = "logs") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(log_dir) / f"signals-{run_id}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: WPS515

    @property
    def path(self) -> str:
        return str(self._path)

    def log_day(
        self,
        date_str: str,
        analyst_signals: AgentSignals,
        signal_prices: dict[str, float],
    ) -> None:
        for agent_id, ticker_map in analyst_signals.items():
            for ticker, sig_data in ticker_map.items():
                record = {
                    "date": date_str,
                    "agent_id": agent_id,
                    "ticker": ticker,
                    "signal": sig_data.get("signal", "neutral"),
                    "confidence": float(sig_data.get("confidence") or 0.0),
                    "price_at_signal": signal_prices.get(ticker),
                }
                self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
