from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def label_signals(
    signal_log_path: str,
    price_data: dict[str, pd.DataFrame],
    output_path: str | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> str:
    """Label each signal in signal_log_path with forward returns.

    Args:
        signal_log_path: Path to JSONL produced by SignalLogger.
        price_data: {ticker: DataFrame with DatetimeIndex and 'close' column}.
        output_path: If None, writes to same dir as input with prefix 'labeled_'.
        horizons: N-day forward return horizons to compute.

    Returns:
        Path to the written labeled JSONL file.
    """
    in_path = Path(signal_log_path)
    if output_path is None:
        output_path = str(in_path.parent / f"labeled_{in_path.name}")

    with open(signal_log_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            ticker = record["ticker"]
            date_str = record["date"]
            price_at_signal = record.get("price_at_signal")

            df = price_data.get(ticker)
            if df is None or df.empty or price_at_signal is None or price_at_signal == 0:
                for h in horizons:
                    record[f"return_{h}d"] = None
                f_out.write(json.dumps(record) + "\n")
                continue

            signal_ts = pd.Timestamp(date_str)
            future_closes = df[df.index > signal_ts]["close"]

            for h in horizons:
                if len(future_closes) >= h:
                    future_price = float(future_closes.iloc[h - 1])
                    record[f"return_{h}d"] = (future_price - price_at_signal) / price_at_signal
                else:
                    record[f"return_{h}d"] = None

            f_out.write(json.dumps(record) + "\n")

    return output_path
