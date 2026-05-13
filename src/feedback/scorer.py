from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def _is_hit(signal: str, forward_return: float | None) -> bool | None:
    """True if signal direction matches return sign. None if ambiguous."""
    if forward_return is None:
        return None
    if signal == "bullish":
        return forward_return > 0
    if signal == "bearish":
        return forward_return < 0
    return None  # neutral signals are excluded from scoring


def compute_weights(
    labeled_path: str,
    horizon: int = 5,
    window_days: int = 60,
    weights_path: str = "src/feedback/weights.json",
    report_path: str = "src/feedback/accuracy_report.json",
    min_samples: int = 5,
) -> dict[str, float]:
    """Compute per-agent conviction weights from a labeled signal log.

    Uses the most recent `window_days` trading-day records per agent.
    Agents with < min_samples valid labels get weight = 1.0.

    Returns the weights dict (also written to weights_path).
    """
    records_by_agent: dict[str, list[dict]] = defaultdict(list)
    with open(labeled_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get(f"return_{horizon}d") is not None:
                records_by_agent[rec["agent_id"]].append(rec)

    hit_rates: dict[str, float] = {}
    sample_counts: dict[str, int] = {}

    for agent_id, recs in records_by_agent.items():
        recs_sorted = sorted(recs, key=lambda r: r["date"])

        dates = sorted({r["date"] for r in recs_sorted})
        cutoff_dates = set(dates[-window_days:]) if len(dates) > window_days else set(dates)
        windowed = [r for r in recs_sorted if r["date"] in cutoff_dates]

        hits = [_is_hit(r["signal"], r[f"return_{horizon}d"]) for r in windowed]
        valid = [h for h in hits if h is not None]

        if len(valid) < min_samples:
            hit_rates[agent_id] = 0.5  # prior: 50% hit rate (neutral weight)
        else:
            hit_rates[agent_id] = sum(valid) / len(valid)
        sample_counts[agent_id] = len(valid)

    if not hit_rates:
        return {}

    raw_weights = {agent: hr for agent, hr in hit_rates.items()}
    mean_w = sum(raw_weights.values()) / len(raw_weights)
    if mean_w == 0:
        weights = {agent: 1.0 for agent in raw_weights}
    else:
        weights = {agent: w / mean_w for agent, w in raw_weights.items()}

    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)

    report = {
        agent: {
            "hit_rate": round(hit_rates[agent], 4),
            "samples": sample_counts[agent],
            "weight": round(weights[agent], 4),
        }
        for agent in sorted(weights)
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return weights
