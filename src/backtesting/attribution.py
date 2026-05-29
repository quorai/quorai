from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from src.utils.analysts import get_agent_to_group


def _directional_spread(
    bull_returns: list[float],
    bear_returns: list[float],
) -> float | None:
    """Mean forward-return when bullish minus mean when bearish.

    Returns None if either set is empty (signal IC cannot be computed).
    Positive → the analyst's directional calls correlate with price direction.
    """
    if not bull_returns or not bear_returns:
        return None
    return sum(bull_returns) / len(bull_returns) - sum(bear_returns) / len(bear_returns)


def _conf_weighted_mean(records: list[dict], horizon_key: str) -> float | None:
    """Confidence-weighted mean forward return, sign-adjusted for direction.

    Bullish signals contribute +confidence × return; bearish −confidence × return.
    Returns None if there are no valid records.
    """
    total_weight = 0.0
    total_weighted = 0.0
    for r in records:
        fwd = r.get(horizon_key)
        if fwd is None:
            continue
        conf = float(r.get("confidence") or 0.0)
        signal = r.get("signal", "neutral")
        if signal == "bullish":
            sign = 1.0
        elif signal == "bearish":
            sign = -1.0
        else:
            continue
        total_weighted += conf * sign * fwd
        total_weight += conf
    if total_weight == 0.0:
        return None
    return total_weighted / total_weight


def _agent_entry(
    agent_id: str,
    recs: list[dict],
    horizon_key: str,
    group: str,
    baseline_mean: float,
) -> dict[str, Any]:
    """Build a single attribution entry dict from a list of directional records."""
    bull_returns = [r[horizon_key] for r in recs if r["signal"] == "bullish"]
    bear_returns = [r[horizon_key] for r in recs if r["signal"] == "bearish"]
    all_valid = [r[horizon_key] for r in recs]

    hits = [r[horizon_key] > 0 if r["signal"] == "bullish" else r[horizon_key] < 0 for r in recs]
    hit_rate = sum(hits) / len(hits) if hits else None
    spread = _directional_spread(bull_returns, bear_returns)
    cwm = _conf_weighted_mean(recs, horizon_key)
    mean_return = sum(all_valid) / len(all_valid) if all_valid else None
    alpha = (mean_return - baseline_mean) if mean_return is not None else None

    return {
        "agent_id": agent_id,
        "group": group,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "sample_count": len(recs),
        "directional_spread": round(spread, 6) if spread is not None else None,
        "conf_weighted_score": round(cwm, 6) if cwm is not None else None,
        "mean_return": round(mean_return, 6) if mean_return is not None else None,
        "alpha_vs_baseline": round(alpha, 6) if alpha is not None else None,
        "bull_count": len(bull_returns),
        "bear_count": len(bear_returns),
    }


def _spread_sort_key(x: dict) -> tuple:
    """Sort key: None spreads last, then descending by spread value."""
    spread = x["directional_spread"]
    return (spread is None, -(spread or 0))


def compute_attribution(
    labeled_signal_log: str,
    horizon: int = 5,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """Compute per-analyst alpha-attribution metrics from a labeled signal log.

    The labeled log is produced by ``feedback/labeler.py`` (run the 'feedback'
    CLI subcommand first).  Each record must include ``return_{horizon}d``.

    Metrics per analyst:
    * ``hit_rate`` — fraction of directional signals whose direction matched the
      forward return sign.
    * ``sample_count`` — number of valid (non-neutral, non-None-return) signals.
    * ``directional_spread`` — mean forward return when bullish minus mean when
      bearish.  A crude per-analyst information coefficient: positive means the
      analyst's calls correlate with subsequent price moves.
    * ``conf_weighted_score`` — confidence-weighted mean sign-adjusted return;
      accounts for conviction as well as direction.
    * ``alpha_vs_baseline`` — mean return minus the cross-section mean (per-name
      alpha proxy).
    * ``group`` — strategy group from the analyst registry.

    Returns a list sorted by ``directional_spread`` descending (best first).
    Group-level roll-ups are appended at the end.
    Writes the full report as JSON to ``output_path`` when provided.
    """
    horizon_key = f"return_{horizon}d"
    # get_agent_to_group() returns {"<key>_agent": group_name}; strip suffix for lookup.
    _raw_group_map = get_agent_to_group()
    # Build a key-without-suffix map for convenience.
    agent_group: dict[str, str] = {k.removesuffix("_agent"): v for k, v in _raw_group_map.items()}

    records_by_agent: dict[str, list[dict]] = defaultdict(list)
    with open(labeled_signal_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get(horizon_key) is not None and rec.get("signal") in ("bullish", "bearish"):
                # Normalise agent key: strip _agent suffix consistent with scorer.py.
                agent_key = rec["agent_id"].removesuffix("_agent")
                records_by_agent[agent_key].append(rec)

    if not records_by_agent:
        return []

    # Cross-section baseline: unweighted mean forward return across all directional records.
    all_returns = [rec[horizon_key] for recs in records_by_agent.values() for rec in recs]
    baseline_mean = sum(all_returns) / len(all_returns) if all_returns else 0.0

    analyst_report: list[dict[str, Any]] = []
    for agent_id, recs in records_by_agent.items():
        group = agent_group.get(agent_id, "")
        analyst_report.append(_agent_entry(agent_id, recs, horizon_key, group, baseline_mean))

    analyst_report.sort(key=_spread_sort_key)

    # Group-level roll-up: aggregate all signals from analysts in each group.
    records_by_group: dict[str, list[dict]] = defaultdict(list)
    for agent_id, recs in records_by_agent.items():
        g = agent_group.get(agent_id, "")
        if g:
            records_by_group[g].extend(recs)

    group_report: list[dict[str, Any]] = []
    for group_name, g_recs in sorted(records_by_group.items()):
        entry = _agent_entry(f"[GROUP] {group_name}", g_recs, horizon_key, group_name, baseline_mean)
        group_report.append(entry)

    group_report.sort(key=_spread_sort_key)
    combined = analyst_report + group_report

    full_report = {
        "horizon_days": horizon,
        "baseline_mean_return": round(baseline_mean, 6),
        "analysts": combined,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)

    return combined
