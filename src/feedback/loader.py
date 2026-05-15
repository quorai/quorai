from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = "src/feedback/weights.json"
_WEIGHTS_MAX_AGE_DAYS = int(os.environ.get("QUORAI_WEIGHTS_MAX_AGE_DAYS", "30"))


def load_weights(path: str = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
    """Load per-agent conviction weights from JSON. Returns {} if file missing."""
    p = Path(path)
    if not p.exists():
        logger.debug("No weights file found at %s; using uniform weights", path)
        return {}
    age_days = (time.time() - p.stat().st_mtime) / 86400
    if age_days > _WEIGHTS_MAX_AGE_DAYS:
        logger.warning(
            "weights.json is %.0f days old (threshold: %d) — run a backtest to refresh conviction weights",
            age_days,
            _WEIGHTS_MAX_AGE_DAYS,
        )
    with open(p) as f:
        data = json.load(f)
    return {str(k).removesuffix("_agent"): float(v) for k, v in data.items()}
