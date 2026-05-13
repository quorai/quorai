from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = "src/feedback/weights.json"


def load_weights(path: str = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
    """Load per-agent conviction weights from JSON. Returns {} if file missing."""
    p = Path(path)
    if not p.exists():
        logger.debug("No weights file found at %s; using uniform weights", path)
        return {}
    with open(p) as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}
