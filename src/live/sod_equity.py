from datetime import date
import json
import logging
import os

logger = logging.getLogger(__name__)


def _sod_path(log_dir: str) -> str:
    return os.path.join(log_dir, f"sod-equity-{date.today()}.json")


def load_sod_equity(log_dir: str = "logs") -> float | None:
    path = _sod_path(log_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return float(data["equity"])


def save_sod_equity(equity: float, log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = _sod_path(log_dir)
    with open(path, "w") as f:
        json.dump({"equity": equity, "date": str(date.today())}, f)
    logger.info("[sod_equity] Saved SOD equity=%.2f to %s", equity, path)
