from datetime import datetime
import json
import logging
import os
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")


def _ny_date_str() -> str:
    return datetime.now(_NY).strftime("%Y-%m-%d")


def _sod_path(log_dir: str) -> str:
    return os.path.join(log_dir, f"sod-equity-{_ny_date_str()}.json")


def load_sod_equity(log_dir: str = "logs") -> float | None:
    path = _sod_path(log_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return float(data["equity"])


def save_sod_equity(equity: float, log_dir: str = "logs") -> None:
    now = datetime.now(_NY)
    # Refuse intraday capture: saving equity after the open would reset the
    # loss-limit baseline to a depressed value, silently disabling DAILY_LOSS_LIMIT_PCT.
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= market_open_time:
        ny_date = now.strftime("%Y-%m-%d")
        raise RuntimeError(f'Cannot establish SOD equity intraday — manually create logs/sod-equity-{ny_date}.json with {{"equity": <value>, "date": "{ny_date}"}} before market open (09:30 ET)')
    os.makedirs(log_dir, exist_ok=True)
    path = _sod_path(log_dir)
    with open(path, "w") as f:
        json.dump({"equity": equity, "date": _ny_date_str()}, f)
    logger.info("[sod_equity] Saved SOD equity=%.2f to %s", equity, path)
