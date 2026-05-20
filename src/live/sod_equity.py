import json
import logging
import os

from src.utils.tz import now_ny, ny_date_str

logger = logging.getLogger(__name__)


def _sod_path(log_dir: str) -> str:
    return os.path.join(log_dir, "sod-equity", f"sod-equity-{ny_date_str()}.json")


def load_sod_equity(log_dir: str = "logs/live") -> float | None:
    path = _sod_path(log_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return float(data["equity"])


def save_sod_equity(equity: float, log_dir: str = "logs/live", *, allow_intraday: bool = False) -> None:
    now = now_ny()
    # Refuse intraday capture: saving equity after the open would reset the
    # loss-limit baseline to a depressed value, silently disabling DAILY_LOSS_LIMIT_PCT.
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= market_open_time and not allow_intraday:
        ny_date = now.strftime("%Y-%m-%d")
        raise RuntimeError(f'Cannot establish SOD equity intraday — manually create logs/live/sod-equity/sod-equity-{ny_date}.json with {{"equity": <value>, "date": "{ny_date}"}} before market open (09:30 ET)')
    os.makedirs(os.path.join(log_dir, "sod-equity"), exist_ok=True)
    path = _sod_path(log_dir)
    with open(path, "w") as f:
        json.dump({"equity": equity, "date": ny_date_str()}, f)
    if allow_intraday and now >= market_open_time:
        logger.warning("[sod_equity] Intraday write override: saved SOD equity=%.2f to %s", equity, path)
    else:
        logger.info("[sod_equity] Saved SOD equity=%.2f to %s", equity, path)
