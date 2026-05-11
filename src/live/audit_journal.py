from datetime import date
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)


class AuditJournal:
    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = log_dir
        self._lock = threading.Lock()
        os.makedirs(log_dir, exist_ok=True)

    def _log_path(self) -> str:
        return os.path.join(self._log_dir, f"trades-{date.today()}.jsonl")

    def record(
        self,
        *,
        ticker: str,
        action: str,
        qty: float,
        side: str,
        status: str,
        reason: str = "",
        order_id: str = "",
        timestamp: str | None = None,
    ) -> None:
        from datetime import datetime, timezone

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": ts,
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "side": side,
            "status": status,
            "reason": reason,
            "order_id": order_id,
        }
        with self._lock:
            with open(self._log_path(), "a") as f:
                f.write(json.dumps(entry) + "\n")
        logger.debug("[journal] %s", entry)

    def list_submitted_today(self) -> list[dict]:
        """Return today's journal entries with status == 'submitted'. Returns [] if no file."""
        path = self._log_path()
        if not os.path.exists(path):
            return []
        entries = []
        with self._lock:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("[journal] skipping malformed line: %.80s", line)
                        continue
                    if entry.get("status") == "submitted":
                        entries.append(entry)
        return entries

    def has_submitted_today(self) -> bool:
        """True if list_submitted_today() is non-empty."""
        return bool(self.list_submitted_today())
