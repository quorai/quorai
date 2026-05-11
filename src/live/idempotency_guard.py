from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from src.live.audit_journal import AuditJournal
    from src.notifications.telegram import TelegramClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriorSubmissions:
    entries: list[dict] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entries)


class PriorRunApprover(Protocol):
    def approve(self, prior: PriorSubmissions) -> Literal["approve", "reject", "timeout"]: ...


class DenyByDefaultApprover:
    """Used when Telegram is not configured: always deny."""

    def approve(self, prior: PriorSubmissions) -> Literal["approve", "reject", "timeout"]:
        return "reject"


class TelegramPriorRunApprover:
    def __init__(self, telegram_client: TelegramClient, timeout_seconds: int) -> None:
        self._tg = telegram_client
        self._timeout = timeout_seconds

    def approve(self, prior: PriorSubmissions) -> Literal["approve", "reject", "timeout"]:
        text = format_prior_submissions(prior)
        try:
            msg_id = self._tg.send_approval_request(text)
            return self._tg.wait_for_decision(msg_id, timeout_seconds=self._timeout)
        except Exception as exc:
            logger.error("[idempotency] Telegram error during re-prompt: %s", exc)
            return "reject"


class IdempotencyGuard:
    def __init__(self, journal: AuditJournal, *, approver: PriorRunApprover) -> None:
        self._journal = journal
        self._approver = approver

    def check(self) -> tuple[bool, str]:
        """Return (allowed, reason).

        'no_prior_run'       — no prior submissions today, proceed.
        'override_approved'  — prior run detected but operator approved re-run.
        'denied:<sub>'       — blocked; sub is reject | timeout | telegram_error.
        """
        entries = self._journal.list_submitted_today()
        if not entries:
            return True, "no_prior_run"

        prior = PriorSubmissions(entries=entries)
        logger.warning(
            "[idempotency] %d prior submission(s) found today — sending Telegram re-prompt.",
            prior.count,
        )
        decision = self._approver.approve(prior)
        if decision == "approve":
            logger.info("[idempotency] override approved — proceeding with second run.")
            return True, "override_approved"
        logger.warning("[idempotency] run blocked: %s", decision)
        return False, f"denied:{decision}"


def format_prior_submissions(prior: PriorSubmissions) -> str:
    """Build a Telegram-ready Markdown summary of today's already-submitted orders."""
    lines = [
        f"⚠️ *Prior run detected today* — {prior.count} order(s) already submitted. Approve to run again?",
        "",
        "Time (UTC) | Ticker | Side | Qty",
        "---------- | ------ | ---- | ---",
    ]
    for e in prior.entries:
        ts = str(e.get("timestamp", ""))[:19].replace("T", " ")
        lines.append(f"{ts} | {e.get('ticker', '?')} | {e.get('side', '?')} | {e.get('qty', '?')}")
    return "\n".join(lines)
