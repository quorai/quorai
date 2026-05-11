from unittest.mock import MagicMock

from src.live.audit_journal import AuditJournal
from src.live.idempotency_guard import (
    DenyByDefaultApprover,
    IdempotencyGuard,
    PriorSubmissions,
    TelegramPriorRunApprover,
    format_prior_submissions,
)


def _journal_with_submission(tmp_path) -> AuditJournal:
    journal = AuditJournal(log_dir=str(tmp_path))
    journal.record(ticker="AAPL", action="buy", qty=5.0, side="buy", status="submitted")
    return journal


def test_check_allows_when_no_prior_submissions(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    approver = MagicMock()
    guard = IdempotencyGuard(journal=journal, approver=approver)

    allowed, reason = guard.check()

    assert allowed is True
    assert reason == "no_prior_run"
    approver.approve.assert_not_called()


def test_check_denies_when_approver_rejects(tmp_path):
    journal = _journal_with_submission(tmp_path)
    approver = MagicMock()
    approver.approve.return_value = "reject"
    guard = IdempotencyGuard(journal=journal, approver=approver)

    allowed, reason = guard.check()

    assert allowed is False
    assert "denied" in reason
    assert "reject" in reason


def test_check_denies_when_approver_times_out(tmp_path):
    journal = _journal_with_submission(tmp_path)
    approver = MagicMock()
    approver.approve.return_value = "timeout"
    guard = IdempotencyGuard(journal=journal, approver=approver)

    allowed, reason = guard.check()

    assert allowed is False
    assert "timeout" in reason


def test_check_allows_when_approver_approves(tmp_path):
    journal = _journal_with_submission(tmp_path)
    approver = MagicMock()
    approver.approve.return_value = "approve"
    guard = IdempotencyGuard(journal=journal, approver=approver)

    allowed, reason = guard.check()

    assert allowed is True
    assert reason == "override_approved"


def test_deny_by_default_approver_always_rejects():
    approver = DenyByDefaultApprover()
    prior = PriorSubmissions(entries=[{"ticker": "AAPL", "qty": 1.0}])
    assert approver.approve(prior) == "reject"


def test_format_prior_submissions_contains_fields():
    entries = [
        {"timestamp": "2026-04-20T15:48:22+00:00", "ticker": "AAPL", "side": "buy", "qty": 71.0},
        {"timestamp": "2026-04-20T15:48:22+00:00", "ticker": "MSFT", "side": "sell", "qty": 3.5},
    ]
    prior = PriorSubmissions(entries=entries)
    text = format_prior_submissions(prior)

    assert "2" in text  # count
    assert "AAPL" in text
    assert "MSFT" in text
    assert "buy" in text
    assert "71.0" in text


def test_telegram_approver_sends_formatted_text_and_relays_decision(tmp_path):
    journal = _journal_with_submission(tmp_path)
    prior = PriorSubmissions(entries=journal.list_submitted_today())

    tg = MagicMock()
    tg.send_approval_request.return_value = 42
    tg.wait_for_decision.return_value = "approve"

    approver = TelegramPriorRunApprover(telegram_client=tg, timeout_seconds=60)
    result = approver.approve(prior)

    assert result == "approve"
    tg.send_approval_request.assert_called_once()
    sent_text = tg.send_approval_request.call_args[0][0]
    assert "AAPL" in sent_text
    tg.wait_for_decision.assert_called_once_with(42, timeout_seconds=60)
