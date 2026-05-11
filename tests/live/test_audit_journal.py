from datetime import date
import json
import os
import tempfile

from src.live.audit_journal import AuditJournal


def test_record_creates_jsonl(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    journal.record(ticker="AAPL", action="buy", qty=1.5, side="buy", status="submitted", order_id="abc123")

    log_file = tmp_path / f"trades-{date.today()}.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text().strip())
    assert entry["ticker"] == "AAPL"
    assert entry["action"] == "buy"
    assert entry["qty"] == 1.5
    assert entry["status"] == "submitted"
    assert entry["order_id"] == "abc123"


def test_record_appends_multiple_lines(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    journal.record(ticker="AAPL", action="buy", qty=1.0, side="buy", status="submitted")
    journal.record(ticker="MSFT", action="sell", qty=2.0, side="sell", status="rejected", reason="kill_switch_active")

    log_file = tmp_path / f"trades-{date.today()}.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["ticker"] == "MSFT"
    assert second["reason"] == "kill_switch_active"


def test_record_custom_timestamp(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    journal.record(ticker="NVDA", action="short", qty=10.0, side="sell", status="rejected", timestamp="2026-01-01T00:00:00+00:00")

    log_file = tmp_path / f"trades-{date.today()}.jsonl"
    entry = json.loads(log_file.read_text().strip())
    assert entry["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_log_dir_created_automatically():
    with tempfile.TemporaryDirectory() as base:
        log_dir = os.path.join(base, "nested", "logs")
        journal = AuditJournal(log_dir=log_dir)
        journal.record(ticker="X", action="buy", qty=1.0, side="buy", status="submitted")
        assert os.path.isdir(log_dir)


def test_list_submitted_today_empty_when_no_file(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    assert journal.list_submitted_today() == []


def test_list_submitted_today_returns_only_submitted(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    journal.record(ticker="AAPL", action="buy", qty=1.0, side="buy", status="submitted")
    journal.record(ticker="MSFT", action="sell", qty=2.0, side="sell", status="rejected")
    journal.record(ticker="NVDA", action="short", qty=3.0, side="sell", status="error")

    entries = journal.list_submitted_today()
    assert len(entries) == 1
    assert entries[0]["ticker"] == "AAPL"
    assert entries[0]["status"] == "submitted"


def test_list_submitted_today_skips_malformed_lines(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    journal.record(ticker="AAPL", action="buy", qty=1.0, side="buy", status="submitted")
    journal.record(ticker="MSFT", action="buy", qty=2.0, side="buy", status="submitted")

    log_file = tmp_path / f"trades-{date.today()}.jsonl"
    lines = log_file.read_text().strip().split("\n")
    lines.insert(1, "{not-valid-json")
    log_file.write_text("\n".join(lines) + "\n")

    entries = journal.list_submitted_today()
    assert len(entries) == 2


def test_has_submitted_today(tmp_path):
    journal = AuditJournal(log_dir=str(tmp_path))
    assert not journal.has_submitted_today()
    journal.record(ticker="AAPL", action="buy", qty=1.0, side="buy", status="submitted")
    assert journal.has_submitted_today()
    journal.record(ticker="MSFT", action="sell", qty=2.0, side="sell", status="rejected")
    assert journal.has_submitted_today()
