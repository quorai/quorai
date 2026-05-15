"""Tests for stale callback replay prevention in wait_for_decision."""

from unittest.mock import MagicMock

from src.notifications.telegram import TelegramClient


def _make_client() -> TelegramClient:
    return TelegramClient(token="test-token", chat_id="12345")


def _make_response(updates: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"result": updates}
    return resp


def test_wait_for_decision_primes_offset():
    """wait_for_decision must call getUpdates(offset=-1) before polling to skip stale callbacks."""
    client = _make_client()
    offsets_used: list[int] = []

    def fake_get(url, params=None, timeout=None):
        offset = params.get("offset", 0) if params else 0
        offsets_used.append(offset)
        if offset == -1:
            return _make_response([{"update_id": 100}])
        # Return the real approval callback on the first real poll
        return _make_response(
            [
                {
                    "update_id": 101,
                    "callback_query": {
                        "id": "cq1",
                        "message": {"message_id": 42},
                        "data": "approve",
                    },
                }
            ]
        )

    client._client.get = fake_get
    client._client.post = MagicMock(return_value=_make_response([]))

    result = client.wait_for_decision(message_id=42, timeout_seconds=30)

    assert result == "approve"
    assert -1 in offsets_used, "Must call getUpdates with offset=-1 to prime past existing updates"


def test_offset_advances_past_priming_updates():
    """After priming with update_id=100, the poll loop must use offset >= 101."""
    client = _make_client()
    poll_offsets: list[int] = []

    def fake_get(url, params=None, timeout=None):
        offset = params.get("offset", 0) if params else 0
        if offset == -1:
            return _make_response([{"update_id": 100}])
        poll_offsets.append(offset)
        # Return real approval on first poll
        return _make_response(
            [
                {
                    "update_id": 101,
                    "callback_query": {
                        "id": "cq2",
                        "message": {"message_id": 7},
                        "data": "reject",
                    },
                }
            ]
        )

    client._client.get = fake_get
    client._client.post = MagicMock(return_value=_make_response([]))

    client.wait_for_decision(message_id=7, timeout_seconds=30)

    # All actual poll calls must use offset >= 101 (past the primed stale update)
    assert all(o >= 101 for o in poll_offsets), f"Poll offsets should be >= 101, got {poll_offsets}"


def test_timeout_when_no_decision(monkeypatch):
    """wait_for_decision returns 'timeout' when no callback arrives within timeout."""
    client = _make_client()

    def fake_get(url, params=None, timeout=None):
        offset = params.get("offset", 0) if params else 0
        if offset == -1:
            return _make_response([])
        return _make_response([])

    client._client.get = fake_get
    client._client.post = MagicMock(return_value=_make_response([]))

    result = client.wait_for_decision(message_id=99, timeout_seconds=1)
    assert result == "timeout"
