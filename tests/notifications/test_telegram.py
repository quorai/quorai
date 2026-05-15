from unittest.mock import MagicMock, patch

import httpx

from src.notifications.telegram import TelegramClient


def _make_client(responses: list[httpx.Response]) -> TelegramClient:
    """Build a TelegramClient whose underlying httpx.Client returns the given responses in order."""
    client = TelegramClient(token="test-token", chat_id="123")
    mock_http = MagicMock(spec=httpx.Client)
    call_iter = iter(responses)
    mock_http.get.side_effect = lambda *a, **kw: next(call_iter)
    mock_http.post.side_effect = lambda *a, **kw: next(call_iter)
    client._client = mock_http
    return client


_DUMMY_REQUEST = httpx.Request("POST", "https://api.telegram.org/")


def _ok(body: dict) -> httpx.Response:
    r = httpx.Response(200, json=body)
    r.request = _DUMMY_REQUEST
    return r


def _send_message_resp(message_id: int) -> httpx.Response:
    return _ok({"ok": True, "result": {"message_id": message_id}})


def _callback_update(update_id: int, message_id: int, callback_data: str) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cq-1",
            "data": callback_data,
            "message": {"message_id": message_id},
        },
    }


def _updates_resp(updates: list[dict]) -> httpx.Response:
    return _ok({"ok": True, "result": updates})


def _empty_updates() -> httpx.Response:
    return _updates_resp([])


def test_approve():
    responses = [
        _send_message_resp(42),  # sendMessage (POST)
        _empty_updates(),  # priming getUpdates(offset=-1) — no stale callbacks
        _updates_resp([_callback_update(1, 42, "approve")]),  # poll getUpdates
        _ok({"ok": True}),  # answerCallbackQuery
        _ok({"ok": True}),  # editMessageText
    ]
    client = _make_client(responses)
    msg_id = client.send_approval_request("Test text")
    assert msg_id == 42
    result = client.wait_for_decision(42, timeout_seconds=60)
    assert result == "approve"


def test_reject():
    responses = [
        _send_message_resp(42),
        _empty_updates(),  # priming getUpdates(offset=-1)
        _updates_resp([_callback_update(1, 42, "reject")]),
        _ok({"ok": True}),
        _ok({"ok": True}),
    ]
    client = _make_client(responses)
    client.send_approval_request("Test text")
    result = client.wait_for_decision(42, timeout_seconds=60)
    assert result == "reject"


def test_timeout():
    time_values = iter([0.0, 0.0, 100.0])

    responses = [
        _empty_updates(),  # priming getUpdates(offset=-1)
        _empty_updates(),  # poll getUpdates — returns nothing
    ]
    client = _make_client(responses)

    with patch("src.notifications.telegram.time.monotonic", side_effect=time_values):
        result = client.wait_for_decision(42, timeout_seconds=60)

    assert result == "timeout"
    # answerCallbackQuery and editMessageText must NOT have been called
    assert client._client.post.call_count == 0


def test_send_approval_request_returns_message_id():
    responses = [_send_message_resp(7)]
    client = _make_client(responses)
    msg_id = client.send_approval_request("some text")
    assert msg_id == 7


def _text_update(update_id: int, text: str) -> dict:
    return {"update_id": update_id, "message": {"text": text}}


def test_poll_text_messages_returns_texts():
    updates = [_text_update(10, "skip next day"), _text_update(11, "hello")]
    responses = [_updates_resp(updates)]
    client = _make_client(responses)
    texts = client.poll_text_messages()
    assert texts == ["skip next day", "hello"]
    assert client._next_offset == 12


def test_poll_text_messages_empty():
    responses = [_empty_updates()]
    client = _make_client(responses)
    texts = client.poll_text_messages()
    assert texts == []
    assert client._next_offset == 0


def test_poll_text_messages_skips_non_text():
    updates = [{"update_id": 5, "message": {}}, _text_update(6, "continue")]
    responses = [_updates_resp(updates)]
    client = _make_client(responses)
    texts = client.poll_text_messages()
    assert texts == ["continue"]


def test_send_message():
    responses = [_ok({"ok": True, "result": {"message_id": 1}})]
    client = _make_client(responses)
    client.send_message("test")
    assert client._client.post.call_count == 1


def test_ignores_updates_for_other_messages():
    responses = [
        _updates_resp([_callback_update(1, 99, "approve")]),  # wrong message_id
        _updates_resp([_callback_update(2, 42, "approve")]),  # correct message_id
        _ok({"ok": True}),  # answerCallbackQuery
        _ok({"ok": True}),  # editMessageText
    ]
    client = _make_client(responses)
    result = client.wait_for_decision(42, timeout_seconds=60)
    assert result == "approve"
