"""Tests that Telegram approval messages use no parse_mode (avoids ticker injection)."""

from unittest.mock import MagicMock

from src.notifications.telegram import TelegramClient


def _make_client() -> TelegramClient:
    return TelegramClient(token="test-token", chat_id="12345")


def _stub_post(client: TelegramClient) -> list[dict]:
    captured: list[dict] = []
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"result": {"message_id": 42}}

    def fake_post(url, json=None, **kwargs):
        captured.append(json or {})
        return mock_response

    client._client.post = fake_post
    return captured


def test_send_approval_request_has_no_parse_mode():
    """Markdown parse_mode must be absent so tickers like BRK.B or RDS-A don't inject."""
    client = _make_client()
    payloads = _stub_post(client)

    client.send_approval_request("Proposed Orders:\nAAPL | buy | 5 | $150.00")

    assert len(payloads) == 1
    assert "parse_mode" not in payloads[0]


def test_format_decisions_table_preserves_special_tickers():
    """Special-char tickers (BRK.B, RDS-A) must appear verbatim in the table."""
    decision = MagicMock()
    decision.action = "buy"
    decision.quantity = 2
    client = _make_client()
    table = client.format_decisions_table(
        {"BRK.B": decision, "RDS-A": decision},
        {"BRK.B": 100.0, "RDS-A": 30.0},
    )
    assert "BRK.B" in table
    assert "RDS-A" in table


def test_format_decisions_table_header_is_plain_text():
    """Table header must not use Markdown bold (*) that Telegram would parse."""
    decision = MagicMock()
    decision.action = "buy"
    decision.quantity = 1
    client = _make_client()
    table = client.format_decisions_table({"AAPL": decision}, {"AAPL": 150.0})
    first_line = table.split("\n")[0]
    assert "*" not in first_line
