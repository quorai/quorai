import logging
import time
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(
        self,
        token: str,
        chat_id: str,
        base_url: str = "https://api.telegram.org",
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)
        self._next_offset: int = 0

    def _url(self, method: str) -> str:
        return f"{self._base_url}/bot{self._token}/{method}"

    def format_decisions_table(self, decisions: dict, latest_prices: dict[str, float]) -> str:
        lines = ["Proposed Orders:", "", "Ticker | Action | Qty | Price", "------ | ------ | --- | -----"]
        for ticker, decision in decisions.items():
            action = getattr(decision, "action", str(decision))
            qty = getattr(decision, "quantity", "—")
            price = latest_prices.get(ticker, 0.0)
            lines.append(f"{ticker} | {action} | {qty} | ${price:.2f}")
        return "\n".join(lines)

    def send_approval_request(self, text: str) -> int:
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "Approve ✅", "callback_data": "approve"},
                    {"text": "Reject ❌", "callback_data": "reject"},
                ]
            ]
        }
        response = self._client.post(
            self._url("sendMessage"),
            json={
                "chat_id": self._chat_id,
                "text": text,
                "reply_markup": reply_markup,
            },
        )
        response.raise_for_status()
        return response.json()["result"]["message_id"]

    def send_message(self, text: str) -> None:
        response = self._client.post(
            self._url("sendMessage"),
            json={"chat_id": self._chat_id, "text": text},
        )
        response.raise_for_status()

    def poll_text_messages(self) -> list[str]:
        """Return any plain-text messages received since the last poll (non-blocking)."""
        response = self._client.get(
            self._url("getUpdates"),
            params={
                "offset": self._next_offset,
                "timeout": 0,
                "allowed_updates": ["message"],
            },
        )
        response.raise_for_status()
        updates = response.json().get("result", [])
        if updates:
            self._next_offset = max(u["update_id"] for u in updates) + 1
        return [u["message"]["text"] for u in updates if u.get("message", {}).get("text")]

    def wait_for_decision(self, message_id: int, timeout_seconds: int) -> Literal["approve", "reject", "timeout"]:
        # Prime the offset past any existing updates so stale callbacks from a
        # prior run's approval prompt are never replayed as a decision for this cycle.
        prime = self._client.get(
            self._url("getUpdates"),
            params={"offset": -1, "timeout": 0, "allowed_updates": ["callback_query"]},
            timeout=5,
        )
        prime.raise_for_status()
        prime_updates = prime.json().get("result", [])
        if prime_updates:
            self._next_offset = max(u["update_id"] for u in prime_updates) + 1
        else:
            self._next_offset = 0
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                return "timeout"

            poll_timeout = min(25, remaining)
            client_timeout = poll_timeout + 5

            response = self._client.get(
                self._url("getUpdates"),
                params={
                    "offset": self._next_offset,
                    "timeout": int(poll_timeout),
                    "allowed_updates": ["callback_query"],
                },
                timeout=client_timeout,
            )
            response.raise_for_status()
            updates = response.json().get("result", [])

            for update in updates:
                cq = update.get("callback_query")
                if cq and cq.get("message", {}).get("message_id") == message_id:
                    callback_data = cq.get("data", "")

                    self._client.post(
                        self._url("answerCallbackQuery"),
                        json={"callback_query_id": cq["id"]},
                    )

                    if callback_data == "approve":
                        new_text = "✅ Approved — executing orders."
                    else:
                        new_text = "❌ Rejected — no orders submitted."

                    try:
                        self._client.post(
                            self._url("editMessageText"),
                            json={
                                "chat_id": self._chat_id,
                                "message_id": message_id,
                                "text": new_text,
                            },
                        )
                    except Exception:
                        logger.warning("Failed to edit Telegram message %d", message_id)

                    if updates:
                        self._next_offset = max(u["update_id"] for u in updates) + 1

                    return "approve" if callback_data == "approve" else "reject"

            if updates:
                self._next_offset = max(u["update_id"] for u in updates) + 1
