"""Persistent command state for Telegram bot directives.

Directives are stored in a JSON file that survives process restarts (e.g. cron).
The live_trading.py entry point reads this file at startup, acts on any active
directive, and clears one-shot directives after they fire.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

Directive = Literal["only_sells", "skip_next", "skip_until_continue", "none"]

_COMMAND_PATTERNS: list[tuple[list[str], Directive]] = [
    (["only sell", "accept only sale", "sells only", "only sale"], "only_sells"),
    (["skip next", "skip tomorrow", "skip day"], "skip_next"),
    (["skip until", "pause", "stop trading"], "skip_until_continue"),
    (["continue", "resume"], "none"),  # "none" = clear any active skip_until_continue
]


def parse_directive(text: str) -> Directive | None:
    """Return the directive matched by *text*, or None if unrecognised."""
    lowered = text.lower().strip()
    for keywords, directive in _COMMAND_PATTERNS:
        if any(kw in lowered for kw in keywords):
            return directive
    return None


@dataclass
class CommandState:
    directive: Directive = "none"
    set_by_message: str = ""


class CommandStore:
    def __init__(self, path: str = "logs/command_state.json") -> None:
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load(self) -> CommandState:
        if not os.path.exists(self._path):
            return CommandState()
        try:
            with open(self._path) as f:
                data = json.load(f)
            return CommandState(**data)
        except Exception:
            logger.warning("Failed to read command state from %s — resetting.", self._path)
            return CommandState()

    def save(self, state: CommandState) -> None:
        with open(self._path, "w") as f:
            json.dump(asdict(state), f, indent=2)

    def apply(self, directive: Directive, raw_text: str) -> None:
        """Set *directive* as the active command."""
        if directive == "none":
            # "continue" / "resume" — clear any active skip_until_continue
            state = self.load()
            if state.directive == "skip_until_continue":
                state.directive = "none"
                state.set_by_message = f"cleared by: {raw_text}"
                self.save(state)
                logger.info("[commands] skip_until_continue cleared by '%s'", raw_text)
        else:
            state = CommandState(directive=directive, set_by_message=raw_text)
            self.save(state)
            logger.info("[commands] directive set to '%s' by '%s'", directive, raw_text)

    def consume_one_shot(self, state: CommandState) -> None:
        """Clear directives that apply only once (skip_next, only_sells)."""
        if state.directive in ("skip_next", "only_sells"):
            cleared = CommandState()
            self.save(cleared)
            logger.info("[commands] one-shot directive '%s' consumed and cleared", state.directive)
