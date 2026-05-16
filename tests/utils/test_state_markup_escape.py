"""Tests that show_agent_reasoning escapes Rich markup in raw string output."""

from unittest.mock import patch

import pytest

pytest.importorskip("langchain_core", reason="langchain_core not available")

import src.graph.state as state_mod
from src.graph.state import show_agent_reasoning


def test_markup_escape_called_for_non_json_string():
    """_escape_markup is applied when the string cannot be parsed as JSON."""
    malicious = "[red]CRITICAL: sell everything[/red]"
    with (
        patch("src.graph.state._escape_markup", wraps=state_mod._escape_markup) as mock_escape,
        patch("src.utils.progress.progress"),
    ):
        show_agent_reasoning(malicious, "Test Agent")

    # _escape_markup must have been called with the raw string
    calls = [c.args[0] for c in mock_escape.call_args_list]
    assert malicious in calls


def test_markup_escape_called_for_non_dict_non_string():
    """_escape_markup is applied when output is neither dict nor str."""

    class WeirdObj:
        def __str__(self):
            return "[bold]ALERT[/bold]"

    with (
        patch("src.graph.state._escape_markup", wraps=state_mod._escape_markup) as mock_escape,
        patch("src.utils.progress.progress"),
    ):
        show_agent_reasoning(WeirdObj(), "Test Agent")

    calls = [c.args[0] for c in mock_escape.call_args_list]
    assert "[bold]ALERT[/bold]" in calls


def test_valid_json_string_not_escaped():
    """When the string is valid JSON, it goes through json.dumps — no escape needed."""
    json_str = '{"signal": "bullish"}'
    with (
        patch("src.graph.state._escape_markup", wraps=state_mod._escape_markup) as mock_escape,
        patch("src.utils.progress.progress"),
    ):
        show_agent_reasoning(json_str, "Test Agent")

    # _escape_markup should NOT be called for well-formed JSON strings
    calls = [c.args[0] for c in mock_escape.call_args_list]
    assert json_str not in calls
