"""Tests for R28: LangGraph messages reducer — agents must return [message], not state['messages']+[message]."""

import inspect


def _get_return_messages_patterns(module) -> list[str]:
    """Extract all return-statement lines containing 'messages' from a module's source."""
    source = inspect.getsource(module)
    lines = [line.strip() for line in source.splitlines() if '"messages"' in line and "return" in line]
    return lines


class TestRiskManagerReducer:
    def test_does_not_return_state_messages_plus_message(self):
        """risk_manager node must return {'messages': [message]}, not state['messages'] + [message]."""
        from src.agents import risk_manager as mod

        source = inspect.getsource(mod)
        assert 'state["messages"] +' not in source, "risk_manager returns state['messages'] + [message]. This causes O(N²) growth via LangGraph's operator.add reducer. Return [message] only."


class TestTechnicalsReducer:
    def test_does_not_return_state_messages_plus_message(self):
        """technicals_agent node must return {'messages': [message]}, not state['messages'] + [message]."""
        from src.agents import technicals as mod

        source = inspect.getsource(mod)
        # Exclude the technicals_agent function's return statement only (sub-functions don't build messages)
        agent_source = source[source.find("def technicals_agent") : source.find("def calculate_trend_signals")]
        assert 'state["messages"] +' not in agent_source, "technicals_agent returns state['messages'] + [message]. Return [message] only."


class TestPortfolioManagerReducer:
    def test_does_not_return_state_messages_plus_message(self):
        """portfolio_manager node must return {'messages': [message]}, not state['messages'] + [message]."""
        from src.agents import portfolio_manager as mod

        source = inspect.getsource(mod)
        agent_source = source[source.find("def portfolio_management_agent") : source.find("def compute_allowed_actions")]
        assert 'state["messages"] +' not in agent_source, "portfolio_manager returns state['messages'] + [message]. Return [message] only."
