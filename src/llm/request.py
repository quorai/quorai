from __future__ import annotations

from dataclasses import dataclass, field
import json
import os


@dataclass
class RunRequest:
    """Carries per-run overrides (API keys, per-agent model routing) through the agent graph.

    Place an instance in ``state["metadata"]["request"]`` so ``get_agent_model_config``
    in ``src/utils/llm.py`` can pick up agent-specific model/provider pairs.

    Agent name matching order:
    1. Exact key match (e.g. ``"warren_buffett_agent"``).
    2. Wildcard ``"*"`` used as catch-all fallback.
    3. ``(None, None)`` — falls back to the global ``model_name``/``model_provider``.

    CLI format for ``--agent-model`` flag:
        ``AGENT=model_slug:PROVIDER``  or  ``AGENT=model_slug``  (provider defaults to "OpenRouter")
        Model slugs may contain ``/`` (e.g. ``deepseek/deepseek-chat-v3.1``); use ``:`` to separate provider.

    Env var ``QUORAI_AGENT_MODELS_JSON`` (parsed at construction time):
        ``{"warren_buffett_agent": ["model_slug", "OpenRouter"], "*": ["cheap/model", "OpenRouter"]}``
    """

    api_keys: dict | None = None
    agent_models: dict[str, tuple[str, str]] = field(default_factory=dict)

    def get_agent_model_config(self, agent_name: str) -> tuple[str | None, str | None]:
        """Return ``(model_name, provider)`` for this agent, or ``(None, None)`` to use global defaults."""
        if not self.agent_models:
            return None, None
        entry = self.agent_models.get(agent_name) or self.agent_models.get("*")
        if entry:
            return entry[0], entry[1]
        return None, None

    @classmethod
    def from_agent_model_args(
        cls,
        agent_model_args: list[str] | None = None,
        api_keys: dict | None = None,
    ) -> "RunRequest":
        """Build a RunRequest from a list of ``AGENT=model[/PROVIDER]`` strings.

        Also reads ``QUORAI_AGENT_MODELS_JSON`` env var if set (CLI args take precedence).
        """
        agent_models: dict[str, tuple[str, str]] = {}

        # 1. Load env var baseline
        env_json = os.environ.get("QUORAI_AGENT_MODELS_JSON", "")
        if env_json:
            try:
                parsed = json.loads(env_json)
                for agent, spec in parsed.items():
                    if isinstance(spec, (list, tuple)) and len(spec) == 2:
                        agent_models[agent] = (str(spec[0]), str(spec[1]))
                    elif isinstance(spec, str):
                        model, provider = _parse_model_spec(spec)
                        agent_models[agent] = (model, provider)
            except json.JSONDecodeError:
                pass

        # 2. CLI args override env var
        for arg in agent_model_args or []:
            if "=" not in arg:
                continue
            agent, spec = arg.split("=", 1)
            agent = agent.strip()
            model, provider = _parse_model_spec(spec.strip())
            agent_models[agent] = (model, provider)

        return cls(api_keys=api_keys, agent_models=agent_models)


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse ``model_slug:PROVIDER`` or ``model_slug`` (defaults provider to OpenRouter).

    Model slugs may contain '/' (e.g. ``deepseek/deepseek-chat-v3.1``) so ':' is the separator.
    """
    if ":" in spec:
        model, provider = spec.rsplit(":", 1)
        return model.strip(), provider.strip()
    return spec.strip(), "OpenRouter"
