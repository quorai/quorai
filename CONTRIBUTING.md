# Contributing to Quorai

## Development setup

```bash
# Install all dependencies including dev extras
uv sync --group dev

# Run tests
uv run pytest -q

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check (enforced subset — src/agents/* is intentionally excluded)
uv run mypy src/broker src/live src/data src/backtesting src/config.py
```

## Submitting changes

1. Fork the repo and create a feature branch from `main`.
2. Make your changes; add or update tests where appropriate.
3. Ensure the full test + lint suite passes (`pytest`, `ruff check`, `ruff format --check`, `mypy`).
4. Open a pull request against `main` with a clear description of what the change does and why.

## Adding a new analyst agent

1. Create `src/agents/<name>.py` with an `<name>_agent(state, agent_id)` function that follows the pattern of existing agents.
2. Register it in `src/utils/analysts.py` by adding an entry to `ANALYST_CONFIG`.
3. Add a test in `tests/` exercising the new agent.

## Reporting issues

Use the [GitHub issue tracker](../../issues). For security vulnerabilities, open an issue and mark it as a security report.
