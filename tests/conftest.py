from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clear_disk_cache(monkeypatch):
    """Remove on-disk cache files and disable LLM cache before each test."""
    targets = [
        Path(".cache/api_cache.pkl"),
        Path(".cache/api_cache.pkl.tmp"),
        Path(".cache/api_cache.db"),
    ]
    for f in targets:
        if f.exists():
            f.unlink()

    import src.utils.llm as llm_mod

    monkeypatch.setattr(llm_mod._llm_cache, "_enabled", False)
    yield
