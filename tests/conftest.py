from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clear_disk_cache():
    """Remove on-disk cache files before each test to prevent cross-test pollution."""
    targets = [
        Path(".cache/api_cache.pkl"),
        Path(".cache/api_cache.pkl.tmp"),
        Path(".cache/api_cache.db"),
    ]
    for f in targets:
        if f.exists():
            f.unlink()
    yield
