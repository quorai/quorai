"""Tests for weights.json staleness warning in load_weights."""

import json
import logging
import os
import time


def test_fresh_file_no_warning(tmp_path, caplog):
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({"agent_a": 1.2}))

    import importlib

    import src.feedback.loader as loader_mod

    importlib.reload(loader_mod)

    with caplog.at_level(logging.WARNING, logger="src.feedback.loader"):
        result = loader_mod.load_weights(str(weights_file))

    assert result == {"agent_a": 1.2}
    assert not any("days old" in r.message for r in caplog.records)


def test_stale_file_emits_warning(tmp_path, caplog):
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({"agent_a": 1.2}))

    # Back-date the mtime by 40 days (above default 30-day threshold)
    stale_mtime = time.time() - 40 * 86400
    os.utime(str(weights_file), (stale_mtime, stale_mtime))

    import importlib

    import src.feedback.loader as loader_mod

    importlib.reload(loader_mod)

    with caplog.at_level(logging.WARNING, logger="src.feedback.loader"):
        result = loader_mod.load_weights(str(weights_file))

    assert result == {"agent_a": 1.2}  # still loads despite being stale
    assert any("days old" in r.message for r in caplog.records)


def test_env_override_threshold(tmp_path, caplog, monkeypatch):
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({"agent_b": 0.8}))

    # 5 days old; default threshold is 30 → no warning
    # But with threshold set to 3, it should warn
    stale_mtime = time.time() - 5 * 86400
    os.utime(str(weights_file), (stale_mtime, stale_mtime))

    monkeypatch.setenv("QUORAI_WEIGHTS_MAX_AGE_DAYS", "3")

    import importlib

    import src.feedback.loader as loader_mod

    importlib.reload(loader_mod)

    with caplog.at_level(logging.WARNING, logger="src.feedback.loader"):
        loader_mod.load_weights(str(weights_file))

    assert any("days old" in r.message for r in caplog.records)
