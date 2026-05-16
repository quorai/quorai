"""Tests that token log is isolated per context (ContextVar scoping)."""

import threading

import src.utils.llm as llm_module
from src.utils.llm import get_token_log, get_token_summary, reset_token_log


def _append_entry(label: str) -> dict:
    return {"agent": label, "model": "m", "input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0, "cache_creation_tokens": 0}


def test_reset_isolates_second_call_from_first():
    """Calling reset_token_log again starts a fresh log in the same context."""
    reset_token_log()
    llm_module._run_token_log.get().append(_append_entry("run1"))
    assert len(get_token_log()) == 1

    reset_token_log()
    assert get_token_log() == [], "second reset should produce an empty log"


def test_two_threads_have_independent_logs():
    """Two threads each calling reset_token_log accumulate separate token logs."""
    results: dict[str, list] = {}
    errors: list[Exception] = []

    def _run(name: str) -> None:
        try:
            reset_token_log()
            for i in range(3):
                llm_module._run_token_log.get().append(_append_entry(f"{name}-{i}"))
            results[name] = get_token_log()
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=("threadA",))
    t2 = threading.Thread(target=_run, args=("threadB",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, errors
    assert len(results["threadA"]) == 3
    assert len(results["threadB"]) == 3
    agents_a = {e["agent"] for e in results["threadA"]}
    agents_b = {e["agent"] for e in results["threadB"]}
    assert not agents_a & agents_b, "threads must not see each other's entries"


def test_token_summary_aggregates_current_context_only():
    reset_token_log()
    llm_module._run_token_log.get().extend(
        [
            _append_entry("a"),
            _append_entry("b"),
        ]
    )
    summary = get_token_summary()
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 20
    assert summary["output_tokens"] == 10
