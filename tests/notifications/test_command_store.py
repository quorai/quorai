import pytest

from src.notifications.command_store import CommandStore, parse_directive

# --- parse_directive ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("accept only sales", "only_sells"),
        ("Accept Only Sells", "only_sells"),
        ("only sells", "only_sells"),
        ("skip next day", "skip_next"),
        ("Skip Next", "skip_next"),
        ("skip tomorrow", "skip_next"),
        ("skip until continue", "skip_until_continue"),
        ("pause", "skip_until_continue"),
        ("stop trading", "skip_until_continue"),
        ("continue", "none"),
        ("Resume", "none"),
        ("hello world", None),
        ("", None),
    ],
)
def test_parse_directive(text, expected):
    assert parse_directive(text) == expected


# --- CommandStore ---


def test_load_returns_default_when_file_missing(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    state = store.load()
    assert state.directive == "none"


def test_apply_and_load(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    store.apply("skip_next", "skip next day")
    state = store.load()
    assert state.directive == "skip_next"
    assert "skip next day" in state.set_by_message


def test_apply_continue_clears_skip_until_continue(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    store.apply("skip_until_continue", "pause")
    store.apply("none", "continue")
    state = store.load()
    assert state.directive == "none"


def test_apply_continue_is_noop_when_no_active_pause(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    store.apply("none", "continue")
    state = store.load()
    assert state.directive == "none"


def test_consume_one_shot_clears_skip_next(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    store.apply("skip_next", "skip next")
    active = store.load()
    store.consume_one_shot(active)
    assert store.load().directive == "none"


def test_consume_one_shot_clears_only_sells(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    store.apply("only_sells", "accept only sales")
    active = store.load()
    store.consume_one_shot(active)
    assert store.load().directive == "none"


def test_consume_one_shot_does_not_clear_skip_until_continue(tmp_path):
    store = CommandStore(path=str(tmp_path / "state.json"))
    store.apply("skip_until_continue", "pause")
    active = store.load()
    store.consume_one_shot(active)
    assert store.load().directive == "skip_until_continue"


def test_load_handles_corrupt_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json")
    store = CommandStore(path=str(path))
    state = store.load()
    assert state.directive == "none"
