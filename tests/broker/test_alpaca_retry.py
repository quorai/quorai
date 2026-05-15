"""Tests for the _retry_api_call wrapper in alpaca_client."""

from unittest.mock import MagicMock

from alpaca.common.exceptions import APIError
import pytest
from requests.exceptions import ConnectionError as _ConnError


def _make_api_error(status_code: int) -> APIError:
    http_error = MagicMock()
    http_error.response.status_code = status_code
    return APIError("{}", http_error)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.broker.alpaca_client.time.sleep", lambda _: None)


def test_retry_fires_on_503():
    from src.broker.alpaca_client import _retry_api_call

    fn = MagicMock(side_effect=[_make_api_error(503), _make_api_error(503), "ok"])
    result = _retry_api_call(fn)
    assert result == "ok"
    assert fn.call_count == 3


def test_retry_fires_on_connection_error():
    from src.broker.alpaca_client import _retry_api_call

    fn = MagicMock(side_effect=[_ConnError("refused"), "ok"])
    result = _retry_api_call(fn)
    assert result == "ok"
    assert fn.call_count == 2


def test_no_retry_on_422():
    from src.broker.alpaca_client import _retry_api_call

    fn = MagicMock(side_effect=_make_api_error(422))
    with pytest.raises(APIError):
        _retry_api_call(fn)
    assert fn.call_count == 1


def test_no_retry_on_403():
    from src.broker.alpaca_client import _retry_api_call

    fn = MagicMock(side_effect=_make_api_error(403))
    with pytest.raises(APIError):
        _retry_api_call(fn)
    assert fn.call_count == 1


def test_raises_after_3_attempts():
    from src.broker.alpaca_client import _retry_api_call

    fn = MagicMock(side_effect=_make_api_error(500))
    with pytest.raises(APIError):
        _retry_api_call(fn)
    assert fn.call_count == 3


def test_passes_args_and_kwargs():
    from src.broker.alpaca_client import _retry_api_call

    fn = MagicMock(return_value="result")
    result = _retry_api_call(fn, "a", "b", key="val")
    assert result == "result"
    fn.assert_called_once_with("a", "b", key="val")
