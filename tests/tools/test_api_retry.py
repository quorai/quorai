"""Tests for _make_api_request retry logic: exponential backoff, RateLimitExhaustedError, 5xx retry."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tools.api import RateLimitExhaustedError, _make_api_request


def _mock_response(status_code: int) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    return r


class TestRateLimitExhaustedError:
    def test_attributes(self):
        err = RateLimitExhaustedError("https://example.com", 4)
        assert err.endpoint == "https://example.com"
        assert err.attempts == 4
        assert "4" in str(err)


class TestSuccessOnFirstAttempt:
    @patch("src.tools.api.requests.get")
    def test_returns_immediately(self, mock_get):
        mock_get.return_value = _mock_response(200)
        resp = _make_api_request("https://example.com/ok")
        assert resp.status_code == 200
        assert mock_get.call_count == 1

    @patch("src.tools.api.requests.get")
    def test_timeout_kwarg_passed(self, mock_get):
        mock_get.return_value = _mock_response(200)
        _make_api_request("https://example.com/ok", params={"k": "v"})
        mock_get.assert_called_once_with("https://example.com/ok", params={"k": "v"}, headers=None, timeout=(5, 30))


class TestRateLimitRetry:
    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_retries_on_429_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [_mock_response(429), _mock_response(200)]
        resp = _make_api_request("https://example.com/api", max_retries=3)
        assert resp.status_code == 200
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_raises_after_all_retries_exhausted(self, mock_get, mock_sleep):
        mock_get.return_value = _mock_response(429)
        with pytest.raises(RateLimitExhaustedError) as exc_info:
            _make_api_request("https://example.com/api", max_retries=2)
        assert exc_info.value.attempts == 3  # 1 initial + 2 retries
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.tools.api.random.uniform", return_value=0.5)
    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_exponential_backoff_delays(self, mock_get, mock_sleep, mock_rand):
        mock_get.side_effect = [_mock_response(429), _mock_response(429), _mock_response(200)]
        _make_api_request("https://example.com/api", max_retries=3)
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        # attempt 0: 2**0 + 0.5 = 1.5, attempt 1: 2**1 + 0.5 = 2.5
        assert delays == pytest.approx([1.5, 2.5])


class TestServerErrorRetry:
    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_retries_on_5xx_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [_mock_response(503), _mock_response(200)]
        resp = _make_api_request("https://example.com/api", max_retries=3)
        assert resp.status_code == 200
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_returns_last_5xx_after_exhaustion(self, mock_get, mock_sleep):
        mock_get.return_value = _mock_response(500)
        resp = _make_api_request("https://example.com/api", max_retries=2)
        assert resp.status_code == 500
        assert mock_get.call_count == 3


class TestNetworkErrorRetry:
    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_retries_on_timeout_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [requests.exceptions.Timeout(), _mock_response(200)]
        resp = _make_api_request("https://example.com/api", max_retries=3)
        assert resp.status_code == 200
        assert mock_get.call_count == 2

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_raises_timeout_after_exhaustion(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.exceptions.Timeout()
        with pytest.raises(requests.exceptions.Timeout):
            _make_api_request("https://example.com/api", max_retries=2)
        assert mock_get.call_count == 3

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        mock_get.side_effect = [requests.exceptions.ConnectionError(), _mock_response(200)]
        resp = _make_api_request("https://example.com/api", max_retries=3)
        assert resp.status_code == 200
        assert mock_get.call_count == 2
