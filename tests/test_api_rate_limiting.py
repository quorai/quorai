"""Legacy tests for _make_api_request — kept for backward-compatibility coverage.

The comprehensive suite is in tests/tools/test_api_retry.py which covers the
current (exponential-backoff, RateLimitExhaustedError) interface.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tools.api import RateLimitExhaustedError, _make_api_request


def _resp(status: int) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    return r


class TestRateLimiting:
    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_handles_single_rate_limit(self, mock_get, mock_sleep):
        """Retries once on 429 then succeeds."""
        mock_get.side_effect = [_resp(429), _resp(200)]
        result = _make_api_request("https://example.com/api")
        assert result.status_code == 200
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_handles_multiple_rate_limits(self, mock_get, mock_sleep):
        """Retries multiple times on 429 then succeeds."""
        mock_get.side_effect = [_resp(429), _resp(429), _resp(429), _resp(200)]
        result = _make_api_request("https://example.com/api")
        assert result.status_code == 200
        assert mock_get.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_retries_server_errors(self, mock_get, mock_sleep):
        """500 responses are retried with backoff."""
        mock_get.side_effect = [_resp(500), _resp(200)]
        result = _make_api_request("https://example.com/api")
        assert result.status_code == 200
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_normal_success_requests(self, mock_get, mock_sleep):
        """Successful requests return immediately without retry."""
        mock_get.return_value = _resp(200)
        result = _make_api_request("https://example.com/api")
        assert result.status_code == 200
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_max_retries_exceeded_raises(self, mock_get, mock_sleep):
        """Raises RateLimitExhaustedError when all 429 retries exhausted."""
        mock_get.return_value = _resp(429)
        with pytest.raises(RateLimitExhaustedError):
            _make_api_request("https://example.com/api", max_retries=2)
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
