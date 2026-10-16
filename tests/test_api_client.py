"""
ISL Connect — Phase 2 API Client Tests.

Tests:
1. Successful health check (GET /health returning 200 OK).
2. Connection error handling (FastAPI offline).
3. Timeout handling.
4. HTTP error handling (non-200 responses).
5. Invalid/non-JSON response handling.
6. Base URL configuration and environment variable overrides (API_BASE_URL).
7. Live integration test against the running FastAPI instance.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import APIClient, check_backend_health, get_default_base_url


# ---------------------------------------------------------------------------
# Unit tests with Mocked HTTP requests
# ---------------------------------------------------------------------------

def test_check_health_success():
    """Verify check_health handles a successful 200 response with JSON payload."""
    client = APIClient(base_url="http://mock-backend:8000")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "service": "ISL Connect API",
        "version": "0.1.0",
        "phase": 1,
        "message": "API is running.  Sign recognition is not yet implemented.",
    }

    with patch("requests.get", return_value=mock_response) as mock_get:
        result = client.check_health()

        mock_get.assert_called_once_with(
            "http://mock-backend:8000/health",
            headers={"Accept": "application/json"},
            timeout=3.0,
        )
        assert result["connected"] is True
        assert result["data"]["status"] == "ok"
        assert result["data"]["service"] == "ISL Connect API"
        assert result["error"] is None
        assert result["status_code"] == 200


def test_check_health_connection_error():
    """Verify check_health handles backend offline gracefully without crashing."""
    client = APIClient(base_url="http://non-existent-backend:9999")

    with patch("requests.get", side_effect=ConnectionError("Failed to establish a new connection")):
        result = client.check_health()

        assert result["connected"] is False
        assert result["data"] is None
        assert "Could not connect to backend" in result["error"]
        assert result["status_code"] is None


def test_check_health_timeout():
    """Verify check_health handles request timeouts gracefully without crashing."""
    client = APIClient(base_url="http://slow-backend:8000", timeout=1.0)

    with patch("requests.get", side_effect=Timeout("Request timed out")):
        result = client.check_health()

        assert result["connected"] is False
        assert result["data"] is None
        assert "timed out" in result["error"].lower()
        assert result["status_code"] is None


def test_check_health_http_error():
    """Verify check_health handles HTTP 500 / non-200 responses gracefully."""
    client = APIClient(base_url="http://mock-backend:8000")

    mock_response = MagicMock()
    mock_response.status_code = 503

    with patch("requests.get", return_value=mock_response):
        result = client.check_health()

        assert result["connected"] is False
        assert result["data"] is None
        assert "HTTP 503" in result["error"]
        assert result["status_code"] == 503


def test_check_health_non_json_response():
    """Verify check_health handles a 200 response that contains non-JSON text."""
    client = APIClient(base_url="http://mock-backend:8000")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")

    with patch("requests.get", return_value=mock_response):
        result = client.check_health()

        assert result["connected"] is False
        assert result["data"] is None
        assert "non-JSON" in result["error"]


def test_api_client_custom_base_url():
    """Verify custom base_url strips trailing slashes properly."""
    client = APIClient(base_url="http://127.0.0.1:8000///")
    assert client.base_url == "http://127.0.0.1:8000"


def test_api_client_env_override(monkeypatch):
    """Verify API_BASE_URL environment variable is respected."""
    monkeypatch.setenv("API_BASE_URL", "http://env-backend:7000")
    assert get_default_base_url() == "http://env-backend:7000"

    client = APIClient()
    assert client.base_url == "http://env-backend:7000"


def test_convenience_function():
    """Verify module-level check_backend_health() works as expected."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}

    with patch("requests.get", return_value=mock_response):
        res = check_backend_health("http://test-server:5000")
        assert res["connected"] is True
        assert res["data"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Integration test against live running backend (if active)
# ---------------------------------------------------------------------------

def test_live_backend_if_running():
    """
    Integration test: checks if real backend is running at http://127.0.0.1:8000.
    If running, verifies the real /health endpoint returns expected fields.
    If not running, verifies graceful failure.
    """
    client = APIClient(base_url="http://127.0.0.1:8000", timeout=2.0)
    result = client.check_health()

    # The test must return a valid structured dictionary in all cases
    assert "connected" in result
    assert "data" in result
    assert "error" in result

    if result["connected"]:
        assert result["data"]["status"] == "ok"
        assert result["data"]["service"] == "ISL Connect API"
        assert result["error"] is None
    else:
        assert result["error"] is not None
