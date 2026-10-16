"""
ISL Connect — Frontend API Client.

Handles communication between the Streamlit frontend and the FastAPI backend
using Python's requests library.
"""

import os
from typing import Any, Dict, Optional
import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout


# Resolve base URL: check API_BASE_URL, fallback to BACKEND_URL, default to local FastAPI
def get_default_base_url() -> str:
    """Read backend base URL from environment or return default."""
    return os.getenv(
        "API_BASE_URL",
        os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
    ).rstrip("/")


DEFAULT_BASE_URL = get_default_base_url()


class APIClient:
    """Client for communicating with the ISL Connect FastAPI backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 3.0,
    ):
        """
        Initialise APIClient.

        :param base_url: Backend root URL (e.g. 'http://127.0.0.1:8000').
                         If omitted, reads from API_BASE_URL or defaults to http://127.0.0.1:8000.
        :param timeout: Request timeout in seconds.
        """
        resolved_url = base_url if base_url is not None else get_default_base_url()
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Calls GET /health on the backend API.

        Gracefully catches connection errors, timeouts, and HTTP errors without
        raising exceptions, returning a structured result dictionary.

        :return: Dict containing:
            - "connected" (bool): True if backend responded with HTTP 200 OK.
            - "data" (Optional[Dict]): Parsed JSON payload if successful.
            - "error" (Optional[str]): Human-readable error message on failure.
            - "status_code" (Optional[int]): HTTP response code if available.
        """
        endpoint = f"{self.base_url}/health"

        try:
            response = requests.get(
                endpoint,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                try:
                    payload = response.json()
                    return {
                        "connected": True,
                        "data": payload,
                        "error": None,
                        "status_code": 200,
                    }
                except ValueError:
                    return {
                        "connected": False,
                        "data": None,
                        "error": "Backend returned non-JSON response.",
                        "status_code": 200,
                    }
            else:
                return {
                    "connected": False,
                    "data": None,
                    "error": f"Backend returned HTTP {response.status_code}",
                    "status_code": response.status_code,
                }

        except ConnectionError:
            return {
                "connected": False,
                "data": None,
                "error": (
                    f"Could not connect to backend at {self.base_url}. "
                    "Make sure the FastAPI server is running."
                ),
                "status_code": None,
            }

        except Timeout:
            return {
                "connected": False,
                "data": None,
                "error": f"Connection timed out contacting {self.base_url} (>{self.timeout}s).",
                "status_code": None,
            }

        except HTTPError as e:
            return {
                "connected": False,
                "data": None,
                "error": f"HTTP error occurred: {str(e)}",
                "status_code": getattr(e.response, "status_code", None),
            }

        except RequestException as e:
            return {
                "connected": False,
                "data": None,
                "error": f"Request failed: {str(e)}",
                "status_code": None,
            }

        except Exception as e:
            return {
                "connected": False,
                "data": None,
                "error": f"Unexpected error: {str(e)}",
                "status_code": None,
            }


    def get_supported_signs(self) -> Dict[str, Any]:
        """Calls GET /recognition/supported-signs on the backend API."""
        endpoint = f"{self.base_url}/recognition/supported-signs"
        try:
            response = requests.get(
                endpoint,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def predict_image(self, image_bytes: bytes, filename: str = "frame.jpg") -> Dict[str, Any]:
        """Calls POST /recognition/predict uploading image bytes to backend."""
        endpoint = f"{self.base_url}/recognition/predict"
        try:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            response = requests.post(
                endpoint,
                files=files,
                timeout=self.timeout + 5.0,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}


    def predict_landmarks(self, landmarks: list) -> Dict[str, Any]:
        """Calls POST /recognition/predict-landmarks with a landmarks payload."""
        endpoint = f"{self.base_url}/recognition/predict-landmarks"
        try:
            response = requests.post(
                endpoint,
                json={"landmarks": landmarks},
                headers={"Accept": "application/json"},
                timeout=self.timeout + 2.0,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Phase 6 — Continuous Sentence Translation
    # ------------------------------------------------------------------

    def translate_gloss_sequence(self, gloss_sequence: list) -> Dict[str, Any]:
        """Calls POST /continuous/translate with an explicit gloss sequence."""
        endpoint = f"{self.base_url}/continuous/translate"
        try:
            response = requests.post(
                endpoint,
                json={"gloss_sequence": gloss_sequence},
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def push_frame_prediction(self, prediction: str, confidence: float) -> Dict[str, Any]:
        """Calls POST /continuous/session/push with a single frame prediction."""
        endpoint = f"{self.base_url}/continuous/session/push"
        try:
            response = requests.post(
                endpoint,
                json={"prediction": prediction, "confidence": confidence},
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def get_session_result(self) -> Dict[str, Any]:
        """Calls GET /continuous/session/result."""
        endpoint = f"{self.base_url}/continuous/session/result"
        try:
            response = requests.get(
                endpoint,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def reset_session(self) -> Dict[str, Any]:
        """Calls POST /continuous/session/reset."""
        endpoint = f"{self.base_url}/continuous/session/reset"
        try:
            response = requests.post(
                endpoint,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def get_grammar_mappings(self) -> Dict[str, Any]:
        """Calls GET /continuous/grammar."""
        endpoint = f"{self.base_url}/continuous/grammar"
        try:
            response = requests.get(
                endpoint,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "error": None}
            return {
                "success": False,
                "data": None,
                "error": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}


# Module-level singleton instance
api_client = APIClient()


def check_backend_health(base_url: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to check backend health using default or custom client."""
    client = APIClient(base_url=base_url) if base_url else api_client
    return client.check_health()
