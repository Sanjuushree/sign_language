"""
ISL Connect â€” Frontend AppTest Suite.

Automated verification of Streamlit frontend navigation and components
using Streamlit's official AppTest API.
"""

from pathlib import Path
import sys
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def app():
    """Initialise and run the Streamlit app test instance."""
    at = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py"))
    at.run(timeout=10)
    return at


def test_app_loads_without_exceptions(app):
    """Verify that frontend/app.py loads cleanly without uncaught exceptions."""
    assert not app.exception, f"App threw unexpected exceptions: {app.exception}"


def test_app_navigation_options(app):
    """Verify that all 7 navigation radio options exist in the sidebar."""
    nav_radio = app.radio[0]
    expected_sections = [
        "Home",
        "Live Recognition",
        "Learn ISL (A\u2013Z)",
        "Dataset & Preprocessing",
        "Model Evaluation",
        "Sentence Translation",
        "About",
    ]
    assert nav_radio.options == expected_sections
    assert nav_radio.value == "Home"


def test_home_section_content(app):
    """Verify the Home section displays the welcome header, subtitle, and features."""
    assert app.radio[0].value == "Home"
    assert not app.exception

    subheaders = [s.value for s in app.subheader]
    assert any("Welcome to ISL Connect" in s for s in subheaders)


def test_home_backend_connected_display():
    """Verify Home page displays backend status metrics when backend is connected."""
    mock_health = {
        "connected": True,
        "data": {
            "status": "ok",
            "service": "ISL Connect API",
            "version": "0.1.0",
            "phase": 1,
            "message": "API is running.  Sign recognition is not yet implemented.",
        },
        "error": None,
        "status_code": 200,
    }

    with patch("frontend.api_client.APIClient.check_health", return_value=mock_health):
        at = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py"))
        at.run()
        assert not at.exception

        # Check success banner
        success_messages = [s.value for s in at.success]
        assert any("Backend Connected & Operational" in m for m in success_messages)

        # Check metrics displayed
        labels = [m.label for m in at.metric]
        assert "Service" in labels
        assert "API Status" in labels
        assert "API Version" in labels


def test_home_backend_offline_display():
    """Verify Home page displays helpful message when backend is offline."""
    mock_health = {
        "connected": False,
        "data": None,
        "error": "Could not connect to backend at http://127.0.0.1:8000.",
        "status_code": None,
    }

    with patch("frontend.api_client.APIClient.check_health", return_value=mock_health):
        at = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py"))
        at.run()
        assert not at.exception

        # Check error banner
        error_messages = [e.value for e in at.error]
        assert any("Backend Disconnected" in m for m in error_messages)

        # Check helpful instructions banner
        warnings = [w.value for w in at.warning]
        assert any("How to Start the Backend" in w for w in warnings)


def test_live_recognition_section(app):
    """Verify Live Recognition section provides camera input component and landmark tracking info."""
    app.radio[0].set_value("Live Recognition").run(timeout=15)
    assert not app.exception
    assert app.radio[0].value == "Live Recognition"

    # Verify subheader
    subheaders = [s.value for s in app.subheader]
    assert any("Live Hand Landmark Detection (Webcam)" in s for s in subheaders)

    # Verify native camera input component exists
    assert len(app.camera_input) > 0

    # Verify camera ready banner is displayed
    info_boxes = [info.value for info in app.info]
    assert any("Camera Ready" in text for text in info_boxes)


def test_learn_isl_section(app):
    """Verify Learn ISL (Aâ€“Z) section renders without error and displays verified vocabulary."""
    app.radio[0].set_value("Learn ISL (A\u2013Z)").run()
    assert not app.exception
    assert app.radio[0].value == "Learn ISL (A\u2013Z)"

    subheaders = [s.value for s in app.subheader]
    assert any("Supported ISL Vocabulary" in s for s in subheaders)


def test_learn_isl_unloaded_disclaimer():
    """Verify Learn ISL displays unverified disclaimer if no model is loaded."""
    from unittest.mock import MagicMock
    mock_clf = MagicMock()
    mock_clf.is_loaded = False
    with patch("src.recognition.classifier.get_classifier", return_value=mock_clf):
        at = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py"))
        at.run(timeout=10)
        at.radio[0].set_value("Learn ISL (A\u2013Z)").run(timeout=10)
        assert not at.exception
        warnings = [w.value for w in at.warning]
        assert any("No Signs Currently Verified" in text for text in warnings)


def test_model_evaluation_section(app):
    """Verify Model Evaluation section renders without error and displays diagnostics."""
    app.radio[0].set_value("Model Evaluation").run()
    assert not app.exception
    assert app.radio[0].value == "Model Evaluation"

    subheaders = [s.value for s in app.subheader]
    assert any("Model Evaluation & Diagnostics" in s for s in subheaders)


def test_sentence_translation_section(app):
    """Verify Sentence Translation section renders without error and displays continuous translation header."""
    app.radio[0].set_value("Sentence Translation").run()
    assert not app.exception
    assert app.radio[0].value == "Sentence Translation"

    subheaders = [s.value for s in app.subheader]
    assert any("Continuous Sentence Translation" in s for s in subheaders)


def test_about_section(app):
    """Verify About section displays purpose, tech stack, and limitations."""
    app.radio[0].set_value("About").run()
    assert not app.exception
    assert app.radio[0].value == "About"

    subheaders = [s.value for s in app.subheader]
    assert any("About ISL Connect" in s for s in subheaders)



