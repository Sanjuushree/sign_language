"""
ISL Connect — Phase 1 Test Suite.

Verifies:
1. Python module imports for all packages and files.
2. FastAPI application startup and /health endpoint response.
3. Database initialization and idempotency.
4. Basic ORM operations using SQLite + SQLAlchemy.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, engine, SessionLocal, Base
from backend.models import SystemEvent
from backend.schemas import HealthResponse


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialise database tables for testing."""
    init_db()
    yield


def test_imports():
    """Verify that all core modules can be imported without error."""
    import backend.main
    import backend.database
    import backend.models
    import backend.schemas
    import frontend.api_client
    import src.vision
    import src.preprocessing
    import src.recognition

    assert backend.main.app is not None
    assert backend.database.engine is not None
    assert backend.models.SystemEvent is not None
    assert backend.schemas.HealthResponse is not None


def test_health_endpoint():
    """Test the GET /health endpoint of the FastAPI backend."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ISL Connect API"
    assert data["phase"] == 1
    assert "Sign recognition is not yet implemented" in data["message"]

    # Validate against Pydantic schema
    validated = HealthResponse(**data)
    assert validated.status == "ok"


def test_database_idempotency():
    """Verify that calling init_db multiple times does not raise errors."""
    try:
        init_db()
        init_db()
    except Exception as e:
        pytest.fail(f"init_db() failed on repeated calls: {e}")


def test_database_model_operations():
    """Verify that we can write and read from the SQLite database using SQLAlchemy."""
    db = SessionLocal()
    try:
        # Create a test event
        test_event = SystemEvent(
            event_type="TEST_EVENT",
            message="Phase 1 verification event",
            is_error=False,
        )
        db.add(test_event)
        db.commit()
        db.refresh(test_event)

        assert test_event.id is not None
        assert test_event.event_type == "TEST_EVENT"

        # Query back
        queried = db.query(SystemEvent).filter_by(id=test_event.id).first()
        assert queried is not None
        assert queried.message == "Phase 1 verification event"

        # Clean up test row
        db.delete(queried)
        db.commit()
    finally:
        db.close()
