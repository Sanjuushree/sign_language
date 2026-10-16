"""
ISL Connect — FastAPI Backend
Phase 1: Project skeleton with health check and CORS configuration.

Start with:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import os
import sys
from pathlib import Path

# Ensure the project root directory (ISL_Connect) is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.schemas import HealthResponse


# ---------------------------------------------------------------------------
# Lifespan context manager — initialise DB cleanly on boot
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise SQLite database tables on application start-up."""
    init_db()
    yield


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ISL Connect API",
    description=(
        "Backend API for the Indian Sign Language Recognition and "
        "Translation System.\n\n"
        "**Phase 1** — Project skeleton.  Sign recognition endpoints will "
        "be added in later phases once the ML pipeline is ready."
    ),
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name": "ISL Connect Team",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# CORS — allow the local Streamlit frontend to reach this API
# ---------------------------------------------------------------------------

_cors_env = os.getenv("CORS_ORIGINS", "")
if _cors_env.strip():
    ALLOWED_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core routes (inline for Phase 1 simplicity)
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health check",
    tags=["System"],
    response_model=HealthResponse,
    response_description="API status information",
)
async def health_check() -> HealthResponse:
    """
    Returns a simple status payload confirming the API is running.

    This endpoint is intentionally lightweight and has no database dependency
    so that infrastructure health checks can succeed even if the DB is slow.
    """
    return {
        "status": "ok",
        "service": "ISL Connect API",
        "version": "0.1.0",
        "phase": 1,
        "message": "API is running.  Sign recognition is not yet implemented.",
    }


@app.get("/", summary="Root redirect info", tags=["System"])
async def root() -> dict:
    """Informational root endpoint — points to the API docs."""
    return {
        "message": "Welcome to the ISL Connect API.",
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# Modular route inclusion
# ---------------------------------------------------------------------------
from backend.routes import recognition
app.include_router(recognition.router)

from backend.routes import continuous
app.include_router(continuous.router)
