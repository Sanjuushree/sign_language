"""
ISL Connect — Pydantic Schemas (Phases 1–6).

Data transfer objects (DTOs) for request validation and response serialisation.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Health Response Schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Schema returned by the /health endpoint."""
    status: str
    service: str
    version: str
    phase: int
    message: str

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# System Event Schemas
# ---------------------------------------------------------------------------

class SystemEventBase(BaseModel):
    event_type: str
    message: str
    is_error: bool = False


class SystemEventCreate(SystemEventBase):
    pass


class SystemEventResponse(SystemEventBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Recognition & Gesture Prediction Schemas (Phase 5)
# ---------------------------------------------------------------------------

class SignPredictionResponse(BaseModel):
    """Schema returned by sign recognition predict endpoints."""
    predicted_sign: str
    confidence: float
    is_confident: bool
    top_k: list[list[Any]] = []
    hands_detected: bool
    latency_ms: float
    message: str
    error: Optional[str] = None


class SupportedSignsResponse(BaseModel):
    """Schema returned by the supported-signs endpoint."""
    model_loaded: bool
    total_classes: int
    classes: list[str]
    accuracy: Optional[float] = None
    model_type: Optional[str] = None
    message: str


class LandmarkPayload(BaseModel):
    """Schema for landmark-based inference."""
    landmarks: list[list[dict[str, float]]]


# ---------------------------------------------------------------------------
# Continuous Sentence Translation Schemas (Phase 6)
# ---------------------------------------------------------------------------

class ContinuousTranslationResponse(BaseModel):
    """Schema returned by /continuous/translate endpoints."""
    gloss_sequence: list[str]
    english_translation: str
    intent: str
    confidence: float
    matched: bool
    source: str


class GlossSessionResponse(BaseModel):
    """Schema for the current session state in /continuous/session endpoints."""
    gloss_sequence: list[str]
    translation: Optional[ContinuousTranslationResponse] = None
    buffer_length: int
