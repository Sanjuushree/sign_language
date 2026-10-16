"""
ISL Connect — Sign Recognition Route Handler (Phase 5).

Exposes REST endpoints for:
1. Predicting ISL alphabet signs from uploaded image files.
2. Predicting ISL alphabet signs from MediaPipe landmark arrays.
3. Querying verified supported signs, model metadata, and accuracy statistics.
"""

from typing import Any, List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.schemas import (
    LandmarkPayload,
    SignPredictionResponse,
    SupportedSignsResponse,
)
from src.recognition.classifier import get_classifier

router = APIRouter(prefix="/recognition", tags=["Recognition"])


@router.get(
    "/supported-signs",
    response_model=SupportedSignsResponse,
    summary="Get verified ISL signs and model metadata",
)
async def get_supported_signs() -> SupportedSignsResponse:
    """Return list of verified signs, classification accuracy, and model metadata."""
    classifier = get_classifier()
    meta = classifier.metadata or {}

    total_classes = len(classifier.classes) if classifier.classes else 0
    accuracy = meta.get("test_accuracy")
    model_type = meta.get("model_type", "RandomForestClassifier")

    if classifier.is_loaded:
        msg = f"Model active with {total_classes} verified ISL letters (A–Z)."
    else:
        msg = "No trained recognition model currently loaded."

    return SupportedSignsResponse(
        model_loaded=classifier.is_loaded,
        total_classes=total_classes,
        classes=classifier.classes,
        accuracy=accuracy,
        model_type=model_type,
        message=msg,
    )


@router.post(
    "/predict",
    response_model=SignPredictionResponse,
    summary="Predict sign from image frame",
)
async def predict_sign_image(
    file: UploadFile = File(..., description="Camera frame image file (JPEG, PNG)")
) -> SignPredictionResponse:
    """Process an image frame, track hand landmarks, and classify the ISL alphabet sign."""
    if not file.content_type or not file.content_type.startswith("image/"):
        # Still allow files with generic octet-stream if extension matches
        filename = (file.filename or "").lower()
        if not any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File provided is not a supported image format (JPEG, PNG, WEBP).",
            )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image bytes: {e}",
        )

    classifier = get_classifier()
    prediction, _ = classifier.predict_image(contents)

    return SignPredictionResponse(
        predicted_sign=prediction.predicted_sign,
        confidence=prediction.confidence,
        is_confident=prediction.is_confident,
        top_k=[[k, v] for k, v in prediction.top_k],
        hands_detected=prediction.hands_detected,
        latency_ms=prediction.latency_ms,
        message=prediction.message,
        error=prediction.error,
    )


@router.post(
    "/predict-landmarks",
    response_model=SignPredictionResponse,
    summary="Predict sign from 3D hand landmark coordinates",
)
async def predict_sign_landmarks(
    payload: LandmarkPayload,
) -> SignPredictionResponse:
    """Classify an ISL sign from raw MediaPipe hand landmark coordinates."""
    classifier = get_classifier()
    prediction = classifier.predict_landmarks(payload.landmarks)

    return SignPredictionResponse(
        predicted_sign=prediction.predicted_sign,
        confidence=prediction.confidence,
        is_confident=prediction.is_confident,
        top_k=[[k, v] for k, v in prediction.top_k],
        hands_detected=prediction.hands_detected,
        latency_ms=prediction.latency_ms,
        message=prediction.message,
        error=prediction.error,
    )
