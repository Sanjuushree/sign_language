"""
ISL Connect — Continuous Sentence Translation Route Handler (Phase 6).

REST endpoints for:
1. Translating an explicit gloss sequence to English (POST /continuous/translate).
2. Batch-translating multiple gloss sequences (POST /continuous/translate-batch).
3. Streaming session: push single-frame predictions and retrieve current sentence
   (POST /continuous/session/push, GET /continuous/session/result).
4. Resetting the current translation session (POST /continuous/session/reset).
5. Retrieving all available grammar translation mappings (GET /continuous/grammar).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.recognition.sequence_decoder import (
    ContinuousTranslator,
    SentenceDecoder,
    TranslationResult,
    get_translator,
)

router = APIRouter(prefix="/continuous", tags=["Continuous Translation"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class GlossSequenceRequest(BaseModel):
    """Request body for a translate call."""

    gloss_sequence: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of ISL gloss labels (e.g. ['YOU', 'NAME', 'WHAT']).",
        examples=[["YOU", "NAME", "WHAT"]],
    )


class GlossBatchRequest(BaseModel):
    """Request body for batch translation."""

    sequences: List[List[str]] = Field(
        ...,
        min_length=1,
        description="List of gloss sequences to translate.",
    )


class FramePredictionRequest(BaseModel):
    """Single-frame prediction to push into the session buffer."""

    prediction: str = Field(
        ...,
        description="Predicted sign label for this frame (e.g. 'HELLO', 'None').",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence in [0, 1].",
    )


class TranslationResponse(BaseModel):
    """Translation result response."""

    gloss_sequence: List[str]
    english_translation: str
    intent: str
    confidence: float
    matched: bool
    source: str


class SessionStatusResponse(BaseModel):
    """Current session state response."""

    gloss_sequence: List[str]
    translation: Optional[TranslationResponse]
    buffer_length: int


class GrammarMappingResponse(BaseModel):
    """One grammar mapping entry."""

    gloss_sequence: List[str]
    english_translation: str
    intent: str


class GrammarListResponse(BaseModel):
    """All available grammar mappings."""

    total_mappings: int
    mappings: List[GrammarMappingResponse]


def _translation_to_response(result: TranslationResult) -> TranslationResponse:
    return TranslationResponse(
        gloss_sequence=result.gloss_sequence,
        english_translation=result.english_translation,
        intent=result.intent,
        confidence=result.confidence,
        matched=result.matched,
        source=result.source,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate a gloss sequence to English",
)
async def translate_gloss_sequence(body: GlossSequenceRequest) -> TranslationResponse:
    """
    Translate an ordered list of ISL gloss labels to a natural English sentence.

    Uses the sentence_grammar.json rule-based lookup with longest-suffix
    fallback when an exact match is not found.
    """
    translator = get_translator()
    result = translator.translate_sequence(body.gloss_sequence)
    return _translation_to_response(result)


@router.post(
    "/translate-batch",
    response_model=List[TranslationResponse],
    summary="Batch-translate multiple gloss sequences",
)
async def translate_batch(body: GlossBatchRequest) -> List[TranslationResponse]:
    """Translate multiple gloss sequences in a single call."""
    if len(body.sequences) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 50 sequences per request.",
        )
    translator = get_translator()
    return [
        _translation_to_response(translator.translate_sequence(seq))
        for seq in body.sequences
    ]


@router.post(
    "/session/push",
    response_model=SessionStatusResponse,
    summary="Push a per-frame sign prediction into the session buffer",
)
async def push_frame_prediction(body: FramePredictionRequest) -> SessionStatusResponse:
    """
    Submit one frame prediction to the continuous session buffer.

    The server-side GlossBuffer accumulates predictions and emits a stable
    gloss event when a sign is held for the required number of consecutive
    frames.  Returns the current session state after processing.
    """
    translator = get_translator()
    translator.push(body.prediction, body.confidence)

    gloss_seq = translator.gloss_sequence
    translation: Optional[TranslationResponse] = None
    if gloss_seq:
        result = translator.translate_current()
        translation = _translation_to_response(result)

    return SessionStatusResponse(
        gloss_sequence=gloss_seq,
        translation=translation,
        buffer_length=len(translator.buffer._window),
    )


@router.get(
    "/session/result",
    response_model=SessionStatusResponse,
    summary="Get current session gloss sequence and translation",
)
async def get_session_result() -> SessionStatusResponse:
    """Return the current gloss sequence and English translation without modifying state."""
    translator = get_translator()
    gloss_seq = translator.gloss_sequence
    translation: Optional[TranslationResponse] = None
    if gloss_seq:
        result = translator.translate_current()
        translation = _translation_to_response(result)

    return SessionStatusResponse(
        gloss_sequence=gloss_seq,
        translation=translation,
        buffer_length=len(translator.buffer._window),
    )


@router.post(
    "/session/reset",
    summary="Reset the continuous translation session",
)
async def reset_session() -> Dict[str, str]:
    """Clear the server-side gloss buffer and start a fresh translation session."""
    translator = get_translator()
    translator.reset()
    return {"status": "ok", "message": "Continuous translation session has been reset."}


@router.get(
    "/grammar",
    response_model=GrammarListResponse,
    summary="List all available grammar translation mappings",
)
async def get_grammar_mappings() -> GrammarListResponse:
    """Return all registered gloss-sequence → English translation rules."""
    decoder = SentenceDecoder()
    mappings = [
        GrammarMappingResponse(
            gloss_sequence=m.get("gloss_sequence", []),
            english_translation=m.get("english_translation", ""),
            intent=m.get("intent", "unknown"),
        )
        for m in decoder._mappings
    ]
    return GrammarListResponse(
        total_mappings=len(mappings),
        mappings=mappings,
    )
