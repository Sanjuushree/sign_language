"""
ISL Connect — Continuous Sentence Decoder (Phase 6).

Converts a stream of per-frame isolated sign predictions (from ISLSignClassifier)
into stable gloss sequences and then to natural English sentences.

Architecture
------------
* GlossBuffer          — Sliding temporal window that debounces and stabilises
                         high-frequency frame predictions into gloss events.
* SentenceDecoder      — Rule-based lookup mapping ISL gloss sequences to
                         natural English translations using sentence_grammar.json.
* ContinuousTranslator — Top-level orchestrator combining buffer + decoder for
                         use in real-time streaming and batch modes.

No sequence model weights are required; the rule-based decoder operates from
the pre-built sentence_grammar.json vocabulary alone.  A learned sequence
model (HMM / TCN) can be plugged in later by subclassing or replacing
GlossBuffer.predict().
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GRAMMAR_PATH = (
    _PROJECT_ROOT / "dataset" / "continuous" / "metadata" / "sentence_grammar.json"
)
DEFAULT_VOCAB_PATH = (
    _PROJECT_ROOT / "dataset" / "continuous" / "metadata" / "vocabulary.json"
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GlossEvent:
    """A single stable gloss prediction extracted from a temporal window."""

    gloss: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TranslationResult:
    """Output of the full continuous translation pipeline."""

    gloss_sequence: List[str]
    english_translation: str
    intent: str
    confidence: float
    matched: bool
    source: str  # "grammar_lookup" | "fallback_join"


# ---------------------------------------------------------------------------
# GlossBuffer — sliding temporal window for per-frame predictions
# ---------------------------------------------------------------------------


class GlossBuffer:
    """
    Converts a high-frequency stream of per-frame sign predictions into a
    stable sequence of gloss events by:

    1. Accumulating predictions into a rolling window of *window_size* frames.
    2. Selecting the majority-vote prediction within the window.
    3. Emitting a new gloss event only when the dominant prediction is:
       - Stable for at least *min_stable_frames* consecutive frames, AND
       - Different from the last emitted gloss OR more than *repetition_gap_s*
         seconds have elapsed (allowing intentional repeated signs).

    Parameters
    ----------
    window_size : int
        Number of recent per-frame predictions kept in the rolling window
        (default 15 frames at 30 fps ≈ 0.5 s window).
    min_stable_frames : int
        Minimum number of consecutive frames with the same dominant prediction
        before a gloss event is emitted (default 10).
    confidence_threshold : float
        Minimum per-frame confidence required for a prediction to be counted
        (default 0.50).
    repetition_gap_s : float
        Minimum seconds between emitting the same gloss twice in a row,
        to support intentional repeated signs (default 1.5 s).
    max_glosses : int
        Maximum gloss events kept in the emitted sequence before oldest are
        dropped (default 8).
    """

    def __init__(
        self,
        window_size: int = 15,
        min_stable_frames: int = 10,
        confidence_threshold: float = 0.50,
        repetition_gap_s: float = 1.5,
        max_glosses: int = 8,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if min_stable_frames < 1 or min_stable_frames > window_size:
            raise ValueError(
                f"min_stable_frames must be in [1, window_size], "
                f"got {min_stable_frames} (window_size={window_size})."
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
            )

        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        self.confidence_threshold = confidence_threshold
        self.repetition_gap_s = repetition_gap_s
        self.max_glosses = max_glosses

        # Rolling window of (prediction_label, confidence) tuples
        self._window: Deque[Tuple[str, float]] = deque(maxlen=window_size)

        # Emitted gloss sequence
        self._gloss_sequence: List[GlossEvent] = []

        # State tracking
        self._last_emitted_gloss: Optional[str] = None
        self._last_emitted_time: float = 0.0
        self._consecutive_count: int = 0
        self._last_dominant: Optional[str] = None
        self._streak_broken: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, prediction: str, confidence: float) -> Optional[GlossEvent]:
        """
        Submit one per-frame prediction to the buffer.

        Parameters
        ----------
        prediction : str
            The predicted sign label for this frame (e.g. "HELLO", "None", "?").
        confidence : float
            Classifier confidence in [0, 1].

        Returns
        -------
        GlossEvent | None
            A new GlossEvent if the window stabilised on a new gloss, else None.
        """
        # Reject uncertain or non-sign predictions
        if confidence < self.confidence_threshold or prediction in ("None", "?", ""):
            self._window.clear()
            self._consecutive_count = 0
            self._last_dominant = None
            self._streak_broken = True
            return None

        self._window.append((prediction, confidence))

        dominant, avg_conf = self._dominant_prediction()
        if dominant is None or dominant == "_none_":
            self._consecutive_count = 0
            self._last_dominant = None
            return None

        # Track consecutive stability
        if dominant == self._last_dominant:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_dominant = dominant

        if self._consecutive_count < self.min_stable_frames:
            return None

        # Repetition guard
        now = time.time()
        if dominant == self._last_emitted_gloss:
            if not self._streak_broken:
                if self.repetition_gap_s <= 0.0 or (now - self._last_emitted_time) < self.repetition_gap_s:
                    return None
            elif self.repetition_gap_s > 0.0 and (now - self._last_emitted_time) < self.repetition_gap_s:
                return None

        # Emit gloss event
        event = GlossEvent(gloss=dominant, confidence=avg_conf, timestamp=now)
        self._gloss_sequence.append(event)
        if len(self._gloss_sequence) > self.max_glosses:
            self._gloss_sequence.pop(0)

        self._last_emitted_gloss = dominant
        self._last_emitted_time = now
        self._consecutive_count = 0
        self._streak_broken = False
        self._window.clear()

        return event

    @property
    def gloss_sequence(self) -> List[str]:
        """Return the current ordered list of emitted gloss labels."""
        return [e.gloss for e in self._gloss_sequence]

    @property
    def gloss_events(self) -> List[GlossEvent]:
        """Return the raw GlossEvent objects."""
        return list(self._gloss_sequence)

    def reset(self) -> None:
        """Clear the buffer and all emitted glosses."""
        self._window.clear()
        self._gloss_sequence.clear()
        self._last_emitted_gloss = None
        self._last_emitted_time = 0.0
        self._consecutive_count = 0
        self._last_dominant = None
        self._streak_broken = True

    def pop_last_gloss(self) -> Optional[str]:
        """Remove and return the most recently emitted gloss label (undo)."""
        if self._gloss_sequence:
            removed = self._gloss_sequence.pop()
            if self._gloss_sequence:
                last = self._gloss_sequence[-1]
                self._last_emitted_gloss = last.gloss
                self._last_emitted_time = last.timestamp
            else:
                self._last_emitted_gloss = None
                self._last_emitted_time = 0.0
            self._streak_broken = True
            return removed.gloss
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dominant_prediction(self) -> Tuple[Optional[str], float]:
        """Return (majority-vote label, mean confidence) from the rolling window."""
        if not self._window:
            return None, 0.0

        counts: Dict[str, int] = {}
        conf_sums: Dict[str, float] = {}
        for label, conf in self._window:
            counts[label] = counts.get(label, 0) + 1
            conf_sums[label] = conf_sums.get(label, 0.0) + conf

        dominant = max(counts, key=lambda k: counts[k])
        avg_conf = conf_sums[dominant] / counts[dominant]
        return dominant, avg_conf


# ---------------------------------------------------------------------------
# SentenceDecoder — rule-based gloss → English translation
# ---------------------------------------------------------------------------


class SentenceDecoder:
    """
    Translates a list of ISL gloss labels to natural English using
    exact-sequence lookup in sentence_grammar.json, with progressive
    and fuzzy fallbacks.

    Lookup strategy (priority order)
    ---------------------------------
    1. Exact full-sequence match in grammar JSON.
    2. Longest suffix sub-sequence match (partial-sentence lookup).
    3. Fallback: join glosses with spaces, title-cased.
    """

    def __init__(
        self,
        grammar_path: Optional[Path] = None,
    ) -> None:
        self.grammar_path = grammar_path or DEFAULT_GRAMMAR_PATH
        self._mappings: List[Dict[str, Any]] = []
        self._load_grammar()

    def _load_grammar(self) -> None:
        """Load sentence_grammar.json from disk."""
        if self.grammar_path.exists():
            try:
                with open(self.grammar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._mappings = data.get("mappings", [])
            except Exception:
                self._mappings = []

    def translate(self, gloss_sequence: List[str]) -> TranslationResult:
        """
        Translate a gloss sequence to English.

        Parameters
        ----------
        gloss_sequence : List[str]
            Ordered ISL gloss labels (e.g. ["YOU", "NAME", "WHAT"]).

        Returns
        -------
        TranslationResult
        """
        if not gloss_sequence:
            return TranslationResult(
                gloss_sequence=[],
                english_translation="",
                intent="none",
                confidence=0.0,
                matched=False,
                source="empty_input",
            )

        # 1. Exact full-sequence match
        result = self._exact_match(gloss_sequence)
        if result is not None:
            return result

        # 2. Longest suffix sub-sequence match (most-recent signs)
        for length in range(min(len(gloss_sequence), 5), 0, -1):
            sub = gloss_sequence[-length:]
            result = self._exact_match(sub)
            if result is not None:
                return result

        # 3. Fallback — join glosses
        joined = " ".join(g.title() for g in gloss_sequence)
        return TranslationResult(
            gloss_sequence=gloss_sequence,
            english_translation=joined,
            intent="unknown",
            confidence=0.0,
            matched=False,
            source="fallback_join",
        )

    def _exact_match(
        self, gloss_sequence: List[str]
    ) -> Optional[TranslationResult]:
        """Check for an exact match in the grammar mappings."""
        seq_upper = [g.strip().upper() for g in gloss_sequence]
        for mapping in self._mappings:
            m_seq = [s.strip().upper() for s in mapping.get("gloss_sequence", [])]
            if m_seq == seq_upper:
                return TranslationResult(
                    gloss_sequence=gloss_sequence,
                    english_translation=mapping.get("english_translation", " ".join(gloss_sequence)),
                    intent=mapping.get("intent", "unknown"),
                    confidence=1.0,
                    matched=True,
                    source="grammar_lookup",
                )
        return None

    def reload(self) -> None:
        """Reload grammar from disk (for hot-reload support)."""
        self._load_grammar()

    @property
    def mapping_count(self) -> int:
        return len(self._mappings)


# ---------------------------------------------------------------------------
# ContinuousTranslator — top-level orchestrator
# ---------------------------------------------------------------------------


class ContinuousTranslator:
    """
    Real-time continuous ISL sentence translator.

    Combines:
    - GlossBuffer (temporal stabilisation of frame predictions)
    - SentenceDecoder (gloss → English rule-based lookup)

    Usage in a webcam loop::

        translator = ContinuousTranslator()
        while True:
            frame = capture_frame()
            prediction = classifier.predict_features(features)
            event = translator.push(prediction.predicted_sign, prediction.confidence)
            if event:
                result = translator.translate_current()
                print(result.english_translation)

    Parameters
    ----------
    buffer_kwargs : dict
        Keyword arguments forwarded to GlossBuffer (e.g. window_size, min_stable_frames).
    grammar_path : Path, optional
        Override path to sentence_grammar.json.
    """

    def __init__(
        self,
        buffer_kwargs: Optional[Dict[str, Any]] = None,
        grammar_path: Optional[Path] = None,
    ) -> None:
        self.buffer = GlossBuffer(**(buffer_kwargs or {}))
        self.decoder = SentenceDecoder(grammar_path=grammar_path)
        self._last_translation: Optional[TranslationResult] = None

    # ------------------------------------------------------------------

    def push(self, prediction: str, confidence: float) -> Optional[GlossEvent]:
        """Submit one frame prediction; returns GlossEvent if a new gloss was emitted."""
        return self.buffer.push(prediction, confidence)

    def translate_current(self) -> TranslationResult:
        """Translate the current gloss sequence to English."""
        result = self.decoder.translate(self.buffer.gloss_sequence)
        self._last_translation = result
        return result

    def translate_sequence(self, gloss_sequence: List[str]) -> TranslationResult:
        """
        Translate an explicit list of glosses without touching the internal buffer.
        Useful for batch/API mode.
        """
        return self.decoder.translate(gloss_sequence)

    def reset(self) -> None:
        """Clear buffer and last translation."""
        self.buffer.reset()
        self._last_translation = None

    def undo_last_gloss(self) -> Optional[str]:
        """Remove the most recently emitted gloss and return its label."""
        return self.buffer.pop_last_gloss()

    @property
    def gloss_sequence(self) -> List[str]:
        return self.buffer.gloss_sequence

    @property
    def last_translation(self) -> Optional[TranslationResult]:
        return self._last_translation

    @property
    def grammar_mapping_count(self) -> int:
        return self.decoder.mapping_count


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_TRANSLATOR_INSTANCE: Optional[ContinuousTranslator] = None


def get_translator(reset: bool = False) -> ContinuousTranslator:
    """Return a cached ContinuousTranslator singleton."""
    global _TRANSLATOR_INSTANCE
    if _TRANSLATOR_INSTANCE is None or reset:
        _TRANSLATOR_INSTANCE = ContinuousTranslator()
    return _TRANSLATOR_INSTANCE
