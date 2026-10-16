"""
ISL Connect — Phase 6: Continuous Sentence Translation Tests.

Validates:
1. GlossBuffer temporal stabilisation and debouncing logic.
2. GlossBuffer constructor parameter validation.
3. GlossBuffer reset, undo, and max-glosses cap.
4. SentenceDecoder exact-match lookup, suffix fallback, and empty-input handling.
5. ContinuousTranslator orchestration and translate_sequence helper.
6. FastAPI /continuous/* endpoint integration tests via TestClient.
7. APIClient continuous translation methods (mocked).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.sequence_decoder import (
    GlossBuffer,
    GlossEvent,
    SentenceDecoder,
    ContinuousTranslator,
    TranslationResult,
    get_translator,
    DEFAULT_GRAMMAR_PATH,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _push_n(buf: GlossBuffer, prediction: str, confidence: float, n: int):
    """Push n identical predictions into buffer; return list of emitted events."""
    events = []
    for _ in range(n):
        ev = buf.push(prediction, confidence)
        if ev is not None:
            events.append(ev)
    return events


# ---------------------------------------------------------------------------
# 1. GlossBuffer — constructor validation
# ---------------------------------------------------------------------------

class TestGlossBufferValidation:
    def test_valid_construction(self):
        buf = GlossBuffer(window_size=20, min_stable_frames=10)
        assert buf.window_size == 20
        assert buf.min_stable_frames == 10

    def test_zero_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            GlossBuffer(window_size=0)

    def test_min_stable_frames_exceeds_window_raises(self):
        with pytest.raises(ValueError, match="min_stable_frames"):
            GlossBuffer(window_size=5, min_stable_frames=10)

    def test_negative_min_stable_frames_raises(self):
        with pytest.raises(ValueError, match="min_stable_frames"):
            GlossBuffer(window_size=5, min_stable_frames=0)

    def test_invalid_confidence_threshold_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            GlossBuffer(confidence_threshold=1.5)

    def test_confidence_threshold_zero_is_valid(self):
        buf = GlossBuffer(confidence_threshold=0.0)
        assert buf.confidence_threshold == 0.0


# ---------------------------------------------------------------------------
# 2. GlossBuffer — stabilisation and debouncing
# ---------------------------------------------------------------------------

class TestGlossBufferStabilisation:
    def test_no_event_before_min_stable_frames(self):
        buf = GlossBuffer(window_size=15, min_stable_frames=10, confidence_threshold=0.5)
        events = _push_n(buf, "HELLO", 0.9, 9)
        assert len(events) == 0, "Should not emit before min_stable_frames"

    def test_event_emitted_at_min_stable_frames(self):
        buf = GlossBuffer(window_size=15, min_stable_frames=10, confidence_threshold=0.5)
        events = _push_n(buf, "HELLO", 0.9, 10)
        assert len(events) == 1
        assert events[0].gloss == "HELLO"
        assert 0.8 < events[0].confidence <= 1.0

    def test_only_one_event_per_stable_run(self):
        buf = GlossBuffer(window_size=15, min_stable_frames=5, confidence_threshold=0.5)
        events = _push_n(buf, "WATER", 0.9, 30)
        # Should emit once then be guarded by repetition_gap_s
        assert len(events) == 1

    def test_low_confidence_suppressed(self):
        buf = GlossBuffer(window_size=10, min_stable_frames=5, confidence_threshold=0.7)
        events = _push_n(buf, "HELP", 0.5, 20)
        assert len(events) == 0

    def test_none_prediction_suppressed(self):
        buf = GlossBuffer(window_size=10, min_stable_frames=5, confidence_threshold=0.5)
        events = _push_n(buf, "None", 0.95, 20)
        assert len(events) == 0

    def test_question_mark_suppressed(self):
        buf = GlossBuffer(window_size=10, min_stable_frames=5, confidence_threshold=0.5)
        events = _push_n(buf, "?", 0.95, 20)
        assert len(events) == 0

    def test_two_sequential_different_signs(self):
        buf = GlossBuffer(
            window_size=20,
            min_stable_frames=10,
            confidence_threshold=0.5,
            repetition_gap_s=0.0,
        )
        _push_n(buf, "HELLO", 0.9, 10)
        _push_n(buf, "None", 0.0, 5)   # Break the streak
        events = _push_n(buf, "WATER", 0.9, 10)
        assert len(events) == 1
        assert events[0].gloss == "WATER"
        assert buf.gloss_sequence == ["HELLO", "WATER"]

    def test_gloss_sequence_accumulates(self):
        buf = GlossBuffer(
            window_size=20,
            min_stable_frames=5,
            confidence_threshold=0.5,
            repetition_gap_s=0.0,
        )
        for gloss in ["YOU", "NAME", "WHAT"]:
            _push_n(buf, gloss, 0.9, 5)
            _push_n(buf, "None", 0.0, 3)
        assert buf.gloss_sequence == ["YOU", "NAME", "WHAT"]


# ---------------------------------------------------------------------------
# 3. GlossBuffer — reset, undo, max_glosses cap
# ---------------------------------------------------------------------------

class TestGlossBufferManagement:
    def test_reset_clears_sequence(self):
        buf = GlossBuffer(
            window_size=10, min_stable_frames=5,
            confidence_threshold=0.5, repetition_gap_s=0.0,
        )
        _push_n(buf, "HELLO", 0.9, 5)
        assert len(buf.gloss_sequence) == 1
        buf.reset()
        assert buf.gloss_sequence == []
        assert buf._last_emitted_gloss is None

    def test_pop_last_gloss_removes_last(self):
        buf = GlossBuffer(
            window_size=10, min_stable_frames=5,
            confidence_threshold=0.5, repetition_gap_s=0.0,
        )
        for gloss in ["HELLO", "WATER"]:
            _push_n(buf, gloss, 0.9, 5)
            _push_n(buf, "None", 0.0, 3)
        assert buf.gloss_sequence == ["HELLO", "WATER"]
        removed = buf.pop_last_gloss()
        assert removed == "WATER"
        assert buf.gloss_sequence == ["HELLO"]

    def test_pop_last_on_empty_returns_none(self):
        buf = GlossBuffer()
        assert buf.pop_last_gloss() is None

    def test_max_glosses_cap(self):
        buf = GlossBuffer(
            window_size=5, min_stable_frames=3,
            confidence_threshold=0.5, repetition_gap_s=0.0,
            max_glosses=3,
        )
        glosses = ["A", "B", "C", "D", "E"]
        for g in glosses:
            _push_n(buf, g, 0.9, 3)
            _push_n(buf, "None", 0.0, 2)
        # Only last 3 should be retained
        assert len(buf.gloss_sequence) <= 3

    def test_gloss_events_list_matches_gloss_sequence(self):
        buf = GlossBuffer(
            window_size=10, min_stable_frames=5,
            confidence_threshold=0.5, repetition_gap_s=0.0,
        )
        _push_n(buf, "HELLO", 0.9, 5)
        events = buf.gloss_events
        assert len(events) == len(buf.gloss_sequence)
        assert all(isinstance(e, GlossEvent) for e in events)


# ---------------------------------------------------------------------------
# 4. SentenceDecoder — lookup logic
# ---------------------------------------------------------------------------

class TestSentenceDecoder:
    def test_exact_single_sign_match(self):
        dec = SentenceDecoder()
        result = dec.translate(["HELLO"])
        assert result.matched is True
        assert result.english_translation == "Hello!"
        assert result.source == "grammar_lookup"
        assert result.intent == "greeting"

    def test_exact_multi_sign_match(self):
        dec = SentenceDecoder()
        result = dec.translate(["YOU", "NAME", "WHAT"])
        assert result.matched is True
        assert "name" in result.english_translation.lower()

    def test_suffix_fallback_match(self):
        dec = SentenceDecoder()
        # ["EXTRA", "HELLO"] — prefix "EXTRA" won't match, but suffix ["HELLO"] will
        result = dec.translate(["EXTRA_GLOSS", "HELLO"])
        assert result.matched is True
        assert result.english_translation == "Hello!"

    def test_no_match_fallback_join(self):
        dec = SentenceDecoder()
        result = dec.translate(["FOO", "BAR", "BAZ"])
        assert result.matched is False
        assert result.source == "fallback_join"
        assert "Foo Bar Baz" in result.english_translation

    def test_empty_sequence_returns_empty(self):
        dec = SentenceDecoder()
        result = dec.translate([])
        assert result.english_translation == ""
        assert result.gloss_sequence == []
        assert result.matched is False

    def test_mapping_count_nonzero(self):
        dec = SentenceDecoder()
        assert dec.mapping_count >= 5

    def test_case_insensitive_matching(self):
        dec = SentenceDecoder()
        result = dec.translate(["hello"])
        assert result.matched is True
        assert result.english_translation == "Hello!"

    def test_reload_does_not_raise(self):
        dec = SentenceDecoder()
        dec.reload()  # Should not raise
        assert dec.mapping_count >= 0

    def test_missing_grammar_file_returns_empty_mappings(self, tmp_path):
        dec = SentenceDecoder(grammar_path=tmp_path / "nonexistent.json")
        assert dec.mapping_count == 0
        result = dec.translate(["HELLO"])
        assert result.matched is False

    def test_translate_danger_help_emergency(self):
        dec = SentenceDecoder()
        result = dec.translate(["DANGER", "HELP"])
        assert result.matched is True
        assert result.intent == "emergency"


# ---------------------------------------------------------------------------
# 5. ContinuousTranslator — orchestration
# ---------------------------------------------------------------------------

class TestContinuousTranslator:
    def test_translate_sequence_hello(self):
        tr = ContinuousTranslator()
        result = tr.translate_sequence(["HELLO"])
        assert result.matched is True
        assert result.english_translation == "Hello!"

    def test_push_and_translate_current(self):
        tr = ContinuousTranslator(buffer_kwargs={
            "window_size": 10, "min_stable_frames": 5,
            "confidence_threshold": 0.5, "repetition_gap_s": 0.0,
        })
        for _ in range(5):
            tr.push("HELLO", 0.9)
        # Sequence should now contain "HELLO"
        result = tr.translate_current()
        assert isinstance(result, TranslationResult)
        assert "HELLO" in result.gloss_sequence or result.english_translation != ""

    def test_reset_clears_state(self):
        tr = ContinuousTranslator(buffer_kwargs={
            "window_size": 10, "min_stable_frames": 5,
            "confidence_threshold": 0.5, "repetition_gap_s": 0.0,
        })
        for _ in range(5):
            tr.push("HELLO", 0.9)
        tr.reset()
        assert tr.gloss_sequence == []
        assert tr.last_translation is None

    def test_undo_last_gloss(self):
        tr = ContinuousTranslator(buffer_kwargs={
            "window_size": 10, "min_stable_frames": 5,
            "confidence_threshold": 0.5, "repetition_gap_s": 0.0,
        })
        for _ in range(5):
            tr.push("HELLO", 0.9)
        tr.push("None", 0.0)  # Break streak
        for _ in range(5):
            tr.push("WATER", 0.9)
        assert "WATER" in tr.gloss_sequence
        removed = tr.undo_last_gloss()
        assert removed == "WATER"

    def test_grammar_mapping_count_nonzero(self):
        tr = ContinuousTranslator()
        assert tr.grammar_mapping_count >= 5

    def test_get_translator_singleton(self):
        t1 = get_translator()
        t2 = get_translator()
        assert t1 is t2

    def test_get_translator_reset_returns_new(self):
        t1 = get_translator(reset=True)
        t2 = get_translator(reset=True)
        assert t1 is not t2


# ---------------------------------------------------------------------------
# 6. FastAPI /continuous/* endpoint tests
# ---------------------------------------------------------------------------

class TestContinuousEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_translate_hello(self, client):
        resp = client.post(
            "/continuous/translate",
            json={"gloss_sequence": ["HELLO"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["english_translation"] == "Hello!"
        assert data["intent"] == "greeting"

    def test_translate_you_name_what(self, client):
        resp = client.post(
            "/continuous/translate",
            json={"gloss_sequence": ["YOU", "NAME", "WHAT"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True

    def test_translate_unknown_gloss_fallback(self, client):
        resp = client.post(
            "/continuous/translate",
            json={"gloss_sequence": ["GLOSS_XYZ_UNKNOWN"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is False
        assert data["source"] == "fallback_join"

    def test_translate_empty_sequence_rejected(self, client):
        resp = client.post(
            "/continuous/translate",
            json={"gloss_sequence": []},
        )
        # Pydantic min_length=1 validation should reject this
        assert resp.status_code == 422

    def test_translate_batch(self, client):
        resp = client.post(
            "/continuous/translate-batch",
            json={"sequences": [["HELLO"], ["GOODBYE"], ["THANK-YOU"]]},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 3
        assert results[0]["matched"] is True
        assert results[1]["matched"] is True

    def test_translate_batch_too_large_rejected(self, client):
        resp = client.post(
            "/continuous/translate-batch",
            json={"sequences": [["HELLO"]] * 51},
        )
        assert resp.status_code == 400

    def test_session_reset(self, client):
        resp = client.post("/continuous/session/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_session_push_and_result(self, client):
        # Reset first
        client.post("/continuous/session/reset")
        # Push enough frames to stabilise on HELLO (default min_stable_frames=10)
        for _ in range(15):
            resp = client.post(
                "/continuous/session/push",
                json={"prediction": "HELLO", "confidence": 0.9},
            )
            assert resp.status_code == 200

        resp = client.get("/continuous/session/result")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["gloss_sequence"], list)
        assert isinstance(data["buffer_length"], int)

    def test_session_push_low_confidence_ignored(self, client):
        client.post("/continuous/session/reset")
        for _ in range(20):
            client.post(
                "/continuous/session/push",
                json={"prediction": "HELLO", "confidence": 0.1},
            )
        resp = client.get("/continuous/session/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gloss_sequence"] == []

    def test_get_grammar_mappings(self, client):
        resp = client.get("/continuous/grammar")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_mappings" in data
        assert data["total_mappings"] >= 5
        assert "mappings" in data
        assert all("gloss_sequence" in m for m in data["mappings"])
        assert all("english_translation" in m for m in data["mappings"])


# ---------------------------------------------------------------------------
# 7. APIClient continuous methods — mocked
# ---------------------------------------------------------------------------

class TestAPIClientContinuous:
    @pytest.fixture
    def client(self):
        from frontend.api_client import APIClient
        return APIClient(base_url="http://testserver", timeout=2.0)

    def test_translate_gloss_sequence_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "gloss_sequence": ["HELLO"],
            "english_translation": "Hello!",
            "intent": "greeting",
            "confidence": 1.0,
            "matched": True,
            "source": "grammar_lookup",
        }
        with patch("requests.post", return_value=mock_resp):
            result = client.translate_gloss_sequence(["HELLO"])
        assert result["success"] is True
        assert result["data"]["matched"] is True

    def test_push_frame_prediction_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "gloss_sequence": [],
            "translation": None,
            "buffer_length": 1,
        }
        with patch("requests.post", return_value=mock_resp):
            result = client.push_frame_prediction("HELLO", 0.85)
        assert result["success"] is True

    def test_get_session_result_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "gloss_sequence": ["HELLO"],
            "translation": None,
            "buffer_length": 5,
        }
        with patch("requests.get", return_value=mock_resp):
            result = client.get_session_result()
        assert result["success"] is True

    def test_reset_session_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "message": "reset"}
        with patch("requests.post", return_value=mock_resp):
            result = client.reset_session()
        assert result["success"] is True

    def test_get_grammar_mappings_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_mappings": 19,
            "mappings": [{"gloss_sequence": ["HELLO"], "english_translation": "Hello!", "intent": "greeting"}],
        }
        with patch("requests.get", return_value=mock_resp):
            result = client.get_grammar_mappings()
        assert result["success"] is True
        assert result["data"]["total_mappings"] == 19

    def test_translate_gloss_sequence_connection_error(self, client):
        from requests.exceptions import ConnectionError
        with patch("requests.post", side_effect=ConnectionError("refused")):
            result = client.translate_gloss_sequence(["HELLO"])
        assert result["success"] is False
        assert "refused" in result["error"]
