"""
ISL Connect — Recognition & Gesture Classification Test Suite (Phase 5).

Verifies:
1. LandmarkFeatureExtractor feature dimensions, invariant normalization, and dual-hand interaction.
2. ISLSignClassifier inference, confidence scoring, top-k ranking, and edge case fallbacks.
3. Backend /recognition/supported-signs and /recognition/predict REST endpoints.
4. Frontend APIClient recognition methods.
"""

from pathlib import Path
import sys
from typing import Dict, List
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from frontend.api_client import APIClient
from src.recognition.feature_extractor import LandmarkFeatureExtractor, TOTAL_FEATURE_DIM
from src.recognition.classifier import (
    ISLSignClassifier,
    SignPrediction,
    get_classifier,
    ISL_ALPHABET,
)


@pytest.fixture
def feature_extractor():
    """Fixture providing a LandmarkFeatureExtractor."""
    return LandmarkFeatureExtractor()


@pytest.fixture
def classifier():
    """Fixture providing loaded ISLSignClassifier instance."""
    return get_classifier(reload=True)


@pytest.fixture
def test_client():
    """Fixture providing FastAPI test client."""
    return TestClient(app)


def make_dummy_hand_landmarks(wrist_x: float = 0.5, wrist_y: float = 0.5) -> List[Dict[str, float]]:
    """Generate 21 synthetic landmark points forming a realistic hand cluster."""
    landmarks = []
    for i in range(21):
        # Slightly offset points around wrist
        landmarks.append({
            "x": wrist_x + (i % 5) * 0.02,
            "y": wrist_y - (i // 5) * 0.03,
            "z": 0.01 * (i % 3),
        })
    return landmarks


# ═══════════════════════════════════════════════════════════════════════════
# 1. Feature Extractor Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_feature_extractor_dimension(feature_extractor):
    """Verify feature extractor produces exact 151-dimensional vectors."""
    assert feature_extractor.get_feature_dimension() == 151
    assert LandmarkFeatureExtractor.FEATURE_DIM == 151


def test_feature_extractor_empty_input(feature_extractor):
    """Verify empty landmarks list produces 151-dimensional zero vector."""
    features = feature_extractor.extract_features([])
    assert isinstance(features, np.ndarray)
    assert features.shape == (151,)
    assert np.all(features == 0)


def test_feature_extractor_single_hand(feature_extractor):
    """Verify single hand extraction populates hand 1 slots and presence flags."""
    hand1 = make_dummy_hand_landmarks(0.5, 0.5)
    features = feature_extractor.extract_features([hand1])

    assert features.shape == (151,)
    # Hand presence flags
    assert features[148] == 1.0  # has_hand_1
    assert features[149] == 0.0  # has_hand_2
    assert features[150] == 1.0  # hand_count

    # Hand 1 features are non-zero
    assert np.any(features[0:63] != 0)
    assert np.any(features[63:72] != 0)

    # Hand 2 features are zero
    assert np.all(features[72:144] == 0)


def test_feature_extractor_two_hands(feature_extractor):
    """Verify two hands extraction populates both hand slots and dual-hand interactions."""
    hand1 = make_dummy_hand_landmarks(0.3, 0.5)
    hand2 = make_dummy_hand_landmarks(0.7, 0.5)
    features = feature_extractor.extract_features([hand1, hand2])

    assert features.shape == (151,)
    assert features[148] == 1.0  # has_hand_1
    assert features[149] == 1.0  # has_hand_2
    assert features[150] == 2.0  # hand_count

    # Both hands populated
    assert np.any(features[0:63] != 0)
    assert np.any(features[72:135] != 0)

    # Two-hand interaction delta
    assert np.any(features[144:148] != 0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Classifier Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_classifier_loaded(classifier):
    """Verify ISL classifier is loaded and initialized with 26 classes."""
    assert classifier.is_loaded is True
    assert len(classifier.classes) == 26
    assert classifier.classes == sorted(ISL_ALPHABET)


def test_classifier_metadata(classifier):
    """Verify classifier metadata exists and reports test accuracy."""
    assert classifier.metadata is not None
    assert "test_accuracy" in classifier.metadata
    assert classifier.metadata["test_accuracy"] >= 0.70


def test_classifier_predict_features(classifier, feature_extractor):
    """Verify classifier predicts valid ISL letter from synthetic landmarks."""
    hand1 = make_dummy_hand_landmarks(0.5, 0.5)
    features = feature_extractor.extract_features([hand1])

    prediction = classifier.predict_features(features)
    assert isinstance(prediction, SignPrediction)
    assert prediction.hands_detected is True
    clean_sign = prediction.predicted_sign.replace(" (?)", "")
    assert clean_sign in ISL_ALPHABET
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.top_k) > 0


def test_classifier_predict_no_hands(classifier):
    """Verify classifier returns structured None prediction when no hands detected."""
    zero_features = np.zeros(151, dtype=np.float32)
    pred = classifier.predict_features(zero_features)

    assert pred.predicted_sign == "None"
    assert pred.hands_detected is False
    assert pred.confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. FastAPI Route Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_api_supported_signs(test_client):
    """Verify GET /recognition/supported-signs returns 200 with 26 classes."""
    res = test_client.get("/recognition/supported-signs")
    assert res.status_code == 200
    data = res.json()
    assert data["model_loaded"] is True
    assert data["total_classes"] == 26
    assert len(data["classes"]) == 26
    assert data["accuracy"] is not None


def test_api_predict_landmarks(test_client):
    """Verify POST /recognition/predict-landmarks returns valid prediction."""
    hand1 = make_dummy_hand_landmarks(0.5, 0.5)
    payload = {"landmarks": [hand1]}

    res = test_client.post("/recognition/predict-landmarks", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_sign" in data
    assert "confidence" in data
    assert data["hands_detected"] is True


def test_api_predict_image(test_client):
    """Verify POST /recognition/predict processes uploaded image frame."""
    letters_dir = PROJECT_ROOT / "dataset" / "raw" / "Letters"
    sample_file = letters_dir / "A.jpg" if letters_dir.exists() else None

    if sample_file and sample_file.exists():
        with open(sample_file, "rb") as f:
            res = test_client.post(
                "/recognition/predict",
                files={"file": ("A.jpg", f, "image/jpeg")},
            )
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_sign"] == "A"
        assert data["hands_detected"] is True
        assert data["confidence"] > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 4. API Client Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_api_client_recognition_methods():
    """Verify APIClient methods for recognition endpoints."""
    client = APIClient()
    # Offline or dummy url check (graceful failure without uncaught exception)
    dummy_client = APIClient(base_url="http://127.0.0.1:9999", timeout=0.5)
    res_signs = dummy_client.get_supported_signs()
    assert res_signs["success"] is False

    res_pred = dummy_client.predict_image(b"fakebytes")
    assert res_pred["success"] is False
