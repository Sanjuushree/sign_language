"""
ISL Connect — ISL Alphabet Classifier (Phase 5).

Loads trained sign classification model and performs low-latency inference
on extracted landmark feature vectors or direct camera frames.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.recognition.feature_extractor import LandmarkFeatureExtractor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "isl_classifier.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "classifier_metadata.json"

# Standard ISL Alphabet classes (A–Z)
ISL_ALPHABET = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


@dataclass
class SignPrediction:
    """Structured sign prediction output."""
    predicted_sign: str
    confidence: float
    is_confident: bool
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    hands_detected: bool = False
    latency_ms: float = 0.0
    message: str = ""
    error: Optional[str] = None


class ISLSignClassifier:
    """Predicts ISL alphabet gestures from hand landmarks or images."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.50,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
        self.confidence_threshold = confidence_threshold

        self.model = None
        self.classes: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self.feature_extractor = LandmarkFeatureExtractor()
        self._try_load()

    def _try_load(self) -> bool:
        """Attempt to load the model and metadata from disk."""
        if not self.model_path.exists():
            self.load_error = f"Model file not found at {self.model_path}"
            return False

        try:
            import joblib
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded["model"]
                self.classes = loaded.get("classes", [])
            else:
                self.model = loaded
                if hasattr(self.model, "classes_"):
                    self.classes = [str(c) for c in self.model.classes_]

            # Load metadata JSON if present
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    if not self.classes and "classes" in self.metadata:
                        self.classes = self.metadata["classes"]
                except Exception:
                    pass

            if not self.classes:
                self.classes = ISL_ALPHABET

            self.is_loaded = True
            self.load_error = None
            return True
        except Exception as e:
            self.load_error = f"Failed to load model from {self.model_path}: {e}"
            self.is_loaded = False
            return False

    def predict_features(self, features: np.ndarray) -> SignPrediction:
        """
        Predict ISL alphabet from a 151-dimensional feature vector.

        Parameters:
        -----------
        features : np.ndarray of shape (151,)
        """
        start_t = time.perf_counter()

        # If model is not loaded, return structured fallback
        if not self.is_loaded or self.model is None:
            return SignPrediction(
                predicted_sign="?",
                confidence=0.0,
                is_confident=False,
                hands_detected=bool(features[148] > 0),
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
                message="No trained model currently loaded.",
                error=self.load_error or "Model not loaded",
            )

        # Check if hands are present
        hands_detected = bool(features[148] > 0 or features[149] > 0)
        if not hands_detected:
            return SignPrediction(
                predicted_sign="None",
                confidence=0.0,
                is_confident=False,
                hands_detected=False,
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
                message="No hands detected in frame.",
            )

        feat_2d = features.reshape(1, -1)

        try:
            # Predict probabilities if supported
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feat_2d)[0]
                top_indices = np.argsort(probs)[::-1]
                top_k = []
                for idx in top_indices[:5]:
                    c_name = str(self.classes[idx]) if idx < len(self.classes) else str(idx)
                    top_k.append((c_name, float(probs[idx])))

                best_idx = top_indices[0]
                best_class = str(self.classes[best_idx]) if best_idx < len(self.classes) else str(best_idx)
                confidence = float(probs[best_idx])
            else:
                pred = self.model.predict(feat_2d)[0]
                best_class = str(pred)
                confidence = 1.0
                top_k = [(best_class, 1.0)]

            is_confident = confidence >= self.confidence_threshold
            predicted_label = best_class if is_confident else f"{best_class} (?)"
            latency = (time.perf_counter() - start_t) * 1000.0

            msg = f"Sign recognized: '{best_class}' ({confidence * 100:.1f}%)" if is_confident else (
                f"Low confidence ({confidence * 100:.1f}%) for '{best_class}'"
            )

            return SignPrediction(
                predicted_sign=predicted_label,
                confidence=confidence,
                is_confident=is_confident,
                top_k=top_k,
                hands_detected=True,
                latency_ms=latency,
                message=msg,
            )
        except Exception as e:
            return SignPrediction(
                predicted_sign="?",
                confidence=0.0,
                is_confident=False,
                hands_detected=hands_detected,
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
                message=f"Prediction error: {e}",
                error=str(e),
            )

    def predict_landmarks(
        self,
        landmarks_list: List[List[Dict[str, float]]],
    ) -> SignPrediction:
        """Predict sign directly from MediaPipe landmarks output."""
        if not landmarks_list:
            return SignPrediction(
                predicted_sign="None",
                confidence=0.0,
                is_confident=False,
                hands_detected=False,
                message="No landmarks provided.",
            )

        features = self.feature_extractor.extract_features(landmarks_list)
        return self.predict_features(features)

    def predict_image(
        self,
        image_input: Any,
        detector: Any = None,
    ) -> Tuple[SignPrediction, Optional[Any]]:
        """
        Process an image frame through HandDetector and classify the sign.

        Parameters:
        -----------
        image_input : numpy array, PIL Image, or bytes
        detector : HandDetector instance (if None, calls get_hand_detector())

        Returns:
        --------
        (SignPrediction, HandDetectionResult)
        """
        if detector is None:
            from src.vision.hand_detector import get_hand_detector
            detector = get_hand_detector()

        detection_result = detector.process_frame(image_input)
        if not detection_result.hands_detected:
            return SignPrediction(
                predicted_sign="None",
                confidence=0.0,
                is_confident=False,
                hands_detected=False,
                message=detection_result.message or "No hands detected in image.",
                error=detection_result.error,
            ), detection_result

        prediction = self.predict_landmarks(detection_result.landmarks)
        return prediction, detection_result


# Singleton instance
_CLASSIFIER_INSTANCE: Optional[ISLSignClassifier] = None


def get_classifier(reload: bool = False) -> ISLSignClassifier:
    """Return a cached or newly loaded ISLSignClassifier instance."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None or reload:
        _CLASSIFIER_INSTANCE = ISLSignClassifier()
    return _CLASSIFIER_INSTANCE
