"""
ISL Connect — Vision Pipeline: Hand Landmark Detection (Phase 3).

Detects 21 hand landmarks using MediaPipe Tasks API and draws landmarks
and skeletal connections using OpenCV.
"""

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.preprocessing.image_processor import (
    adaptive_light_correction,
    assess_illumination,
)

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Standard MediaPipe 21 Hand Landmark Connections
# ---------------------------------------------------------------------------
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Palm
    (0, 1), (0, 5), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20),
)

# Default path to the local model asset file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "hand_landmarker.task"


@dataclass
class HandDetectionResult:
    """Structured result returned by the HandDetector."""
    hands_detected: bool
    hand_count: int
    processed_image: Optional[np.ndarray] = None  # RGB image with landmarks drawn
    landmarks: List[List[Dict[str, float]]] = field(default_factory=list)
    message: str = ""
    error: Optional[str] = None
    low_light_detected: bool = False
    brightness_corrected: bool = False
    mean_luminance: float = 0.0
    effective_gamma: float = 1.0
    original_image: Optional[np.ndarray] = None   # Untouched input frame (RGB)


class HandDetector:
    """Encapsulates MediaPipe HandLandmarker with OpenCV landmark rendering."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        auto_light_correction: bool = True,
        low_light_threshold: float = 75.0,
    ):
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.auto_light_correction = auto_light_correction
        self.low_light_threshold = low_light_threshold
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.landmarker = None
        self.is_initialized = False
        self.init_error = None

        if not MEDIAPIPE_AVAILABLE:
            self.init_error = "MediaPipe is not installed. Please install mediapipe."
            return

        self._init_landmarker(
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _init_landmarker(
        self,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """Initialize the MediaPipe HandLandmarker instance."""
        if not self.model_path.exists():
            self.init_error = (
                f"Model file not found at {self.model_path}. "
                "Ensure models/hand_landmarker.task exists."
            )
            return

        try:
            base_options = python.BaseOptions(
                model_asset_path=str(self.model_path)
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.is_initialized = True
        except Exception as e:
            self.init_error = f"Failed to initialize MediaPipe HandLandmarker: {str(e)}"
            self.is_initialized = False

    def process_frame(
        self,
        image_input: Union[np.ndarray, bytes, Image.Image],
        is_bgr: bool = False,
        auto_correct_light: Optional[bool] = None,
    ) -> HandDetectionResult:
        """
        Process a single image/camera frame, automatically correcting low light if needed,
        and detecting hands via MediaPipe.

        :param image_input: Raw image bytes, PIL Image, or numpy array (RGB or BGR).
        :param is_bgr: Set to True if image_input is a BGR numpy array (e.g. from cv2.imread).
        :param auto_correct_light: Override detector auto_light_correction setting if provided.
        :return: HandDetectionResult with landmarks, original image, and rendered image.
        """
        if not self.is_initialized:
            return HandDetectionResult(
                hands_detected=False,
                hand_count=0,
                processed_image=None,
                message="Hand detector not initialized.",
                error=self.init_error or "Detector unavailable.",
            )

        # 1. Convert input to RGB numpy array
        rgb_image = self._convert_to_rgb_array(image_input, is_bgr=is_bgr)
        if rgb_image is None or rgb_image.size == 0:
            return HandDetectionResult(
                hands_detected=False,
                hand_count=0,
                processed_image=None,
                message="Empty or invalid image frame received.",
                error="Invalid frame format.",
            )

        original_rgb = rgb_image.copy()

        # 2. Illumination assessment & automatic adaptive correction
        use_correction = self.auto_light_correction if auto_correct_light is None else auto_correct_light
        low_light_detected = False
        brightness_corrected = False
        mean_lum = 0.0
        eff_gamma = 1.0

        if use_correction:
            corrected_rgb, brightness_corrected, info = adaptive_light_correction(
                rgb_image,
                is_bgr=False,
                low_light_threshold=self.low_light_threshold,
            )
            low_light_detected = info.is_low_light
            mean_lum = info.mean_luminance
            eff_gamma = info.recommended_gamma
            detection_target = corrected_rgb if brightness_corrected else rgb_image
        else:
            detection_target = rgb_image
            info = assess_illumination(rgb_image, is_bgr=False, low_light_threshold=self.low_light_threshold)
            low_light_detected = info.is_low_light
            mean_lum = info.mean_luminance

        # 3. Run MediaPipe Detection on detection_target
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detection_target)
            detection_result = self.landmarker.detect(mp_image)
        except Exception as e:
            return HandDetectionResult(
                hands_detected=False,
                hand_count=0,
                processed_image=detection_target,
                message="Error processing frame with MediaPipe.",
                error=str(e),
                low_light_detected=low_light_detected,
                brightness_corrected=brightness_corrected,
                mean_luminance=mean_lum,
                effective_gamma=eff_gamma,
                original_image=original_rgb,
            )

        # 4. Extract landmark coordinates
        detected_hands = detection_result.hand_landmarks or []
        hand_count = len(detected_hands)

        if hand_count == 0:
            return HandDetectionResult(
                hands_detected=False,
                hand_count=0,
                processed_image=detection_target,
                message="No hand detected in the camera frame.",
                error=None,
                low_light_detected=low_light_detected,
                brightness_corrected=brightness_corrected,
                mean_luminance=mean_lum,
                effective_gamma=eff_gamma,
                original_image=original_rgb,
            )

        # 5. Draw landmarks and connections on detection_target
        annotated_image = self.draw_landmarks(detection_target.copy(), detected_hands)

        # 6. Format landmark coordinates
        all_landmarks = []
        for hand in detected_hands:
            hand_coords = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in hand
            ]
            all_landmarks.append(hand_coords)

        hand_label = "1 hand" if hand_count == 1 else f"{hand_count} hands"
        return HandDetectionResult(
            hands_detected=True,
            hand_count=hand_count,
            processed_image=annotated_image,
            landmarks=all_landmarks,
            message=f"Successfully detected {hand_label} with 21 landmarks each.",
            error=None,
            low_light_detected=low_light_detected,
            brightness_corrected=brightness_corrected,
            mean_luminance=mean_lum,
            effective_gamma=eff_gamma,
            original_image=original_rgb,
        )

    def draw_landmarks(
        self,
        image_rgb: np.ndarray,
        hand_landmarks_list: List[Any],
    ) -> np.ndarray:
        """
        Draw 21 hand landmarks and connections onto the RGB image.

        :param image_rgb: RGB image numpy array.
        :param hand_landmarks_list: List of detected hands from MediaPipe.
        :return: Annotated RGB image numpy array.
        """
        height, width, _ = image_rgb.shape

        for hand_landmarks in hand_landmarks_list:
            # First draw skeletal connection lines (green/cyan)
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    lm_start = hand_landmarks[start_idx]
                    lm_end = hand_landmarks[end_idx]

                    pt1 = (int(lm_start.x * width), int(lm_start.y * height))
                    pt2 = (int(lm_end.x * width), int(lm_end.y * height))

                    cv2.line(image_rgb, pt1, pt2, color=(37, 99, 235), thickness=2)  # Blue line

            # Draw landmark keypoints (solid vibrant dots with border)
            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * width), int(lm.y * height)

                # Special color for wrist (index 0) and fingertips (4, 8, 12, 16, 20)
                if idx in (4, 8, 12, 16, 20):
                    dot_color = (239, 68, 68)   # Red/coral for fingertips
                    radius = 5
                elif idx == 0:
                    dot_color = (16, 185, 129)  # Green for wrist base
                    radius = 6
                else:
                    dot_color = (59, 130, 246)  # Blue for joints
                    radius = 4

                # Outer ring and filled circle
                cv2.circle(image_rgb, (cx, cy), radius + 2, (255, 255, 255), thickness=1)
                cv2.circle(image_rgb, (cx, cy), radius, dot_color, thickness=-1)

        return image_rgb

    @staticmethod
    def _convert_to_rgb_array(
        image_input: Union[np.ndarray, bytes, Image.Image],
        is_bgr: bool = False,
    ) -> Optional[np.ndarray]:
        """Convert various image input formats to a contiguous RGB uint8 numpy array."""
        try:
            if isinstance(image_input, bytes):
                # Decode bytes using PIL
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                return np.array(pil_img, dtype=np.uint8)

            elif isinstance(image_input, Image.Image):
                return np.array(image_input.convert("RGB"), dtype=np.uint8)

            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 2:
                    # Grayscale to RGB
                    return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                elif image_input.ndim == 3:
                    if image_input.shape[2] == 4:
                        # RGBA/BGRA to RGB
                        code = cv2.COLOR_BGRA2RGB if is_bgr else cv2.COLOR_RGBA2RGB
                        return cv2.cvtColor(image_input, code)
                    elif image_input.shape[2] == 3:
                        if is_bgr:
                            return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                        return np.ascontiguousarray(image_input, dtype=np.uint8)
            return None
        except Exception:
            return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
            self.is_initialized = False


# Module-level singleton instance for shared use
_default_detector: Optional[HandDetector] = None


def get_hand_detector() -> HandDetector:
    """Retrieve or initialize the global HandDetector instance."""
    global _default_detector
    if _default_detector is None or not _default_detector.is_initialized:
        _default_detector = HandDetector()
    return _default_detector
