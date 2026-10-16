"""
ISL Connect — Vision Pipeline & Landmark Detection Test Suite (Phase 3).

Verifies:
1. HandDetector initialization and model loading.
2. Graceful handling of missing or corrupt model files.
3. Frame processing on non-hand / blank frames (hand-not-detected scenario).
4. Frame processing on corrupt/invalid image inputs (bytes, empty arrays).
5. Input conversion across bytes, PIL images, and numpy arrays.
6. Landmark rendering on image frames via OpenCV.
7. Structural integrity of HAND_CONNECTIONS topology.

Tests do NOT require a physical webcam.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.hand_detector import (
    HandDetector,
    HandDetectionResult,
    HAND_CONNECTIONS,
    get_hand_detector,
)


@pytest.fixture(scope="module")
def detector():
    """Module-level HandDetector instance."""
    return get_hand_detector()


def test_hand_detector_initialization(detector):
    """Verify HandDetector initializes successfully with the bundled model."""
    assert detector is not None
    assert detector.is_initialized is True
    assert detector.init_error is None
    assert detector.landmarker is not None


def test_hand_detector_missing_model():
    """Verify HandDetector handles a missing model file gracefully without crashing."""
    fake_path = PROJECT_ROOT / "models" / "non_existent_model.task"
    bad_detector = HandDetector(model_path=fake_path)

    assert bad_detector.is_initialized is False
    assert "not found" in bad_detector.init_error.lower()

    # Processing with an uninitialized detector returns a structured result
    res = bad_detector.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert res.hands_detected is False
    assert res.error is not None
    assert "not initialized" in res.message.lower()


def test_process_frame_blank_image(detector):
    """
    Test hand-not-detected scenario:
    Process a solid black RGB image with no hands.
    """
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.process_frame(blank_frame)

    assert isinstance(result, HandDetectionResult)
    assert result.hands_detected is False
    assert result.hand_count == 0
    assert len(result.landmarks) == 0
    assert result.error is None
    assert "No hand detected" in result.message
    assert result.processed_image is not None
    assert result.processed_image.shape == (480, 640, 3)


def test_process_frame_pil_image(detector):
    """Verify PIL Image input is accepted and processed without error."""
    pil_img = Image.new("RGB", (320, 240), color="white")
    result = detector.process_frame(pil_img)

    assert result.hands_detected is False
    assert result.processed_image is not None
    assert result.processed_image.shape == (240, 320, 3)


def test_process_frame_bytes_input(detector):
    """Verify raw JPEG/PNG encoded image bytes are handled correctly."""
    import io
    pil_img = Image.new("RGB", (200, 200), color="blue")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    result = detector.process_frame(image_bytes)
    assert result.hands_detected is False
    assert result.processed_image is not None
    assert result.processed_image.shape == (200, 200, 3)


def test_process_frame_invalid_bytes(detector):
    """Verify corrupted / invalid bytes do not crash the application."""
    corrupt_bytes = b"not_an_image_random_corrupted_data_12345"
    result = detector.process_frame(corrupt_bytes)

    assert result.hands_detected is False
    assert result.hand_count == 0
    assert result.processed_image is None
    assert result.error is not None
    assert "invalid" in result.message.lower() or "empty" in result.message.lower()


def test_process_frame_empty_numpy_array(detector):
    """Verify empty numpy array is handled gracefully."""
    empty_arr = np.array([])
    result = detector.process_frame(empty_arr)

    assert result.hands_detected is False
    assert result.error is not None


def test_draw_landmarks_renders_on_image(detector):
    """
    Verify draw_landmarks modifies the canvas with keypoint circles and lines.
    Uses mock landmark points.
    """
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # Create mock hand landmark objects with x, y, z normalized attributes
    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    # 21 mock landmarks forming a small hand-like grid
    mock_hand = [MockLandmark(0.1 + (i % 5) * 0.15, 0.1 + (i // 5) * 0.2) for i in range(21)]
    mock_hands_list = [mock_hand]

    annotated = detector.draw_landmarks(canvas.copy(), mock_hands_list)

    # The canvas must have been drawn upon (not all zeros anymore)
    assert np.any(annotated > 0), "Landmarks and lines should have non-zero pixels"
    assert annotated.shape == (200, 200, 3)


def test_hand_connections_topology():
    """Verify HAND_CONNECTIONS contains valid pairs bounded between 0 and 20."""
    assert len(HAND_CONNECTIONS) == 21
    for start, end in HAND_CONNECTIONS:
        assert 0 <= start <= 20
        assert 0 <= end <= 20
        assert start != end


# ---------------------------------------------------------------------------
# Low-Light Brightness Correction in Vision Pipeline
# ---------------------------------------------------------------------------

def test_hand_detector_low_light_auto_correction(detector):
    """Verify HandDetector detects low light, brightens frame, and keeps original image."""
    dark_frame = np.full((200, 200, 3), 30, dtype=np.uint8)
    result = detector.process_frame(dark_frame)

    assert result.low_light_detected is True
    assert result.brightness_corrected is True
    assert result.mean_luminance < 75.0
    assert result.effective_gamma > 1.2
    assert result.original_image is not None
    assert np.array_equal(result.original_image, dark_frame)
    # The processed image should have been brightened
    assert np.mean(result.processed_image) > np.mean(dark_frame)


def test_hand_detector_normal_light_no_correction(detector):
    """Verify normally lit frames are untouched by the vision pipeline."""
    normal_frame = np.full((200, 200, 3), 125, dtype=np.uint8)
    result = detector.process_frame(normal_frame)

    assert result.low_light_detected is False
    assert result.brightness_corrected is False
    assert result.effective_gamma == 1.0
    # Processed image (when no hands detected) should match input exactly
    assert np.array_equal(result.processed_image, normal_frame)


def test_hand_detector_disable_auto_correction(detector):
    """Verify auto light correction can be explicitly bypassed."""
    dark_frame = np.full((200, 200, 3), 30, dtype=np.uint8)
    result = detector.process_frame(dark_frame, auto_correct_light=False)

    assert result.low_light_detected is True
    assert result.brightness_corrected is False
    assert np.array_equal(result.processed_image, dark_frame)
