"""
ISL Connect — Landmark Feature Extractor (Phase 5).

Transforms raw 3D hand landmark coordinates from MediaPipe into a scale-,
rotation-, and translation-invariant feature representation suitable for
machine learning classifiers.

Supports both single-hand and two-hand Indian Sign Language (ISL) gestures:
- Wrist-relative translation normalization (landmark 0 as origin).
- Max-span euclidean scale normalization (invariant to distance from camera).
- Inter-finger tip distances (open vs. closed/curled finger geometries).
- Dual-hand spatial interaction vectors (for two-handed ISL signs like B, D, P).
- Fixed-length 151-dimensional numerical vector.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Landmark indices for finger tips and wrist
WRIST_IDX = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
TOTAL_FEATURE_DIM = 151


def normalize_single_hand(landmarks: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize 21 coordinates for a single hand.

    Returns:
    --------
    normalized_coords : np.ndarray
        Shape (21, 3), coordinates relative to wrist and divided by max euclidean span.
    raw_coords : np.ndarray
        Shape (21, 3), unnormalized coordinates.
    scale : float
        Maximum distance from wrist (scale factor).
    """
    coords = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks[:21]):
        coords[i] = [lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)]

    wrist = coords[WRIST_IDX].copy()
    rel_coords = coords - wrist

    # Euclidean distance of all points from wrist
    distances = np.linalg.norm(rel_coords, axis=1)
    max_dist = float(np.max(distances))
    scale = max_dist if max_dist > 1e-6 else 1.0

    normalized_coords = rel_coords / scale
    return normalized_coords, coords, scale


def compute_finger_geometry(normalized_coords: np.ndarray) -> np.ndarray:
    """
    Compute auxiliary geometric features for a single hand:
    - 5 distances from wrist to each finger tip
    - 4 adjacent fingertip-to-fingertip distances (thumb-index, index-mid, mid-ring, ring-pinky)
    Returns:
    --------
    geom_features : np.ndarray of shape (9,)
    """
    geom = np.zeros(9, dtype=np.float32)
    # Wrist to tips (5)
    for idx, tip in enumerate(FINGER_TIPS):
        geom[idx] = np.linalg.norm(normalized_coords[tip])

    # Adjacent tip-to-tip distances (4)
    geom[5] = np.linalg.norm(normalized_coords[THUMB_TIP] - normalized_coords[INDEX_TIP])
    geom[6] = np.linalg.norm(normalized_coords[INDEX_TIP] - normalized_coords[MIDDLE_TIP])
    geom[7] = np.linalg.norm(normalized_coords[MIDDLE_TIP] - normalized_coords[RING_TIP])
    geom[8] = np.linalg.norm(normalized_coords[RING_TIP] - normalized_coords[PINKY_TIP])

    return geom


class LandmarkFeatureExtractor:
    """Extracts invariant 151-dimensional feature vectors from hand landmarks."""

    FEATURE_DIM = TOTAL_FEATURE_DIM

    def __init__(self):
        pass

    @staticmethod
    def get_feature_dimension() -> int:
        """Return the fixed feature vector length."""
        return TOTAL_FEATURE_DIM

    def extract_features(
        self,
        landmarks_list: List[List[Dict[str, float]]],
    ) -> np.ndarray:
        """
        Extract fixed-length feature vector from detected hands landmarks.

        Parameters:
        -----------
        landmarks_list : list of list of dict
            MediaPipe landmarks output, where each element is a list of 21 landmark dicts
            with 'x', 'y', 'z' keys.

        Returns:
        --------
        features : np.ndarray of shape (151,)
        """
        features = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

        if not landmarks_list:
            return features

        # Sort hands deterministically by wrist X coordinate (left to right in frame)
        sorted_hands = sorted(
            landmarks_list[:2],
            key=lambda hand: hand[0].get("x", 0.0) if len(hand) > 0 else 0.0
        )

        hand_count = len(sorted_hands)
        features[148] = 1.0 if hand_count >= 1 else 0.0  # has_hand_1
        features[149] = 1.0 if hand_count >= 2 else 0.0  # has_hand_2
        features[150] = float(min(hand_count, 2))        # hand_count

        hand1_raw = None
        hand2_raw = None
        scale1 = 1.0
        scale2 = 1.0

        # Hand 1 features (indices 0..71)
        # 0..62: 21 * 3 normalized coordinates
        # 63..71: 9 geometric finger features
        if hand_count >= 1 and len(sorted_hands[0]) >= 21:
            norm1, raw1, s1 = normalize_single_hand(sorted_hands[0])
            hand1_raw = raw1
            scale1 = s1
            features[0:63] = norm1.flatten()
            features[63:72] = compute_finger_geometry(norm1)

        # Hand 2 features (indices 72..143)
        # 72..134: 21 * 3 normalized coordinates
        # 135..143: 9 geometric finger features
        if hand_count >= 2 and len(sorted_hands[1]) >= 21:
            norm2, raw2, s2 = normalize_single_hand(sorted_hands[1])
            hand2_raw = raw2
            scale2 = s2
            features[72:135] = norm2.flatten()
            features[135:144] = compute_finger_geometry(norm2)

        # Two-hand interaction features (indices 144..147)
        # Delta vector and euclidean distance between wrist 1 and wrist 2
        if hand1_raw is not None and hand2_raw is not None:
            avg_scale = (scale1 + scale2) / 2.0
            if avg_scale < 1e-6:
                avg_scale = 1.0
            wrist_delta = (hand2_raw[WRIST_IDX] - hand1_raw[WRIST_IDX]) / avg_scale
            features[144:147] = wrist_delta[:3]
            features[147] = float(np.linalg.norm(wrist_delta))

        return features

    def extract_from_detection_result(self, detection_result: Any) -> Tuple[np.ndarray, bool]:
        """
        Extract features from a HandDetectionResult object.

        Returns:
        --------
        features : np.ndarray of shape (151,)
        hands_detected : bool
        """
        if not detection_result or not getattr(detection_result, "hands_detected", False):
            return np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32), False

        landmarks = getattr(detection_result, "landmarks", [])
        features = self.extract_features(landmarks)
        return features, True
