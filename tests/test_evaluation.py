"""
ISL Connect — Model Evaluation and Diagnostics Test Suite (Phase 6).

Verifies:
1. Baseline accuracy replication (80.43% on held-out test data).
2. Confusion matrix dimensions (26x26) and support consistency (506 samples).
3. Per-letter metric completeness for all 26 ISL alphabet classes (A–Z).
4. Identification of top misclassified pairs (I->Y, Q->N, S->G, K->V/T, W->N).
5. Confidence calibration (higher confidence on correct vs error samples).
6. Pipeline consistency between HandDetector, LandmarkFeatureExtractor, and Classifier.
7. Safe artifact serialization without model modification.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.classifier import get_classifier
from src.recognition.evaluator import ISLModelEvaluator, EvaluationReport
from src.recognition.feature_extractor import LandmarkFeatureExtractor, TOTAL_FEATURE_DIM


@pytest.fixture
def evaluator():
    """Fixture providing initialized ISLModelEvaluator."""
    return ISLModelEvaluator()


def test_evaluator_initialization(evaluator):
    """Verify evaluator loads trained model and 26 ISL classes correctly."""
    assert evaluator.model is not None
    assert len(evaluator.classes) == 26
    assert evaluator.classes[0] == "A"
    assert evaluator.classes[-1] == "Z"


def test_baseline_accuracy_replication(evaluator):
    """
    Verify reproducible baseline evaluation on held-out test data.
    Must replicate the verified 80.43% test accuracy (407/506 correct).
    """
    report = evaluator.evaluate()
    assert isinstance(report, EvaluationReport)
    assert report.total_test_samples == 506
    assert report.total_correct == 407
    assert report.total_errors == 99
    assert report.overall_accuracy == 0.8043
    assert abs(report.overall_accuracy - (407 / 506)) < 1e-4


def test_confusion_matrix_structure(evaluator):
    """Verify confusion matrix is 26x26 and total elements equal test sample count."""
    report = evaluator.evaluate()
    cm = np.array(report.confusion_matrix)

    assert cm.shape == (26, 26)
    assert int(np.sum(cm)) == 506
    # Trace of confusion matrix must equal total correct
    assert int(np.trace(cm)) == 407


def test_per_class_metrics_all_26_letters(evaluator):
    """Verify every letter from A to Z has complete, non-null evaluation metrics."""
    report = evaluator.evaluate()
    assert len(report.per_class_metrics) == 26

    for char_code in range(ord("A"), ord("Z") + 1):
        letter = chr(char_code)
        assert letter in report.per_class_metrics
        m = report.per_class_metrics[letter]

        assert "precision" in m
        assert "recall" in m
        assert "f1_score" in m
        assert "support" in m
        assert "correct_count" in m
        assert "errors_count" in m

        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0
        assert m["correct_count"] + m["errors_count"] == m["support"]


def test_top_confusions_detected(evaluator):
    """Verify that known error modes are detected in top_confusions list."""
    report = evaluator.evaluate()
    confusions = report.top_confusions
    assert len(confusions) > 0

    pairs = {(c["true_class"], c["predicted_class"]): c["count"] for c in confusions}

    # Verify primary confusion pairs identified in Phase 6 analysis
    assert ("I", "Y") in pairs and pairs[("I", "Y")] == 20
    assert ("Q", "N") in pairs and pairs[("Q", "N")] == 20
    assert ("S", "G") in pairs and pairs[("S", "G")] == 20
    assert ("K", "V") in pairs and pairs[("K", "V")] == 12
    assert ("K", "T") in pairs and pairs[("K", "T")] == 8
    assert ("W", "N") in pairs and pairs[("W", "N")] == 16


def test_confidence_calibration(evaluator):
    """
    Verify model confidence is well-calibrated:
    Average confidence on correct predictions is significantly higher than on errors.
    """
    report = evaluator.evaluate()
    assert report.mean_confidence_correct > 0.70  # Typically ~0.85
    assert report.mean_confidence_errors < 0.45   # Typically ~0.27
    assert report.mean_confidence_correct > report.mean_confidence_errors * 2.0


def test_feature_pipeline_consistency():
    """Verify feature extractor dimension matches classifier expectation."""
    clf = get_classifier()
    extractor = LandmarkFeatureExtractor()

    assert extractor.get_feature_dimension() == TOTAL_FEATURE_DIM
    assert TOTAL_FEATURE_DIM == 151
    if clf.is_loaded and clf.model is not None:
        assert clf.model.n_features_in_ == TOTAL_FEATURE_DIM


def test_report_serialization_safe(evaluator, tmp_path):
    """Verify reports can be serialized to disk without touching existing model files."""
    report = evaluator.evaluate()
    json_p, md_p = evaluator.save_report(report, output_dir=tmp_path)

    assert json_p.exists()
    assert md_p.exists()
    assert json_p.stat().st_size > 500
    assert md_p.stat().st_size > 500

    # Verify JSON content is valid
    import json
    with open(json_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["overall_accuracy"] == 0.8043
    assert len(data["classes"]) == 26
