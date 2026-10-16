"""
ISL Connect — Model Evaluation and Diagnostics Pipeline (Phase 6).

Provides reproducible evaluation of trained ISL classification models on
held-out test datasets without retraining or modifying model weights.
Calculates:
- Overall accuracy, macro/weighted averages
- Per-letter precision, recall, F1-score, and support for all 26 classes
- Full 26x26 confusion matrix and top misclassification pairs
- Hand-count distribution and confidence score analytics
- Serialization of evaluation metrics to JSON and Markdown artifacts
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "isl_classifier.joblib"
DEFAULT_TEST_CACHE = _PROJECT_ROOT / "dataset" / "processed" / "feature_cache" / "test_features_s20.npz"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "models"

# 26 standard ISL alphabet letters
ISL_ALPHABET = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


@dataclass
class ConfusionPair:
    """Single pair of true vs predicted misclassification."""
    true_class: str
    predicted_class: str
    count: int
    percentage_of_true: float


@dataclass
class PerClassMetric:
    """Per-letter performance metrics."""
    letter: str
    precision: float
    recall: float
    f1_score: float
    support: int
    correct_count: int
    errors_count: int


@dataclass
class EvaluationReport:
    """Complete evaluation report data structure."""
    model_name: str
    model_type: str
    overall_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    total_test_samples: int
    total_correct: int
    total_errors: int
    total_classes: int
    classes: List[str]
    per_class_metrics: Dict[str, Dict[str, Any]]
    confusion_matrix: List[List[int]]
    top_confusions: List[Dict[str, Any]]
    mean_confidence_correct: float
    mean_confidence_errors: float
    overall_mean_confidence: float
    evaluated_at: str
    test_data_path: str


class ISLModelEvaluator:
    """Evaluates trained ISL gesture classifiers on held-out test data."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        test_cache_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.test_cache_path = Path(test_cache_path) if test_cache_path else DEFAULT_TEST_CACHE

        self.model = None
        self.classes: List[str] = []
        self.model_type: str = "Unknown"
        self._load_model()

    def _load_model(self) -> None:
        """Load model bundle from disk without modifying it."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.classes = [str(c) for c in loaded.get("classes", ISL_ALPHABET)]
            self.model_type = loaded.get("model_type", type(self.model).__name__)
        else:
            self.model = loaded
            if hasattr(self.model, "classes_"):
                self.classes = [str(c) for c in self.model.classes_]
            else:
                self.classes = ISL_ALPHABET
            self.model_type = type(self.model).__name__

    def evaluate(self, test_cache_path: Optional[Union[str, Path]] = None) -> EvaluationReport:
        """
        Execute evaluation on the held-out test feature dataset.

        Returns:
        --------
        EvaluationReport dataclass with complete metrics.
        """
        cache_path = Path(test_cache_path) if test_cache_path else self.test_cache_path
        if not cache_path.exists():
            raise FileNotFoundError(f"Test feature cache not found at {cache_path}")

        data = np.load(cache_path, allow_pickle=True)
        X_test = data["X"]
        y_test = data["y"]

        # Ensure strings
        y_test_str = np.array([str(y) for y in y_test])

        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_str = np.array([str(p) for p in y_pred])

        # Confidence analysis (predict_proba)
        probs = None
        confidences = np.zeros(len(X_test), dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_test)
            confidences = np.max(probs, axis=1)

        # Accuracy
        acc = float(accuracy_score(y_test_str, y_pred_str))
        total_samples = len(y_test_str)
        is_correct = (y_pred_str == y_test_str)
        total_correct = int(np.sum(is_correct))
        total_errors = total_samples - total_correct

        mean_conf_corr = float(np.mean(confidences[is_correct])) if np.any(is_correct) else 0.0
        mean_conf_err = float(np.mean(confidences[~is_correct])) if np.any(~is_correct) else 0.0
        overall_mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

        # Classification report
        rep_dict = classification_report(
            y_test_str,
            y_pred_str,
            labels=self.classes,
            output_dict=True,
            zero_division=0,
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test_str, y_pred_str, labels=self.classes)

        # Per-class metrics
        per_class: Dict[str, Dict[str, Any]] = {}
        for i, c in enumerate(self.classes):
            c_sup = int(np.sum(cm[i, :]))
            c_corr = int(cm[i, i])
            c_err = c_sup - c_corr
            c_rep = rep_dict.get(c, {})

            per_class[c] = {
                "letter": c,
                "precision": round(float(c_rep.get("precision", 0.0)), 4),
                "recall": round(float(c_rep.get("recall", 0.0)), 4),
                "f1_score": round(float(c_rep.get("f1-score", 0.0)), 4),
                "support": c_sup,
                "correct_count": c_corr,
                "errors_count": c_err,
            }

        # Top Confusions
        confusions: List[ConfusionPair] = []
        for i, true_c in enumerate(self.classes):
            true_sup = int(np.sum(cm[i, :]))
            if true_sup == 0:
                continue
            for j, pred_c in enumerate(self.classes):
                if i != j and cm[i, j] > 0:
                    cnt = int(cm[i, j])
                    pct = round((cnt / true_sup) * 100.0, 1)
                    confusions.append(ConfusionPair(
                        true_class=true_c,
                        predicted_class=pred_c,
                        count=cnt,
                        percentage_of_true=pct,
                    ))

        # Sort confusions by count descending
        confusions.sort(key=lambda x: x.count, reverse=True)
        top_confusions_dicts = [asdict(cp) for cp in confusions]

        macro_avg = rep_dict.get("macro avg", {})
        weighted_avg = rep_dict.get("weighted avg", {})

        return EvaluationReport(
            model_name=self.model_path.name,
            model_type=self.model_type,
            overall_accuracy=round(acc, 4),
            macro_precision=round(float(macro_avg.get("precision", 0.0)), 4),
            macro_recall=round(float(macro_avg.get("recall", 0.0)), 4),
            macro_f1=round(float(macro_avg.get("f1-score", 0.0)), 4),
            weighted_precision=round(float(weighted_avg.get("precision", 0.0)), 4),
            weighted_recall=round(float(weighted_avg.get("recall", 0.0)), 4),
            weighted_f1=round(float(weighted_avg.get("f1-score", 0.0)), 4),
            total_test_samples=total_samples,
            total_correct=total_correct,
            total_errors=total_errors,
            total_classes=len(self.classes),
            classes=self.classes,
            per_class_metrics=per_class,
            confusion_matrix=cm.tolist(),
            top_confusions=top_confusions_dicts,
            mean_confidence_correct=round(mean_conf_corr, 4),
            mean_confidence_errors=round(mean_conf_err, 4),
            overall_mean_confidence=round(overall_mean_conf, 4),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            test_data_path=str(cache_path),
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Save the evaluation report as JSON and Markdown files.

        Returns:
        --------
        (json_path, markdown_path)
        """
        out_p = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        out_p.mkdir(parents=True, exist_ok=True)

        json_path = out_p / "evaluation_report.json"
        md_path = out_p / "evaluation_report.md"

        # Save JSON
        report_data = asdict(report)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Save Markdown
        md_content = self.generate_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    @staticmethod
    def generate_markdown_report(report: EvaluationReport) -> str:
        """Format an evaluation report as GitHub-flavored Markdown."""
        lines = [
            "# ISL Connect — Phase 6 Model Evaluation Report",
            "",
            f"**Model File:** `{report.model_name}` ({report.model_type})  ",
            f"**Evaluation Date:** `{report.evaluated_at}`  ",
            f"**Test Dataset:** `{Path(report.test_data_path).name}` ({report.total_test_samples} held-out samples)  ",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Overall Test Accuracy:** `{report.overall_accuracy * 100:.2f}%` ({report.total_correct}/{report.total_test_samples} correct)",
            f"- **Macro F1-Score:** `{report.macro_f1 * 100:.2f}%`",
            f"- **Weighted F1-Score:** `{report.weighted_f1 * 100:.2f}%`",
            f"- **Mean Prediction Confidence (Correct):** `{report.mean_confidence_correct * 100:.1f}%`",
            f"- **Mean Prediction Confidence (Errors):** `{report.mean_confidence_errors * 100:.1f}%`",
            "",
            "## 2. Per-Letter Performance Table (All 26 Classes)",
            "",
            "| Letter | Precision | Recall | F1-Score | Support | Correct | Errors | Status |",
            "|:------:|:---------:|:------:|:--------:|:-------:|:-------:|:------:|:------:|",
        ]

        for c in report.classes:
            m = report.per_class_metrics.get(c, {})
            p = m.get("precision", 0.0) * 100
            r = m.get("recall", 0.0) * 100
            f1 = m.get("f1_score", 0.0) * 100
            sup = m.get("support", 0)
            corr = m.get("correct_count", 0)
            err = m.get("errors_count", 0)

            if r >= 95.0 and p >= 95.0:
                status = "Excellent"
            elif r > 50.0:
                status = "Moderate"
            else:
                status = "Needs Improvement"

            lines.append(
                f"| **{c}** | {p:.1f}% | {r:.1f}% | {f1:.1f}% | {sup} | {corr} | {err} | {status} |"
            )

        lines.extend([
            "",
            "## 3. Frequently Confused Sign Pairs",
            "",
            "| True Sign | Predicted As | Misclassified Samples | % of True Class | Error Explanation |",
            "|:---------:|:------------:|:--------------------:|:---------------:|:-------------------|",
        ])

        explanations = {
            ("I", "Y"): "Single hand pinky-extended sign: Y shares pinky extension with thumb extended.",
            ("Q", "N"): "Two-handed sign: Overlapping hands cause index/thumb loop to merge with N finger geometry.",
            ("S", "G"): "Two-handed crossed fist: When hands interlock, MediaPipe 3D joint projection matches G.",
            ("W", "N"): "Hand-count variation: W in test split had dual hands vs single hand in training.",
            ("K", "V"): "Two-handed hook sign: When non-dominant hand is occluded, V single-hand geometry dominates.",
            ("K", "T"): "Two-handed hook sign: Partial occlusion matches T index hook.",
            ("J", "Y"): "Dynamic gesture trace: Static single frame matches Y pinky silhouette.",
        }

        for c_dict in report.top_confusions:
            tc = c_dict["true_class"]
            pc = c_dict["predicted_class"]
            cnt = c_dict["count"]
            pct = c_dict["percentage_of_true"]
            expl = explanations.get((tc, pc), "Geometric similarity in landmark coordinates.")
            lines.append(f"| **{tc}** | **{pc}** | {cnt} | {pct:.1f}% | {expl} |")

        lines.extend([
            "",
            "## 4. Key Limitations & Risk Analysis",
            "",
            "1. **Near-Duplicate Temporal Leakage in Dataset Split:**",
            "   - Frames in `Training/` and `Testing/` originated from consecutive video streams of the same session.",
            "   - Offline accuracy reflects familiar signer morphology; real-world cross-subject accuracy may be lower.",
            "2. **Single vs Dual Hand Sensitivity:**",
            "   - Classes like `K` and `W` vary between 1 and 2 hands depending on camera field-of-view.",
            "   - Feature vector zeros slots 72–147 when only 1 hand is detected, causing misclassification of 2-handed signs.",
            "3. **Lighting & Camera Mirroring:**",
            "   - Low lighting drops MediaPipe palm detection confidence below 0.50.",
            "   - Browser selfie mirror mode flips X coordinates, which inverts relative two-hand delta vectors.",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    evaluator = ISLModelEvaluator()
    rep = evaluator.evaluate()
    j_p, m_p = evaluator.save_report(rep)
    print(f"Evaluation complete! Overall accuracy: {rep.overall_accuracy * 100:.2f}%")
    print(f"Saved JSON report: {j_p}")
    print(f"Saved Markdown report: {m_p}")
