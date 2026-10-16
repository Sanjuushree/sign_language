"""
ISL Connect — Model Training Pipeline (Phase 5).

Extracts landmark features from the RealSign dataset splits (Training & Testing),
caches extracted features to disk, trains an ensemble gesture classifier, evaluates
offline test metrics, and serializes the model and metadata into models/.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from src.vision.hand_detector import HandDetector, get_hand_detector
from src.recognition.feature_extractor import LandmarkFeatureExtractor, TOTAL_FEATURE_DIM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "dataset" / "processed" / "feature_cache"

# All 26 ISL fingerspelled alphabet letters
ISL_CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


class ISLModelTrainer:
    """End-to-end trainer for Indian Sign Language gesture classification."""

    def __init__(
        self,
        hand_detector: Optional[HandDetector] = None,
        feature_extractor: Optional[LandmarkFeatureExtractor] = None,
        models_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.detector = hand_detector or get_hand_detector()
        self.extractor = feature_extractor or LandmarkFeatureExtractor()
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_features_from_directory(
        self,
        split_dir: Path,
        max_samples_per_class: int = 80,
        cache_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract landmark features for each class in a dataset split directory.

        Returns:
        --------
        X : np.ndarray of shape (N, 151)
        y : np.ndarray of shape (N,)
        classes : list of unique class names
        """
        split_p = Path(split_dir)
        cache_file = self.cache_dir / f"{cache_name}.npz" if cache_name else None

        if use_cache and cache_file and cache_file.exists():
            print(f"Loading cached features from {cache_file}...", flush=True)
            data = np.load(cache_file, allow_pickle=True)
            return data["X"], data["y"], list(data["classes"])

        print(f"Extracting features from {split_p} (max {max_samples_per_class} per class)...", flush=True)
        start_time = time.time()

        X_list: List[np.ndarray] = []
        y_list: List[str] = []

        class_dirs = [d for d in split_p.iterdir() if d.is_dir() and d.name.upper() in ISL_CLASSES]
        class_dirs.sort(key=lambda d: d.name.upper())

        total_classes = len(class_dirs)
        for c_idx, c_dir in enumerate(class_dirs):
            c_name = c_dir.name.upper()
            images = [
                p for p in c_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
            images.sort()
            sampled = images[:max_samples_per_class]

            valid_count = 0
            for img_path in sampled:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                res = self.detector.process_frame(img, is_bgr=True)
                if res.hands_detected:
                    feat = self.extractor.extract_features(res.landmarks)
                    X_list.append(feat)
                    y_list.append(c_name)
                    valid_count += 1

            print(f"  [{c_idx + 1}/{total_classes}] Class '{c_name}': {valid_count}/{len(sampled)} valid hand detections", flush=True)

        if not X_list:
            raise ValueError(f"No valid hand features extracted from {split_p}!")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        classes = sorted(list(set(y_list)))

        elapsed = time.time() - start_time
        print(f"Feature extraction done in {elapsed:.1f}s: X shape = {X.shape}, classes = {len(classes)}", flush=True)

        if cache_file:
            print(f"Saving feature cache to {cache_file}...", flush=True)
            np.savez_compressed(cache_file, X=X, y=y, classes=classes)

        return X, y, classes

    def train_and_evaluate(
        self,
        raw_root: Path,
        max_train_samples: int = 80,
        max_test_samples: int = 25,
        model_name: str = "isl_classifier.joblib",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full training & evaluation loop using Training and Testing splits.

        Returns:
        --------
        dict of evaluation results, accuracy, and file paths.
        """
        raw_root = Path(raw_root)
        train_dir = raw_root / "Training"
        test_dir = raw_root / "Testing"

        if not train_dir.exists():
            raise FileNotFoundError(f"Training split not found: {train_dir}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Testing split not found: {test_dir}")

        train_cache_name = f"train_features_s{max_train_samples}"
        test_cache_name = f"test_features_s{max_test_samples}"

        # 1. Feature Extraction
        X_train, y_train, classes_train = self.extract_features_from_directory(
            train_dir,
            max_samples_per_class=max_train_samples,
            cache_name=train_cache_name,
            use_cache=use_cache,
        )

        X_test, y_test, classes_test = self.extract_features_from_directory(
            test_dir,
            max_samples_per_class=max_test_samples,
            cache_name=test_cache_name,
            use_cache=use_cache,
        )

        # 2. Train Random Forest Classifier
        print(f"Training Random Forest classifier on {len(X_train)} samples...", flush=True)
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        train_time = time.time() - t0

        # 3. Evaluate
        y_pred = rf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, zero_division=0)

        print("\n" + "=" * 60, flush=True)
        print(f"MODEL EVALUATION REPORT — Test Accuracy: {acc * 100:.2f}%", flush=True)
        print("=" * 60, flush=True)
        print(report_text, flush=True)

        # 4. Save Model Bundle
        model_out = self.models_dir / model_name
        model_bundle = {
            "model": rf,
            "classes": sorted(list(set(y_train))),
            "feature_dim": TOTAL_FEATURE_DIM,
            "model_type": "RandomForestClassifier",
            "accuracy": acc,
            "created_at": datetime.utcnow().isoformat(),
        }
        joblib.dump(model_bundle, model_out)
        print(f"Saved trained model to {model_out}", flush=True)

        # 5. Save Metadata
        meta_out = self.models_dir / "classifier_metadata.json"
        metadata = {
            "model_name": model_name,
            "model_type": "RandomForestClassifier",
            "feature_dim": TOTAL_FEATURE_DIM,
            "total_classes": len(classes_train),
            "classes": sorted(list(set(y_train))),
            "test_accuracy": round(acc, 4),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_time_seconds": round(train_time, 2),
            "created_at": datetime.utcnow().isoformat(),
            "classification_report": {
                k: v for k, v in report_dict.items() if isinstance(v, dict)
            },
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_out}", flush=True)

        return {
            "accuracy": acc,
            "model_path": str(model_out),
            "metadata_path": str(meta_out),
            "classes": classes_train,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }


if __name__ == "__main__":
    trainer = ISLModelTrainer()
    trainer.train_and_evaluate(PROJECT_ROOT / "dataset" / "raw")
