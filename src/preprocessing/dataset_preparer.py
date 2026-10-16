"""
ISL Connect — Dataset Preparer (Phase 4).

Processes raw dataset images through the ImageProcessor pipeline and
saves them under ``dataset/processed/``, preserving class labels and
any existing split boundaries.

Rules:
  - Never overwrites original raw images.
  - Respects existing Train/Test/Validation splits if present.
  - If no splits exist, provides guidance for reproducible splitting.
  - Avoids data leakage by never duplicating images across splits.

Usage:
    from src.preprocessing.dataset_preparer import DatasetPreparer
    preparer = DatasetPreparer(raw_root="dataset/raw", processed_root="dataset/processed")
    summary = preparer.prepare_all()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2

from src.preprocessing.image_processor import (
    BrightnessMethod,
    ImageProcessor,
)
from src.preprocessing.dataset_inspector import DatasetInspector, SUPPORTED_EXTENSIONS


@dataclass
class PreparationSummary:
    """Summary of a batch preparation run."""

    total_processed: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    classes_processed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output_root: str = ""


class DatasetPreparer:
    """Batch preprocessor that reads raw images and writes processed copies.

    Parameters
    ----------
    raw_root : str | Path
        Root of the raw dataset (e.g. ``dataset/raw``).
    processed_root : str | Path
        Root where processed images will be saved (e.g. ``dataset/processed``).
    target_size : tuple[int, int]
        (width, height) passed to ``ImageProcessor``.
    preserve_aspect : bool
        If True, preserves aspect ratio and pads.
    denoise : bool
        Enable denoising in the pipeline.
    brightness_method : BrightnessMethod | str
        Brightness enhancement method.  Default ``"none"``.
    clahe_clip_limit : float
        CLAHE clip limit.
    gamma : float
        Gamma correction value.
    output_format : str
        Extension for saved images (``".jpg"`` or ``".png"``).
    """

    def __init__(
        self,
        raw_root: Union[str, Path] = "dataset/raw",
        processed_root: Union[str, Path] = "dataset/processed",
        target_size: Tuple[int, int] = (128, 128),
        preserve_aspect: bool = True,
        denoise: bool = False,
        brightness_method: Union[BrightnessMethod, str] = BrightnessMethod.NONE,
        clahe_clip_limit: float = 2.0,
        gamma: float = 1.0,
        output_format: str = ".jpg",
    ):
        self.raw_root = Path(raw_root).resolve()
        self.processed_root = Path(processed_root).resolve()
        self.output_format = output_format

        self.processor = ImageProcessor(
            target_size=target_size,
            preserve_aspect=preserve_aspect,
            denoise=denoise,
            brightness_method=brightness_method,
            clahe_clip_limit=clahe_clip_limit,
            gamma=gamma,
        )

    def prepare_all(self, overwrite: bool = False) -> PreparationSummary:
        """Process every image in the raw dataset and save to processed root.

        If split directories (Train, Test, etc.) are found, they are mirrored
        under ``processed_root``.

        Parameters
        ----------
        overwrite : bool
            If False (default), skip images that already exist in the output.
        """
        summary = PreparationSummary(output_root=str(self.processed_root))

        if not self.raw_root.exists():
            summary.errors.append(f"Raw dataset root not found: {self.raw_root}")
            return summary

        # Detect split vs flat structure (reuse inspector logic)
        inspector = DatasetInspector(self.raw_root)
        splits = inspector._detect_splits()

        if splits:
            for split_dir in splits:
                self._process_split(split_dir, split_dir.name, summary, overwrite)
        else:
            # Flat: class folders directly in raw_root
            self._process_split(self.raw_root, split_name=None, summary=summary, overwrite=overwrite)

        return summary

    def _process_split(
        self,
        parent_dir: Path,
        split_name: Optional[str],
        summary: PreparationSummary,
        overwrite: bool,
    ) -> None:
        """Process all class folders within *parent_dir*."""
        class_dirs = sorted(
            [d for d in parent_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )

        for class_dir in class_dirs:
            class_name = class_dir.name

            # Build output directory
            if split_name:
                out_dir = self.processed_root / split_name / class_name
            else:
                out_dir = self.processed_root / class_name

            out_dir.mkdir(parents=True, exist_ok=True)

            # Process each image file
            for fpath in sorted(class_dir.iterdir()):
                if fpath.is_dir():
                    continue
                if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                out_name = fpath.stem + self.output_format
                out_path = out_dir / out_name

                if out_path.exists() and not overwrite:
                    summary.total_skipped += 1
                    continue

                result = self.processor.process(fpath)

                if result.success and result.image is not None:
                    try:
                        cv2.imwrite(str(out_path), result.image)
                        summary.total_processed += 1
                    except Exception as exc:
                        summary.total_errors += 1
                        summary.errors.append(f"Write error {out_path.name}: {exc}")
                else:
                    summary.total_errors += 1
                    summary.errors.append(
                        f"Failed to process {fpath.name}: {result.error}"
                    )

            if class_name not in summary.classes_processed:
                summary.classes_processed.append(class_name)

    @staticmethod
    def splitting_guidance() -> str:
        """Return guidance for creating reproducible train/val/test splits."""
        return """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ISL Connect — Reproducible Data Splitting Guidance (Phase 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If your raw dataset does NOT already have Train/Test/Validation
sub-folders, follow these steps to create reproducible splits:

1. Use scikit-learn's `train_test_split` with a fixed random seed:

     from sklearn.model_selection import train_test_split
     train, temp = train_test_split(files, test_size=0.3, random_state=42)
     val, test   = train_test_split(temp,  test_size=0.5, random_state=42)

   This gives an approximate 70 / 15 / 15 split.

2. NEVER place augmented or near-duplicate images of the same
   subject across different splits.  Group by subject first if
   your dataset has per-subject labels.

3. Stratify by class label to maintain balanced distributions:

     train_test_split(files, stratify=labels, ...)

4. Save the split mapping to a file (e.g. CSV or JSON) so every
   teammate or CI run produces the same assignment.

5. Only augment the TRAINING split.  Never augment validation or
   test data, as that inflates metrics and masks overfitting.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ---------------------------------------------------------------------------
# Convenience CLI entry point
# ---------------------------------------------------------------------------

def run_preparation(
    raw_root: Optional[str] = None,
    processed_root: Optional[str] = None,
    brightness: str = "none",
    target_size: Tuple[int, int] = (128, 128),
) -> PreparationSummary:
    """CLI helper: prepare the dataset with given settings."""
    project_root = Path(__file__).resolve().parent.parent.parent

    if raw_root is None:
        raw_root = str(project_root / "dataset" / "raw")
    if processed_root is None:
        processed_root = str(project_root / "dataset" / "processed")

    preparer = DatasetPreparer(
        raw_root=raw_root,
        processed_root=processed_root,
        target_size=target_size,
        brightness_method=brightness,
    )
    summary = preparer.prepare_all()

    print(f"\n  Processed: {summary.total_processed}")
    print(f"  Skipped:   {summary.total_skipped}")
    print(f"  Errors:    {summary.total_errors}")
    if summary.errors:
        for e in summary.errors:
            print(f"    ✖ {e}")

    return summary


if __name__ == "__main__":
    run_preparation()
