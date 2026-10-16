"""
ISL Connect — Dataset Inspector (Phase 4).

Scans the raw dataset directory and produces a structured report containing:
  - Class names discovered (expected: A–Z sub-folders).
  - Image count per class.
  - File format distribution.
  - Unreadable or corrupt files.
  - Class imbalance warnings.

Usage:
    from src.preprocessing.dataset_inspector import DatasetInspector
    inspector = DatasetInspector("dataset/raw")
    report = inspector.inspect()
    inspector.print_report(report)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# Image extensions we recognise as valid candidates
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Expected ISL alphabet classes (A–Z)
ISL_ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


@dataclass
class ClassReport:
    """Inspection result for a single class (sub-folder)."""

    class_name: str
    image_count: int = 0
    format_counts: Dict[str, int] = field(default_factory=dict)
    unreadable_files: List[str] = field(default_factory=list)
    non_image_files: List[str] = field(default_factory=list)
    sample_dimensions: Optional[Tuple[int, int]] = None  # (height, width) of first valid image


@dataclass
class DatasetReport:
    """Complete dataset inspection result."""

    root_path: str
    dataset_found: bool = False
    total_classes: int = 0
    total_images: int = 0
    class_reports: List[ClassReport] = field(default_factory=list)
    missing_classes: List[str] = field(default_factory=list)
    extra_classes: List[str] = field(default_factory=list)
    unreadable_total: int = 0
    format_distribution: Dict[str, int] = field(default_factory=dict)
    splits_detected: List[str] = field(default_factory=list)
    imbalance_warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DatasetInspector:
    """Scans a dataset directory and produces a structured DatasetReport."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).resolve()

    def inspect(self) -> DatasetReport:
        """
        Perform a full inspection of the dataset directory.

        Handles two common structures:
          1. Flat:  dataset/raw/A/, dataset/raw/B/, ...
          2. Split: dataset/raw/Train/A/, dataset/raw/Train/B/, ...
        """
        report = DatasetReport(root_path=str(self.dataset_root))

        if not self.dataset_root.exists():
            report.errors.append(f"Dataset root does not exist: {self.dataset_root}")
            return report

        if not self.dataset_root.is_dir():
            report.errors.append(f"Dataset root is not a directory: {self.dataset_root}")
            return report

        # Detect structure: check for split directories first
        split_dirs = self._detect_splits()
        if split_dirs:
            report.splits_detected = [s.name for s in split_dirs]
            report.dataset_found = True
            # Inspect each split
            for split_dir in split_dirs:
                self._inspect_class_folders(split_dir, report, split_name=split_dir.name)
        else:
            # Flat structure: class folders directly inside dataset_root
            class_dirs = self._find_class_folders(self.dataset_root)
            if class_dirs:
                report.dataset_found = True
                self._inspect_class_folders(self.dataset_root, report, split_name=None)
            else:
                report.errors.append(
                    f"No class sub-folders (A–Z) found in {self.dataset_root}. "
                    "Expected either: (a) sub-folders named A/ B/ ... Z/ directly, "
                    "or (b) split folders like Train/ Test/ containing A/ B/ ... Z/."
                )
                return report

        # Compute global stats
        all_class_names = set()
        for cr in report.class_reports:
            report.total_images += cr.image_count
            report.unreadable_total += len(cr.unreadable_files)
            all_class_names.add(cr.class_name)
            for fmt, cnt in cr.format_counts.items():
                report.format_distribution[fmt] = report.format_distribution.get(fmt, 0) + cnt

        report.total_classes = len(all_class_names)

        # Check missing / extra classes against ISL_ALPHABET
        expected = set(ISL_ALPHABET)
        report.missing_classes = sorted(expected - all_class_names)
        report.extra_classes = sorted(all_class_names - expected)

        # Imbalance detection
        self._check_imbalance(report)

        return report

    def _detect_splits(self) -> List[Path]:
        """Detect common split directory names (Train, Test, Validation, etc.)."""
        split_names = {"train", "training", "test", "testing", "val", "validation", "valid"}
        found = []
        for child in sorted(self.dataset_root.iterdir()):
            if child.is_dir() and child.name.lower() in split_names:
                found.append(child)
        return found

    def _find_class_folders(self, parent: Path) -> List[Path]:
        """Return sorted list of sub-directories that look like class folders."""
        if not parent.exists():
            return []
        return sorted(
            [d for d in parent.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )

    def _inspect_class_folders(
        self,
        parent_dir: Path,
        report: DatasetReport,
        split_name: Optional[str] = None,
    ) -> None:
        """Inspect each class sub-folder inside *parent_dir*."""
        class_dirs = self._find_class_folders(parent_dir)

        for class_dir in class_dirs:
            cr = self._inspect_single_class(class_dir)
            report.class_reports.append(cr)

    def _inspect_single_class(self, class_dir: Path) -> ClassReport:
        """Inspect all files in a single class folder."""
        cr = ClassReport(class_name=class_dir.name)

        for fpath in sorted(class_dir.iterdir()):
            if fpath.is_dir():
                continue  # skip nested dirs

            ext = fpath.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                cr.non_image_files.append(str(fpath.name))
                continue

            # Try reading the image to verify it is valid
            try:
                img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
                if img is None:
                    cr.unreadable_files.append(str(fpath.name))
                    continue
            except Exception:
                cr.unreadable_files.append(str(fpath.name))
                continue

            cr.image_count += 1
            cr.format_counts[ext] = cr.format_counts.get(ext, 0) + 1

            # Capture dimensions of the first valid image
            if cr.sample_dimensions is None:
                cr.sample_dimensions = (img.shape[0], img.shape[1])

        return cr

    def _check_imbalance(self, report: DatasetReport) -> None:
        """Flag classes whose image count deviates significantly from the mean."""
        counts = [cr.image_count for cr in report.class_reports if cr.image_count > 0]
        if len(counts) < 2:
            return

        mean_count = sum(counts) / len(counts)
        if mean_count == 0:
            return

        for cr in report.class_reports:
            if cr.image_count == 0:
                report.imbalance_warnings.append(
                    f"Class '{cr.class_name}' has 0 images."
                )
            elif cr.image_count < mean_count * 0.5:
                report.imbalance_warnings.append(
                    f"Class '{cr.class_name}' has only {cr.image_count} images "
                    f"(mean: {mean_count:.0f}). Under-represented."
                )
            elif cr.image_count > mean_count * 1.5:
                report.imbalance_warnings.append(
                    f"Class '{cr.class_name}' has {cr.image_count} images "
                    f"(mean: {mean_count:.0f}). Over-represented."
                )

    @staticmethod
    def print_report(report: DatasetReport) -> str:
        """Return a human-readable string summary of the inspection report."""
        lines: List[str] = []
        lines.append("=" * 65)
        lines.append("  ISL Connect — Dataset Inspection Report (Phase 4)")
        lines.append("=" * 65)
        lines.append(f"  Root:             {report.root_path}")
        lines.append(f"  Dataset Found:    {report.dataset_found}")
        lines.append(f"  Splits Detected:  {report.splits_detected or 'None (flat structure)'}")
        lines.append(f"  Total Classes:    {report.total_classes}")
        lines.append(f"  Total Images:     {report.total_images}")
        lines.append(f"  Unreadable Files: {report.unreadable_total}")
        lines.append(f"  Format Distribution: {report.format_distribution}")
        lines.append("")

        if report.missing_classes:
            lines.append(f"  ⚠ Missing ISL classes: {', '.join(report.missing_classes)}")
        if report.extra_classes:
            lines.append(f"  ℹ Extra classes:       {', '.join(report.extra_classes)}")
        lines.append("")

        # Per-class table
        lines.append(f"  {'Class':<10} {'Images':>8}  {'Formats':<25} {'Unreadable':>10}")
        lines.append("  " + "-" * 60)
        for cr in report.class_reports:
            fmt_str = ", ".join(f"{k}:{v}" for k, v in sorted(cr.format_counts.items()))
            lines.append(
                f"  {cr.class_name:<10} {cr.image_count:>8}  {fmt_str:<25} "
                f"{len(cr.unreadable_files):>10}"
            )
        lines.append("")

        if report.imbalance_warnings:
            lines.append("  ⚠ Imbalance Warnings:")
            for w in report.imbalance_warnings:
                lines.append(f"    - {w}")
            lines.append("")

        if report.errors:
            lines.append("  ✖ Errors:")
            for e in report.errors:
                lines.append(f"    - {e}")
            lines.append("")

        lines.append("=" * 65)

        text = "\n".join(lines)
        print(text)
        return text


# ---------------------------------------------------------------------------
# Convenience CLI entry point
# ---------------------------------------------------------------------------

def run_inspection(dataset_path: Optional[str] = None) -> DatasetReport:
    """Run dataset inspection and print report. Returns the report object."""
    from pathlib import Path as _P
    project_root = _P(__file__).resolve().parent.parent.parent

    if dataset_path is None:
        dataset_path = str(project_root / "dataset" / "raw")

    inspector = DatasetInspector(dataset_path)
    report = inspector.inspect()
    DatasetInspector.print_report(report)
    return report


if __name__ == "__main__":
    import sys as _sys
    path_arg = _sys.argv[1] if len(_sys.argv) > 1 else None
    run_inspection(path_arg)
