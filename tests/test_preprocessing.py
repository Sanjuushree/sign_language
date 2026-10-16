"""
ISL Connect — Preprocessing & Dataset Test Suite (Phase 4).

Tests:
  1. ImageProcessor: load/validate, resize, dimensions, brightness, denoise.
  2. DatasetInspector: missing dirs, empty dirs, class detection, imbalance.
  3. DatasetPreparer: batch processing, split preservation, skip-existing.
  4. Edge cases: invalid paths, corrupt images, empty arrays.

All tests use small synthetic images — no downloaded dataset required.
"""

import sys
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.image_processor import (
    BrightnessMethod,
    ImageProcessor,
    ProcessingResult,
)
from src.preprocessing.dataset_inspector import (
    DatasetInspector,
    DatasetReport,
    ClassReport,
    ISL_ALPHABET,
    SUPPORTED_EXTENSIONS,
)
from src.preprocessing.dataset_preparer import (
    DatasetPreparer,
    PreparationSummary,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: temporary directories with synthetic images
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a clean temporary directory."""
    return tmp_path


@pytest.fixture()
def synthetic_image_path(tmp_path: Path) -> Path:
    """Write a small synthetic 64×64 BGR image to disk and return its path."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    fpath = tmp_path / "test_image.jpg"
    cv2.imwrite(str(fpath), img)
    return fpath


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a flat A/B/C dataset with 3 synthetic images per class."""
    raw_root = tmp_path / "raw"
    for label in ("A", "B", "C"):
        class_dir = raw_root / label
        class_dir.mkdir(parents=True)
        for i in range(3):
            img = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_{i:03d}.jpg"), img)
    return raw_root


@pytest.fixture()
def synthetic_split_dataset(tmp_path: Path) -> Path:
    """Create a Train/Test split dataset with A/B classes."""
    raw_root = tmp_path / "raw_split"
    for split in ("Train", "Test"):
        for label in ("A", "B"):
            class_dir = raw_root / split / label
            class_dir.mkdir(parents=True)
            count = 4 if split == "Train" else 2
            for i in range(count):
                img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
                cv2.imwrite(str(class_dir / f"img_{i:03d}.png"), img)
    return raw_root


# ═══════════════════════════════════════════════════════════════════════════
# ImageProcessor Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestImageProcessorLoad:
    """Tests for loading and validating images."""

    def test_load_valid_image(self, synthetic_image_path: Path):
        proc = ImageProcessor()
        result = proc.load_image(synthetic_image_path)
        assert result.success is True
        assert result.image is not None
        assert result.image.shape == (64, 64, 3)

    def test_load_nonexistent_path(self, tmp_path: Path):
        proc = ImageProcessor()
        result = proc.load_image(tmp_path / "no_such_file.png")
        assert result.success is False
        assert "not found" in result.message.lower() or "does not exist" in result.error.lower()

    def test_load_directory_path(self, tmp_path: Path):
        proc = ImageProcessor()
        result = proc.load_image(tmp_path)
        assert result.success is False
        assert "not a" in result.error.lower() or "file" in result.error.lower()

    def test_load_corrupt_file(self, tmp_path: Path):
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"this is definitely not a jpeg")
        proc = ImageProcessor()
        result = proc.load_image(corrupt)
        assert result.success is False
        assert result.error is not None


class TestImageProcessorResize:
    """Tests for resizing behaviour."""

    def test_resize_square_no_aspect(self):
        proc = ImageProcessor(target_size=(64, 64), preserve_aspect=False)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = proc.resize(img)
        assert resized.shape == (64, 64, 3)

    def test_resize_square_with_aspect(self):
        proc = ImageProcessor(target_size=(64, 64), preserve_aspect=True)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = proc.resize(img)
        # Must be exactly 64×64 (padded)
        assert resized.shape == (64, 64, 3)

    def test_resize_rectangular_target(self):
        proc = ImageProcessor(target_size=(128, 64), preserve_aspect=False)
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        resized = proc.resize(img)
        assert resized.shape == (64, 128, 3)  # (height, width, channels)

    def test_resize_preserves_content(self):
        """A non-zero image should still have non-zero pixels after resize."""
        proc = ImageProcessor(target_size=(32, 32), preserve_aspect=False)
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        resized = proc.resize(img)
        assert np.any(resized > 0)


class TestImageProcessorDimensions:
    """Tests verifying output dimensions through the full pipeline."""

    @pytest.mark.parametrize(
        "target_size", [(64, 64), (128, 128), (100, 50)]
    )
    def test_process_output_dimensions(self, synthetic_image_path: Path, target_size):
        proc = ImageProcessor(target_size=target_size, preserve_aspect=False)
        result = proc.process(synthetic_image_path)
        assert result.success is True
        w, h = target_size
        assert result.image.shape[:2] == (h, w)

    def test_process_records_original_shape(self, synthetic_image_path: Path):
        proc = ImageProcessor(target_size=(32, 32))
        result = proc.process(synthetic_image_path)
        assert result.original_shape == (64, 64, 3)


class TestImageProcessorBrightness:
    """Tests for brightness enhancement methods."""

    def test_no_brightness_is_default(self):
        proc = ImageProcessor()
        assert proc.brightness_method == BrightnessMethod.NONE

    def test_clahe_modifies_image(self):
        proc = ImageProcessor(brightness_method=BrightnessMethod.CLAHE)
        img = np.random.randint(30, 100, (64, 64, 3), dtype=np.uint8)
        enhanced = proc.apply_clahe(img)
        assert enhanced.shape == img.shape
        # CLAHE should change at least some pixel values
        assert not np.array_equal(enhanced, img)

    def test_clahe_on_grayscale(self):
        proc = ImageProcessor(brightness_method=BrightnessMethod.CLAHE)
        gray = np.random.randint(30, 100, (64, 64), dtype=np.uint8)
        enhanced = proc.apply_clahe(gray)
        assert enhanced.shape == gray.shape

    def test_gamma_brighten(self):
        # gamma > 1.0 brightens: inv_gamma = 1/gamma < 1, raising mid-tone values
        proc = ImageProcessor(brightness_method=BrightnessMethod.GAMMA, gamma=2.0)
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        brightened = proc.apply_gamma(img)
        assert brightened.mean() > img.mean()

    def test_gamma_darken(self):
        # gamma < 1.0 darkens: inv_gamma = 1/gamma > 1, lowering mid-tone values
        proc = ImageProcessor(brightness_method=BrightnessMethod.GAMMA, gamma=0.5)
        img = np.full((32, 32, 3), 200, dtype=np.uint8)
        darkened = proc.apply_gamma(img)
        assert darkened.mean() < img.mean()

    def test_brightness_via_process(self, synthetic_image_path: Path):
        proc_none = ImageProcessor(brightness_method="none")
        proc_clahe = ImageProcessor(brightness_method="clahe")
        r1 = proc_none.process(synthetic_image_path)
        r2 = proc_clahe.process(synthetic_image_path)
        assert r1.success and r2.success
        # The two results should differ
        assert not np.array_equal(r1.image, r2.image)


class TestImageProcessorDenoise:
    """Tests for optional denoising."""

    def test_denoise_flag_off_by_default(self):
        proc = ImageProcessor()
        assert proc.denoise is False

    def test_denoise_produces_valid_output(self):
        proc = ImageProcessor(denoise=True, target_size=(32, 32))
        # Create a noisy image
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        denoised = proc.apply_denoise(img)
        assert denoised.shape == img.shape
        assert denoised.dtype == np.uint8


class TestImageProcessorArray:
    """Tests for processing in-memory arrays."""

    def test_process_array_valid(self):
        proc = ImageProcessor(target_size=(32, 32))
        img = np.random.randint(0, 256, (80, 60, 3), dtype=np.uint8)
        result = proc.process_image_array(img)
        assert result.success is True
        assert result.processed_shape is not None

    def test_process_array_empty(self):
        proc = ImageProcessor()
        result = proc.process_image_array(np.array([]))
        assert result.success is False

    def test_process_array_none(self):
        proc = ImageProcessor()
        result = proc.process_image_array(None)
        assert result.success is False


# ═══════════════════════════════════════════════════════════════════════════
# DatasetInspector Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetInspector:
    """Tests for dataset inspection."""

    def test_inspect_nonexistent_directory(self, tmp_path: Path):
        inspector = DatasetInspector(tmp_path / "nonexistent")
        report = inspector.inspect()
        assert report.dataset_found is False
        assert len(report.errors) > 0

    def test_inspect_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty_ds"
        empty.mkdir()
        inspector = DatasetInspector(empty)
        report = inspector.inspect()
        assert report.dataset_found is False

    def test_inspect_flat_dataset(self, synthetic_dataset: Path):
        inspector = DatasetInspector(synthetic_dataset)
        report = inspector.inspect()
        assert report.dataset_found is True
        assert report.total_classes == 3
        assert report.total_images == 9  # 3 classes × 3 images
        assert len(report.splits_detected) == 0

    def test_inspect_split_dataset(self, synthetic_split_dataset: Path):
        inspector = DatasetInspector(synthetic_split_dataset)
        report = inspector.inspect()
        assert report.dataset_found is True
        assert "Train" in report.splits_detected
        assert "Test" in report.splits_detected
        assert report.total_images == 12  # 2 splits × 2 classes × (4+2 avg) = 12

    def test_inspect_reports_missing_classes(self, synthetic_dataset: Path):
        """Only A, B, C exist — 23 classes should be missing."""
        inspector = DatasetInspector(synthetic_dataset)
        report = inspector.inspect()
        assert len(report.missing_classes) == 23  # 26 - 3

    def test_inspect_detects_unreadable_files(self, tmp_path: Path):
        class_dir = tmp_path / "X"
        class_dir.mkdir()
        # Write a corrupt "image"
        (class_dir / "corrupt.jpg").write_bytes(b"not-a-jpeg")
        # Write a valid image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "valid.jpg"), img)

        inspector = DatasetInspector(tmp_path)
        report = inspector.inspect()
        assert report.total_images == 1
        assert report.unreadable_total == 1

    def test_print_report_returns_string(self, synthetic_dataset: Path):
        inspector = DatasetInspector(synthetic_dataset)
        report = inspector.inspect()
        text = DatasetInspector.print_report(report)
        assert isinstance(text, str)
        assert "Dataset Inspection Report" in text

    def test_inspect_imbalance_detection(self, tmp_path: Path):
        """Create classes with very uneven image counts."""
        root = tmp_path / "imbalanced"
        # Class A: 10 images, Class B: 1 image
        for label, count in [("A", 10), ("B", 1)]:
            d = root / label
            d.mkdir(parents=True)
            for i in range(count):
                img = np.zeros((10, 10, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"img_{i}.jpg"), img)

        inspector = DatasetInspector(root)
        report = inspector.inspect()
        assert len(report.imbalance_warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════
# DatasetPreparer Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetPreparer:
    """Tests for batch dataset preparation."""

    def test_prepare_flat_dataset(self, synthetic_dataset: Path, tmp_path: Path):
        out = tmp_path / "processed"
        preparer = DatasetPreparer(
            raw_root=synthetic_dataset,
            processed_root=out,
            target_size=(32, 32),
        )
        summary = preparer.prepare_all()
        assert summary.total_processed == 9
        assert summary.total_errors == 0
        # Verify output files exist
        for label in ("A", "B", "C"):
            assert (out / label).exists()
            assert len(list((out / label).iterdir())) == 3

    def test_prepare_split_dataset(self, synthetic_split_dataset: Path, tmp_path: Path):
        out = tmp_path / "processed_split"
        preparer = DatasetPreparer(
            raw_root=synthetic_split_dataset,
            processed_root=out,
            target_size=(32, 32),
        )
        summary = preparer.prepare_all()
        assert summary.total_processed == 12
        assert (out / "Train" / "A").exists()
        assert (out / "Test" / "B").exists()

    def test_prepare_skips_existing(self, synthetic_dataset: Path, tmp_path: Path):
        out = tmp_path / "processed_skip"
        preparer = DatasetPreparer(
            raw_root=synthetic_dataset,
            processed_root=out,
            target_size=(32, 32),
        )
        # First run
        s1 = preparer.prepare_all(overwrite=False)
        assert s1.total_processed == 9
        # Second run — should skip all
        s2 = preparer.prepare_all(overwrite=False)
        assert s2.total_skipped == 9
        assert s2.total_processed == 0

    def test_prepare_missing_raw_root(self, tmp_path: Path):
        preparer = DatasetPreparer(
            raw_root=tmp_path / "nope",
            processed_root=tmp_path / "out",
        )
        summary = preparer.prepare_all()
        assert len(summary.errors) > 0

    def test_prepare_with_brightness(self, synthetic_dataset: Path, tmp_path: Path):
        out = tmp_path / "processed_bright"
        preparer = DatasetPreparer(
            raw_root=synthetic_dataset,
            processed_root=out,
            target_size=(32, 32),
            brightness_method="clahe",
        )
        summary = preparer.prepare_all()
        assert summary.total_processed == 9
        assert summary.total_errors == 0

    def test_splitting_guidance_is_nonempty(self):
        text = DatasetPreparer.splitting_guidance()
        assert len(text) > 100
        assert "data leakage" in text.lower() or "duplicate" in text.lower()

    def test_prepare_never_touches_raw(self, synthetic_dataset: Path, tmp_path: Path):
        """Verify raw images are NOT modified after preparation."""
        # Record original file sizes
        original_sizes = {}
        for fpath in synthetic_dataset.rglob("*.jpg"):
            original_sizes[str(fpath)] = fpath.stat().st_size

        out = tmp_path / "processed_safe"
        preparer = DatasetPreparer(
            raw_root=synthetic_dataset,
            processed_root=out,
            target_size=(32, 32),
        )
        preparer.prepare_all()

        # Check raw files are untouched
        for fpath_str, orig_size in original_sizes.items():
            assert Path(fpath_str).stat().st_size == orig_size


# ═══════════════════════════════════════════════════════════════════════════
# Automatic Low-Light & Adaptive Correction Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveIlluminationCorrection:
    """Tests verifying automatic detection and conservative adaptive correction
    across dark, normally lit, and bright/overexposed images.
    """

    def test_calculate_luminance_dark_normal_bright(self):
        from src.preprocessing.image_processor import calculate_luminance

        dark = np.full((50, 50, 3), 30, dtype=np.uint8)
        normal = np.full((50, 50, 3), 125, dtype=np.uint8)
        bright = np.full((50, 50, 3), 220, dtype=np.uint8)

        assert calculate_luminance(dark) < 50.0
        assert 110.0 <= calculate_luminance(normal) <= 140.0
        assert calculate_luminance(bright) > 210.0

    def test_assess_illumination_categories(self):
        from src.preprocessing.image_processor import assess_illumination

        dark = np.full((40, 40, 3), 25, dtype=np.uint8)
        normal = np.full((40, 40, 3), 115, dtype=np.uint8)
        bright = np.full((40, 40, 3), 230, dtype=np.uint8)

        info_dark = assess_illumination(dark, low_light_threshold=75.0)
        info_normal = assess_illumination(normal, low_light_threshold=75.0)
        info_bright = assess_illumination(bright, low_light_threshold=75.0, overexposed_threshold=210.0)

        assert info_dark.is_low_light is True
        assert info_dark.is_overexposed is False
        assert info_dark.recommended_gamma > 1.2

        assert info_normal.is_low_light is False
        assert info_normal.is_overexposed is False
        assert info_normal.recommended_gamma == 1.0

        assert info_bright.is_low_light is False
        assert info_bright.is_overexposed is True
        assert info_bright.recommended_gamma == 1.0

    def test_adaptive_light_correction_dark_image(self):
        from src.preprocessing.image_processor import adaptive_light_correction

        dark = np.full((64, 64, 3), 35, dtype=np.uint8)
        corrected, was_corrected, info = adaptive_light_correction(dark, low_light_threshold=75.0)

        assert was_corrected is True
        assert info.is_low_light is True
        assert corrected.shape == dark.shape
        # Corrected image should have higher mean luminance than dark original
        assert np.mean(corrected) > np.mean(dark)

    def test_adaptive_light_correction_normal_image_untouched(self):
        from src.preprocessing.image_processor import adaptive_light_correction

        normal = np.full((64, 64, 3), 120, dtype=np.uint8)
        corrected, was_corrected, info = adaptive_light_correction(normal, low_light_threshold=75.0)

        assert was_corrected is False
        assert info.is_low_light is False
        # Normal image MUST be strictly untouched (zero modifications)
        assert np.array_equal(corrected, normal)

    def test_adaptive_light_correction_bright_image_untouched(self):
        from src.preprocessing.image_processor import adaptive_light_correction

        bright = np.full((64, 64, 3), 225, dtype=np.uint8)
        corrected, was_corrected, info = adaptive_light_correction(bright, low_light_threshold=75.0)

        assert was_corrected is False
        # Overexposed/bright image MUST be strictly untouched
        assert np.array_equal(corrected, bright)

    def test_adaptive_preserves_lab_chroma_channels(self):
        """Verify A and B chrominance channels are preserved to prevent skin tone distortion."""
        from src.preprocessing.image_processor import adaptive_light_correction

        # Create image with specific color tint
        colored_dark = np.zeros((40, 40, 3), dtype=np.uint8)
        colored_dark[:, :, 0] = 50  # Blue
        colored_dark[:, :, 1] = 20  # Green
        colored_dark[:, :, 2] = 40  # Red

        corrected, was_corrected, _ = adaptive_light_correction(
            colored_dark, is_bgr=True, low_light_threshold=75.0
        )
        assert was_corrected is True

        lab_orig = cv2.cvtColor(colored_dark, cv2.COLOR_BGR2LAB)
        lab_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)

        # A and B channels must match closely (allowing small rounding differences from color conversions)
        assert np.allclose(lab_orig[:, :, 1], lab_corr[:, :, 1], atol=1.0)
        assert np.allclose(lab_orig[:, :, 2], lab_corr[:, :, 2], atol=1.0)

    def test_image_processor_adaptive_method(self):
        from src.preprocessing.image_processor import BrightnessMethod, ImageProcessor

        proc = ImageProcessor(brightness_method=BrightnessMethod.ADAPTIVE)
        dark_img = np.full((48, 48, 3), 30, dtype=np.uint8)
        res_dark = proc.process_image_array(dark_img)
        assert res_dark.success is True
        assert res_dark.is_low_light is True
        assert res_dark.brightness_corrected is True
        assert np.mean(res_dark.image) > np.mean(dark_img)

        normal_img = np.full((48, 48, 3), 130, dtype=np.uint8)
        res_norm = proc.process_image_array(normal_img)
        assert res_norm.success is True
        assert res_norm.is_low_light is False
        assert res_norm.brightness_corrected is False
