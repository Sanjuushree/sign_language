"""
ISL Connect — Preprocessing package (Phase 4).

Provides dataset inspection, image preprocessing, and batch preparation
utilities for the ISL alphabet dataset.
"""

from src.preprocessing.image_processor import (
    BrightnessMethod,
    ImageProcessor,
    ProcessingResult,
    IlluminationInfo,
    calculate_luminance,
    assess_illumination,
    adaptive_light_correction,
)
from src.preprocessing.dataset_inspector import (
    DatasetInspector,
    DatasetReport,
    ClassReport,
    SUPPORTED_EXTENSIONS,
    ISL_ALPHABET,
)
from src.preprocessing.dataset_preparer import (
    DatasetPreparer,
    PreparationSummary,
)

__all__ = [
    # Image processing
    "BrightnessMethod",
    "ImageProcessor",
    "ProcessingResult",
    "IlluminationInfo",
    "calculate_luminance",
    "assess_illumination",
    "adaptive_light_correction",
    # Dataset inspection
    "DatasetInspector",
    "DatasetReport",
    "ClassReport",
    "SUPPORTED_EXTENSIONS",
    "ISL_ALPHABET",
    # Dataset preparation
    "DatasetPreparer",
    "PreparationSummary",
]
