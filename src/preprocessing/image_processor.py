"""
ISL Connect — OpenCV Image Preprocessor (Phase 4).

Reusable module for loading, validating, resizing, colour-space conversion,
denoising, and configurable brightness enhancement of sign language images.

Usage:
    from src.preprocessing.image_processor import ImageProcessor

    processor = ImageProcessor(target_size=(128, 128))
    result = processor.process("dataset/raw/A/img001.jpg")
    if result.success:
        cv2.imwrite("dataset/processed/A/img001.jpg", result.image)

All functions return structured results and never raise exceptions to the
caller; errors are captured in the result object.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class BrightnessMethod(str, Enum):
    """Supported brightness enhancement algorithms."""

    NONE = "none"
    CLAHE = "clahe"
    GAMMA = "gamma"
    ADAPTIVE = "adaptive"  # Automatic low-light detection & conservative correction


@dataclass
class IlluminationInfo:
    """Lighting metrics and low-light detection state for an image frame."""

    mean_luminance: float       # Scene mean luminance (0.0 to 255.0)
    is_low_light: bool         # True if mean_luminance < low_light_threshold
    is_overexposed: bool       # True if mean_luminance > overexposed_threshold
    recommended_gamma: float   # Computed conservative gamma (1.0 to 1.6)


@dataclass
class ProcessingResult:
    """Structured result from ImageProcessor.process()."""

    success: bool
    image: Optional[np.ndarray] = None          # Processed image (BGR uint8)
    original_shape: Optional[Tuple[int, ...]] = None
    processed_shape: Optional[Tuple[int, ...]] = None
    message: str = ""
    error: Optional[str] = None
    is_low_light: bool = False
    brightness_corrected: bool = False
    mean_luminance: float = 0.0


# ---------------------------------------------------------------------------
# Standalone Illumination & Adaptive Correction Utilities
# ---------------------------------------------------------------------------

def calculate_luminance(image: np.ndarray, is_bgr: bool = True) -> float:
    """Calculate mean perceptual scene luminance (0.0 to 255.0).
    
    Uses standard perceptual weights via OpenCV grayscale conversion.
    """
    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 2:
        return float(np.mean(image))

    gray_code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
    gray = cv2.cvtColor(image, gray_code)
    return float(np.mean(gray))


def assess_illumination(
    image: np.ndarray,
    is_bgr: bool = True,
    low_light_threshold: float = 75.0,
    overexposed_threshold: float = 210.0,
) -> IlluminationInfo:
    """Assess lighting conditions of an image frame and determine whether correction is needed."""
    lum = calculate_luminance(image, is_bgr=is_bgr)
    is_low = lum < low_light_threshold
    is_over = lum > overexposed_threshold

    if is_low and low_light_threshold > 0:
        deficit = (low_light_threshold - lum) / low_light_threshold
        # Conservative gamma: scales smoothly from 1.0 (at threshold) up to 1.6 (pitch black)
        gamma = 1.0 + min(0.6, deficit * 0.6)
    else:
        gamma = 1.0

    return IlluminationInfo(
        mean_luminance=round(lum, 2),
        is_low_light=is_low,
        is_overexposed=is_over,
        recommended_gamma=round(gamma, 3),
    )


def adaptive_light_correction(
    image: np.ndarray,
    is_bgr: bool = True,
    low_light_threshold: float = 75.0,
    clip_limit: float = 1.8,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Tuple[np.ndarray, bool, IlluminationInfo]:
    """Automatically detect low-light conditions and apply conservative adaptive correction.

    Preserves well-lit and overexposed frames completely untouched.
    When low-light is detected:
    - Preserves chromaticity (A and B channels in LAB space are unchanged).
    - Applies gentle adaptive gamma and CLAHE to the L channel only.

    Returns:
    --------
    (corrected_image, was_corrected, illumination_info)
    """
    if image is None or image.size == 0:
        info = IlluminationInfo(0.0, False, False, 1.0)
        return image, False, info

    info = assess_illumination(image, is_bgr=is_bgr, low_light_threshold=low_light_threshold)

    # If not in low-light, return the original image without any modifications
    if not info.is_low_light:
        return image, False, info

    # Single channel / grayscale handling
    if len(image.shape) == 2:
        inv_gamma = 1.0 / max(info.recommended_gamma, 1e-6)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        gamma_corrected = cv2.LUT(image, table)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gamma_corrected)
        return enhanced, True, info

    # 3-channel BGR / RGB handling
    lab_code = cv2.COLOR_BGR2LAB if is_bgr else cv2.COLOR_RGB2LAB
    inv_lab_code = cv2.COLOR_LAB2BGR if is_bgr else cv2.COLOR_LAB2RGB

    lab = cv2.cvtColor(image, lab_code)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 1. Apply gentle adaptive gamma correction to L channel
    inv_gamma = 1.0 / max(info.recommended_gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    l_gamma = cv2.LUT(l_channel, table)

    # 2. Apply gentle CLAHE to lift localized edge contrast and finger contours
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_gamma)

    # 3. Merge back with untouched A and B channels (preserves original skin tones and color balance)
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    corrected_image = cv2.cvtColor(enhanced_lab, inv_lab_code)

    return corrected_image, True, info


class ImageProcessor:
    """Configurable image preprocessing pipeline for ISL dataset preparation.

    Parameters
    ----------
    target_size : tuple[int, int]
        (width, height) of the output image.  Default ``(128, 128)``.
    preserve_aspect : bool
        If True, resize the image to fit within *target_size* while keeping its
        aspect ratio, then pad the remaining space with *pad_colour*.
    pad_colour : tuple[int, int, int]
        BGR colour used for aspect-ratio padding.  Default black ``(0, 0, 0)``.
    denoise : bool
        If True, apply ``cv2.fastNlMeansDenoisingColored`` after loading.
    brightness_method : BrightnessMethod
        Brightness enhancement strategy.  ``NONE`` skips enhancement.
    clahe_clip_limit : float
        CLAHE clip limit when ``brightness_method`` is ``CLAHE``.
    clahe_tile_grid_size : tuple[int, int]
        CLAHE tile grid size.
    gamma : float
        Gamma value when ``brightness_method`` is ``GAMMA``.
        Values > 1.0 brighten; values < 1.0 darken.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        preserve_aspect: bool = True,
        pad_colour: Tuple[int, int, int] = (0, 0, 0),
        denoise: bool = False,
        brightness_method: Union[BrightnessMethod, str] = BrightnessMethod.NONE,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 1.0,
        low_light_threshold: float = 75.0,
    ):
        self.target_size = target_size  # (width, height)
        self.preserve_aspect = preserve_aspect
        self.pad_colour = pad_colour
        self.denoise = denoise
        self.brightness_method = BrightnessMethod(brightness_method)
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.gamma = gamma
        self.low_light_threshold = low_light_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, path: Union[str, Path]) -> ProcessingResult:
        """Load an image from disk, validating it is a readable image file.

        Returns a ProcessingResult with the raw BGR image or an error.
        """
        path = Path(path)
        if not path.exists():
            return ProcessingResult(
                success=False,
                error=f"File does not exist: {path}",
                message="File not found.",
            )
        if not path.is_file():
            return ProcessingResult(
                success=False,
                error=f"Path is not a file: {path}",
                message="Not a regular file.",
            )

        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception as exc:
            return ProcessingResult(
                success=False,
                error=f"OpenCV read error: {exc}",
                message="Failed to read image.",
            )

        if img is None or img.size == 0:
            return ProcessingResult(
                success=False,
                error=f"Unreadable or corrupt image: {path.name}",
                message="Image could not be decoded by OpenCV.",
            )

        return ProcessingResult(
            success=True,
            image=img,
            original_shape=img.shape,
            message="Image loaded successfully.",
        )

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize *image* to ``self.target_size``, optionally preserving aspect ratio."""
        target_w, target_h = self.target_size
        h, w = image.shape[:2]

        if not self.preserve_aspect:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Scale to fit within target while keeping aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exact target size
        canvas = np.full((target_h, target_w, 3), self.pad_colour, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        return canvas

    def convert_colour(self, image: np.ndarray, target: str = "BGR") -> np.ndarray:
        """Convert colour space.  Supported targets: 'BGR', 'RGB', 'GRAY'."""
        target = target.upper()
        if target == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target == "GRAY":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image  # Already BGR

    def apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local-means denoising (colour-aware)."""
        if len(image.shape) == 2 or image.shape[2] == 1:
            return cv2.fastNlMeansDenoising(image, None, 10)
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation) to the L
        channel of the image in LAB colour space."""
        if len(image.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction.  gamma > 1.0 brightens; gamma < 1.0 darkens."""
        inv_gamma = 1.0 / max(self.gamma, 1e-6)
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    def apply_adaptive(self, image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
        """Apply automatic adaptive low-light correction."""
        enhanced, _, _ = adaptive_light_correction(
            image,
            is_bgr=is_bgr,
            low_light_threshold=self.low_light_threshold,
            clip_limit=self.clahe_clip_limit,
            tile_grid_size=self.clahe_tile_grid_size,
        )
        return enhanced

    def apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured brightness enhancement method."""
        if self.brightness_method == BrightnessMethod.CLAHE:
            return self.apply_clahe(image)
        elif self.brightness_method == BrightnessMethod.GAMMA:
            return self.apply_gamma(image)
        elif self.brightness_method == BrightnessMethod.ADAPTIVE:
            return self.apply_adaptive(image)
        return image  # NONE — no enhancement

    def process(self, path: Union[str, Path]) -> ProcessingResult:
        """Run the full preprocessing pipeline on one image file.

        Pipeline order:
            1. Load & validate
            2. Optional denoising
            3. Optional brightness enhancement
            4. Resize (with optional aspect-ratio preservation)
        """
        load_result = self.load_image(path)
        if not load_result.success:
            return load_result

        img = load_result.image
        original_shape = img.shape
        illumination = assess_illumination(img, is_bgr=True, low_light_threshold=self.low_light_threshold)
        brightness_corrected = False

        try:
            if self.denoise:
                img = self.apply_denoise(img)

            if self.brightness_method == BrightnessMethod.ADAPTIVE:
                img, brightness_corrected, illumination = adaptive_light_correction(
                    img,
                    is_bgr=True,
                    low_light_threshold=self.low_light_threshold,
                    clip_limit=self.clahe_clip_limit,
                    tile_grid_size=self.clahe_tile_grid_size,
                )
            else:
                img = self.apply_brightness(img)
                brightness_corrected = self.brightness_method != BrightnessMethod.NONE

            img = self.resize(img)
        except Exception as exc:
            return ProcessingResult(
                success=False,
                original_shape=original_shape,
                error=f"Processing error: {exc}",
                message="An error occurred during image preprocessing.",
                is_low_light=illumination.is_low_light,
                mean_luminance=illumination.mean_luminance,
            )

        return ProcessingResult(
            success=True,
            image=img,
            original_shape=original_shape,
            processed_shape=img.shape,
            message="Image processed successfully.",
            is_low_light=illumination.is_low_light,
            brightness_corrected=brightness_corrected,
            mean_luminance=illumination.mean_luminance,
        )

    def process_image_array(
        self,
        image: np.ndarray,
    ) -> ProcessingResult:
        """Run the preprocessing pipeline on an in-memory numpy image (BGR)."""
        if image is None or image.size == 0:
            return ProcessingResult(
                success=False,
                error="Empty or None image array.",
                message="Invalid input image.",
            )

        original_shape = image.shape
        illumination = assess_illumination(image, is_bgr=True, low_light_threshold=self.low_light_threshold)
        brightness_corrected = False

        try:
            img = image.copy()
            if self.denoise:
                img = self.apply_denoise(img)

            if self.brightness_method == BrightnessMethod.ADAPTIVE:
                img, brightness_corrected, illumination = adaptive_light_correction(
                    img,
                    is_bgr=True,
                    low_light_threshold=self.low_light_threshold,
                    clip_limit=self.clahe_clip_limit,
                    tile_grid_size=self.clahe_tile_grid_size,
                )
            else:
                img = self.apply_brightness(img)
                brightness_corrected = self.brightness_method != BrightnessMethod.NONE

            img = self.resize(img)
        except Exception as exc:
            return ProcessingResult(
                success=False,
                original_shape=original_shape,
                error=f"Processing error: {exc}",
                message="An error occurred during image preprocessing.",
                is_low_light=illumination.is_low_light,
                mean_luminance=illumination.mean_luminance,
            )

        return ProcessingResult(
            success=True,
            image=img,
            original_shape=original_shape,
            processed_shape=img.shape,
            message="Image processed successfully.",
            is_low_light=illumination.is_low_light,
            brightness_corrected=brightness_corrected,
            mean_luminance=illumination.mean_luminance,
        )
