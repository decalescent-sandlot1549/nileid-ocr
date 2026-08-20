"""Image enhancement for OCR.

The original pipeline applied a fixed chain to every crop: 2.5x upscale,
CLAHE, unsharp mask, morphological close and non-local-means denoising.
That chain is expensive (``fastNlMeansDenoising`` dominates the runtime of
a field read) and actively harmful on clean input -- morphological closing
merges adjacent Arabic strokes, and denoising a sharp crop erodes thin
diacritics.

Here each step is applied only when a cheap measurement says the image
needs it. :func:`assess_quality` produces those measurements, and they are
also reported to the caller as warnings so a poor scan is visible rather
than silently degraded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nileid.logging_utils import get_logger

log = get_logger("preprocessing.enhance")

#: Variance of the Laplacian below which an image is considered blurred.
BLUR_THRESHOLD = 100.0
#: Standard deviation below which an image is considered low contrast.
CONTRAST_THRESHOLD = 45.0
#: Estimated noise sigma above which denoising is worth its cost.
NOISE_THRESHOLD = 6.0
#: Target height, in pixels, for a single line of text handed to OCR.
TARGET_LINE_HEIGHT = 64
#: Never upscale beyond this factor; past it, interpolation only invents
#: smooth edges and slows OCR down.
MAX_UPSCALE = 4.0


@dataclass(frozen=True)
class Quality:
    """Cheap, interpretable image-quality measurements."""

    blur: float
    contrast: float
    noise: float
    brightness: float

    @property
    def is_blurred(self) -> bool:
        return self.blur < BLUR_THRESHOLD

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < CONTRAST_THRESHOLD

    @property
    def is_noisy(self) -> bool:
        return self.noise > NOISE_THRESHOLD

    @property
    def is_dark(self) -> bool:
        return self.brightness < 70.0

    @property
    def is_washed_out(self) -> bool:
        return self.brightness > 200.0

    def warnings(self) -> list[str]:
        """Human-readable notes describing why a read may be unreliable."""
        notes = []
        if self.is_blurred:
            notes.append(f"image appears blurred (sharpness={self.blur:.0f})")
        if self.is_low_contrast:
            notes.append(f"low contrast (std={self.contrast:.0f})")
        if self.is_noisy:
            notes.append(f"noisy image (sigma={self.noise:.1f})")
        if self.is_dark:
            notes.append("image is underexposed")
        if self.is_washed_out:
            notes.append("image is overexposed")
        return notes


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _estimate_noise(gray: np.ndarray) -> float:
    """Estimate the noise standard deviation.

    Uses the fast Laplacian-convolution estimator of Immerkaer (1996): the
    absolute response of a noise-sensitive kernel, scaled so that a clean
    image tends to zero.
    """
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma = float(np.abs(response).sum()) / (36.0**0.5)
    return sigma * (np.pi**0.5) / (6.0 * (width - 2) * (height - 2)) * 6.0


def assess_quality(image: np.ndarray) -> Quality:
    """Measure blur, contrast, noise and brightness of an image."""
    gray = _to_gray(image)
    return Quality(
        blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        contrast=float(gray.std()),
        noise=float(_estimate_noise(gray)),
        brightness=float(gray.mean()),
    )


def _upscale_to_line_height(gray: np.ndarray, target: int = TARGET_LINE_HEIGHT) -> np.ndarray:
    """Scale a crop so a text line reaches the height OCR performs best at.

    A fixed multiplier (the original used 2.5x unconditionally) either
    starves OCR on small crops or wastes time on already-large ones.
    """
    height = gray.shape[0]
    if height <= 0:
        return gray
    factor = min(MAX_UPSCALE, max(1.0, target / float(height)))
    if factor <= 1.01:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _unsharp(gray: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    return cv2.addWeighted(gray, amount, blurred, 1.0 - amount, 0)


def enhance_for_text(image: np.ndarray, quality: Quality | None = None) -> np.ndarray:
    """Prepare a text crop for OCR, applying only the steps it needs.

    Returns a 3-channel image because both EasyOCR and PaddleOCR expect
    colour input.
    """
    try:
        gray = _to_gray(image)
        quality = quality or assess_quality(gray)

        gray = _upscale_to_line_height(gray)

        # Contrast: CLAHE equalises local illumination, which is what
        # shadows across a card actually produce. Applied only when the
        # crop is flat, dark or blown out.
        if quality.is_low_contrast or quality.is_dark or quality.is_washed_out:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Noise: bilateral filtering preserves glyph edges and costs a
        # fraction of non-local means, which the original chain ran on
        # every crop regardless of need.
        if quality.is_noisy:
            gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # Sharpness: only when the source is genuinely soft, and gently,
        # to avoid ringing that OCR reads as extra strokes.
        if quality.is_blurred:
            gray = _unsharp(gray, amount=1.5)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    except cv2.error:
        log.exception("enhance_for_text failed; using the unmodified crop.")
        return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def enhance_for_digits(image: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Prepare the National ID strip for the digit detector.

    The digits are printed small relative to the card, so the detector
    benefits from a fixed upscale and a mild sharpen. Kept separate from
    :func:`enhance_for_text` because a *detector* wants crisp edges rather
    than the smoothed, contrast-flattened input a *recogniser* prefers.
    """
    try:
        height, width = image.shape[:2]
        # Guard against allocating an enormous buffer on a large input.
        scale = min(scale, 4000.0 / max(width, 1), 4000.0 / max(height, 1))
        if scale > 1.01:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    except cv2.error:
        log.exception("enhance_for_digits failed; using the unmodified crop.")
        return image


def pad(
    image: np.ndarray, border: int = 16, value: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Add a quiet margin so OCR does not clip glyphs at the crop edge."""
    return cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_CONSTANT, value=value
    )
