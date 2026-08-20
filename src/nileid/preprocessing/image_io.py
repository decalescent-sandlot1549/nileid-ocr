"""Image loading and basic colour-space normalisation.

Accepts a path, raw bytes or an existing array and always yields a
contiguous 3-channel BGR ``uint8`` image, so every downstream stage can
assume one representation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nileid.logging_utils import get_logger

log = get_logger("preprocessing.image_io")

ImageInput = str | Path | bytes | bytearray | np.ndarray

#: Extensions OpenCV can decode and that we advertise as supported.
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


class ImageLoadError(ValueError):
    """Raised when the input cannot be decoded into an image."""


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce grayscale / BGRA / RGB-like arrays to 3-channel BGR uint8."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ImageLoadError(f"Unsupported image shape {image.shape!r}.")

    if image.dtype != np.uint8:
        # Float images from other libraries may be 0-1 or 0-255.
        maximum = float(image.max()) if image.size else 0.0
        scale = 255.0 if 0.0 < maximum <= 1.0 else 1.0
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def decode_bytes(data: bytes | bytearray) -> np.ndarray:
    """Decode encoded image bytes into a BGR array."""
    if not data:
        raise ImageLoadError("Received empty image data.")
    buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(
            "Could not decode the image. Supported formats: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    return to_bgr(image)


def load_image(source: ImageInput) -> np.ndarray:
    """Load an image from a path, raw bytes or an array.

    Raises:
        ImageLoadError: if the source is missing, empty or undecodable.
    """
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ImageLoadError("Received an empty array.")
        return to_bgr(source.copy())

    if isinstance(source, (bytes, bytearray)):
        return decode_bytes(source)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ImageLoadError(f"File not found: {path}")
        if not path.is_file():
            raise ImageLoadError(f"Not a file: {path}")
        # imread does not handle non-ASCII paths reliably on Windows, so
        # the file is read through Python and decoded from memory.
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ImageLoadError(f"Could not read {path}: {exc}") from exc
        return decode_bytes(data)

    raise ImageLoadError(f"Unsupported input type {type(source).__name__!r}.")


def safe_crop(
    image: np.ndarray, x1: float, y1: float, x2: float, y2: float, pad: int = 0
) -> np.ndarray | None:
    """Crop with clamping to the image bounds; ``None`` if the box is empty."""
    height, width = image.shape[:2]
    xa = max(0, int(x1) - pad)
    ya = max(0, int(y1) - pad)
    xb = min(width, int(x2) + pad)
    yb = min(height, int(y2) + pad)
    if xb <= xa or yb <= ya:
        return None
    return image[ya:yb, xa:xb].copy()


def expand_box(
    box: tuple[float, float, float, float],
    scale_w: float,
    scale_h: float,
    shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Grow a box about its centre, clamped to the image.

    Field detectors tend to hug the glyphs; a small margin gives OCR the
    surrounding whitespace it needs to segment the line correctly.
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    cx, cy = x1 + width / 2.0, y1 + height / 2.0
    new_w, new_h = width * scale_w, height * scale_h
    img_h, img_w = shape[0], shape[1]
    return (
        max(0, int(cx - new_w / 2.0)),
        max(0, int(cy - new_h / 2.0)),
        min(img_w, int(cx + new_w / 2.0)),
        min(img_h, int(cy + new_h / 2.0)),
    )
