"""Geometric normalisation: perspective correction, deskew and rotation.

A card photographed by hand is rarely axis-aligned. Two distinct problems
are handled separately because they need different tools:

* **Perspective** -- the card is a quadrilateral, not a rectangle, because
  the camera was not parallel to it. Fixed with a four-point transform
  onto the ID-1 aspect ratio.
* **Skew** -- the card is rectangular but rotated by a few degrees. Fixed
  by estimating the dominant text-line angle and rotating back.

Both are best-effort: if the evidence is weak the input is returned
unchanged, because a wrong warp is far more damaging than no warp.
"""

from __future__ import annotations

import cv2
import numpy as np

from nileid.config import CARD_HEIGHT, CARD_WIDTH
from nileid.logging_utils import get_logger

log = get_logger("preprocessing.geometry")

#: The card must fill at least this fraction of the frame for the contour
#: search to accept it as the document boundary.
MIN_CARD_AREA_RATIO = 0.20
#: ID-1 aspect ratio (85.6 x 54 mm). Candidate quadrilaterals must be close.
ID1_ASPECT = 85.6 / 54.0
ASPECT_TOLERANCE = 0.45
#: Skew angles smaller than this are not worth a resampling pass.
MIN_SKEW_DEGREES = 0.4
#: Text-line skew beyond this is rotation, not skew, and is left to the
#: orientation stage.
MAX_SKEW_DEGREES = 20.0

#: Width the frame is reduced to before searching for the card outline.
_CONTOUR_WORK_WIDTH = 800.0
#: Canny thresholds are set to this fraction either side of the image
#: median, so exposure does not have to be assumed.
_CANNY_SIGMA = 0.33


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(total)]  # smallest x+y
    ordered[2] = points[np.argmax(total)]  # largest x+y
    ordered[1] = points[np.argmin(diff)]  # smallest y-x
    ordered[3] = points[np.argmax(diff)]  # largest y-x
    return ordered


def four_point_transform(
    image: np.ndarray,
    corners: np.ndarray,
    width: int = CARD_WIDTH,
    height: int = CARD_HEIGHT,
) -> np.ndarray:
    """Warp the quadrilateral ``corners`` onto a ``width`` x ``height`` rectangle."""
    source = order_corners(corners)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


def find_card_quadrilateral(image: np.ndarray) -> np.ndarray | None:
    """Locate the card outline by contour search.

    Returns the four corners, or ``None`` when no convincing quadrilateral
    is found -- which is common against a cluttered background, and is why
    this is an optional refinement on top of the YOLO card detector rather
    than a replacement for it.
    """
    try:
        height, width = image.shape[:2]
        frame_area = float(height * width)
        if frame_area <= 0:
            return None

        # Work on a downscaled copy. Camera noise fragments the border edge
        # into dozens of unusable pieces at full resolution; area-averaging
        # down to a fixed width removes that noise and leaves the card
        # outline as the dominant edge. Corners are scaled back afterwards.
        scale = min(1.0, _CONTOUR_WORK_WIDTH / float(width))
        small = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Bilateral filtering suppresses the card's guilloche pattern while
        # keeping the true border edge intact.
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Canny thresholds derived from the image median, so a shadowed or
        # underexposed photograph is not silently edge-free.
        median = float(np.median(gray))
        lower = int(max(0.0, (1.0 - _CANNY_SIGMA) * median))
        upper = int(min(255.0, (1.0 + _CANNY_SIGMA) * median))
        edges = cv2.Canny(gray, lower, upper)

        # Close small gaps so a partially shadowed border still forms a loop.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        small_area = float(small.shape[0] * small.shape[1])
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(contour)
            if area / small_area < MIN_CARD_AREA_RATIO:
                break  # sorted by area: everything after this is smaller
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            corners = order_corners(approx)
            if _aspect_is_card_like(corners):
                return corners / scale if scale < 1.0 else corners
        return None
    except cv2.error:
        log.exception("find_card_quadrilateral failed.")
        return None


def _aspect_is_card_like(corners: np.ndarray) -> bool:
    """Reject quadrilaterals whose aspect ratio is not ID-1-like."""
    top_left, top_right, bottom_right, bottom_left = corners
    width = (np.linalg.norm(top_right - top_left) + np.linalg.norm(bottom_right - bottom_left)) / 2
    height = (np.linalg.norm(bottom_left - top_left) + np.linalg.norm(bottom_right - top_right)) / 2
    if height <= 1 or width <= 1:
        return False
    aspect = width / height
    # Accept a portrait-oriented card too; rotation is corrected later.
    return (
        abs(aspect - ID1_ASPECT) <= ASPECT_TOLERANCE
        or abs((1 / aspect) - ID1_ASPECT) <= ASPECT_TOLERANCE
    )


def correct_perspective(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Apply a four-point transform when a card outline can be found.

    Returns the (possibly unchanged) image and whether a warp was applied.
    """
    corners = find_card_quadrilateral(image)
    if corners is None:
        return image, False
    try:
        return four_point_transform(image, corners), True
    except cv2.error:
        log.exception("correct_perspective failed; returning the input image.")
        return image, False


#: Width the image is reduced to before searching for the skew angle.
_SKEW_WORK_WIDTH = 600
#: Fraction trimmed from each edge before measuring, to exclude the card
#: border. The border is a strong axis-aligned feature that otherwise
#: dominates the measurement no matter how the card is rotated.
_SKEW_INSET = 0.06


def _text_mask(image: np.ndarray) -> np.ndarray | None:
    """Binary mask of dark text on the card, with the border trimmed off."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    height, width = gray.shape[:2]
    inset_y, inset_x = int(height * _SKEW_INSET), int(width * _SKEW_INSET)
    if height - 2 * inset_y < 20 or width - 2 * inset_x < 20:
        return None
    gray = gray[inset_y : height - inset_y, inset_x : width - inset_x]

    # Work small: the angle of a text block does not need full resolution.
    if gray.shape[1] > _SKEW_WORK_WIDTH:
        scale = _SKEW_WORK_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) < 50:
        return None
    return mask


def _profile_score(mask: np.ndarray, angle: float) -> float:
    """Sharpness of the horizontal projection profile at a given angle.

    When text lines are level, every row is either dense with ink or empty,
    so the row-sum profile has large row-to-row jumps. The squared
    difference between adjacent rows peaks at the correct angle.
    """
    height, width = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(np.square(np.diff(profile)).sum())


def estimate_skew(image: np.ndarray) -> float:
    """Estimate the dominant text-line angle, in degrees.

    Returns the rotation that would bring the text level, so a card tilted
    clockwise by 3 degrees yields ``-3.0``.

    Uses a projection-profile search rather than the minimum-area rectangle
    of the text mask. The rectangle approach is defeated by the card's own
    border: that border is a strong axis-aligned shape, so the enclosing
    rectangle stays axis-aligned regardless of how the card is rotated, and
    the estimate collapses to zero. Hough lines over Canny edges -- the
    original approach -- fail for the same reason, and additionally lock
    onto the guilloche background pattern.

    Returns ``0.0`` when the estimate is unreliable or exceeds
    :data:`MAX_SKEW_DEGREES`, so an uncertain measurement leaves the image
    untouched instead of applying a wrong rotation.
    """
    try:
        mask = _text_mask(image)
        if mask is None:
            return 0.0

        # Coarse sweep across the supported range, then a fine refinement.
        coarse = max(
            (float(a) for a in np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + 1, 1.0)),
            key=lambda a: _profile_score(mask, a),
        )
        fine = max(
            (float(a) for a in np.arange(coarse - 1.0, coarse + 1.0 + 1e-9, 0.2)),
            key=lambda a: _profile_score(mask, a),
        )

        if abs(fine) < MIN_SKEW_DEGREES or abs(fine) > MAX_SKEW_DEGREES:
            return 0.0
        return round(fine, 2)
    except cv2.error:
        log.exception("estimate_skew failed.")
        return 0.0


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, keeping the border pixels replicated."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct small in-plane rotation. Returns the image and the angle used."""
    angle = estimate_skew(image)
    if abs(angle) < MIN_SKEW_DEGREES:
        return image, 0.0
    return rotate_bound(image, angle), angle


def rotate_quarter_turns(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate by 0, 90, 180 or 270 degrees clockwise without interpolation."""
    normalised = degrees % 360
    if normalised == 0:
        return image
    if normalised == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalised == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalised == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Expected a multiple of 90 degrees, got {degrees}.")


def resize_to_card(image: np.ndarray) -> np.ndarray:
    """Resize to the canonical card size used by the field detector."""
    return cv2.resize(image, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_CUBIC)
