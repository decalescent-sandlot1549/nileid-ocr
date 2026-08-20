"""National ID digit recognition.

``detect_id.pt`` is an object detector whose ten classes are the digits
``0``-``9``, so the class index *is* the digit value. Reading the number is
therefore a detection problem, not a text-recognition problem: detect every
digit, sort left to right, concatenate.

Three failure modes the original implementation did not guard against:

* **Duplicates.** Two overlapping boxes on the same printed digit produce a
  15- or 16-digit string. Non-maximum suppression inside the detector is
  per class, so two *different* classes firing on one glyph both survive.
* **Low-confidence noise.** Inference ran at ``conf=0.10`` and every box
  was accepted, so guilloche texture became digits.
* **No length check.** Whatever came out was returned as the National ID,
  even when it was nine digits long.

Here overlapping boxes are collapsed to the most confident one, weak boxes
are dropped, and the result is returned with per-digit confidences so the
caller can see how solid the read is. Missing digits are never invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nileid.config import Settings
from nileid.logging_utils import get_logger
from nileid.models import get_detector
from nileid.preprocessing.enhance import enhance_for_digits, pad
from nileid.validation.national_id import ID_LENGTH

log = get_logger("detection.digits")


@dataclass
class DigitRead:
    """The outcome of reading the National ID strip."""

    digits: str
    #: Mean confidence across the accepted digit boxes.
    confidence: float
    #: Confidence of each digit, in reading order.
    per_digit: list[float]

    @property
    def is_complete(self) -> bool:
        return len(self.digits) == ID_LENGTH


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _suppress_overlaps(
    candidates: list[tuple[float, int, float, tuple[float, ...]]], iou_threshold: float
) -> list[tuple[float, int, float, tuple[float, ...]]]:
    """Class-agnostic non-maximum suppression.

    Ultralytics applies NMS per class, so two different digit classes can
    both survive on a single glyph. Suppressing across classes is what
    prevents a 14-digit number being read as 15 digits.
    """
    kept: list[tuple[float, int, float, tuple[float, ...]]] = []
    for candidate in sorted(candidates, key=lambda c: c[2], reverse=True):
        if all(_iou(candidate[3], k[3]) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def read_digits(strip: np.ndarray, settings: Settings) -> DigitRead:
    """Read the digits of the National ID from a crop of the number strip."""
    if strip is None or strip.size == 0:
        return DigitRead("", 0.0, [])

    model = get_detector("digits", settings)
    prepared = enhance_for_digits(pad(strip, border=24))

    predictions = model.predict(prepared, conf=settings.digit_conf, verbose=False)

    candidates: list[tuple[float, int, float, tuple[float, ...]]] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            # The class index is the digit value; anything outside 0-9
            # means the checkpoint is not the digit model we expect.
            if not 0 <= class_id <= 9:
                log.warning("Digit detector produced unexpected class %d; ignoring.", class_id)
                continue
            coordinates = tuple(float(v) for v in box.xyxy[0])
            candidates.append((coordinates[0], class_id, float(box.conf[0]), coordinates))

    if not candidates:
        return DigitRead("", 0.0, [])

    kept = _suppress_overlaps(candidates, settings.digit_iou)
    # Egyptian National ID digits read left to right, even on an otherwise
    # right-to-left card.
    kept.sort(key=lambda c: c[0])

    digits = "".join(str(c[1]) for c in kept)
    confidences = [c[2] for c in kept]
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0

    if len(digits) != ID_LENGTH:
        log.debug("Read %d digits (expected %d): %s", len(digits), ID_LENGTH, digits)

    return DigitRead(digits=digits, confidence=mean_confidence, per_digit=confidences)
