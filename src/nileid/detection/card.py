"""Card localisation and orientation.

``detect_id_card.pt`` predicts eight classes: ``{front,back}-{up,down,left,
right}``. The class therefore carries two facts the original pipeline threw
away -- which side of the card is showing, and how it is rotated.

Using them replaces the previous strategy of running the field detector on
all four rotations and picking a winner by heuristic score. That strategy
cost four inference passes per image and, because its score table referred
to class names the model does not have, the positional bonuses never fired.

A brute-force fallback is retained for images where the card detector is
not confident, but it now scores against class names read from the model
itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nileid.config import Settings
from nileid.logging_utils import get_logger
from nileid.models import get_detector
from nileid.preprocessing.geometry import resize_to_card, rotate_quarter_turns
from nileid.preprocessing.image_io import expand_box, safe_crop

log = get_logger("detection.card")

#: Clockwise rotation, in degrees, that brings each detected orientation
#: upright. "up" already is upright; "left" means the card's top edge points
#: left, so it needs a quarter turn clockwise to stand up.
ORIENTATION_ROTATION: dict[str, int] = {
    "up": 0,
    "right": 90,
    "down": 180,
    "left": 270,
}

#: Margin added around the detected card box before cropping, so the card
#: border and any text near it survive.
CARD_BOX_MARGIN = 1.04


@dataclass
class CardDetection:
    """A localised card, normalised to the canonical size and orientation."""

    image: np.ndarray
    confidence: float
    side: str | None
    rotation: int
    #: True when the orientation came from the card detector rather than
    #: from the fallback search.
    orientation_from_model: bool = True


def _parse_class_name(name: str) -> tuple[str | None, str | None]:
    """Split ``"front-up"`` into ``("front", "up")``."""
    parts = name.lower().split("-")
    if len(parts) != 2:
        return None, None
    side, orientation = parts
    if side not in ("front", "back") or orientation not in ORIENTATION_ROTATION:
        return None, None
    return side, orientation


def detect_card(image: np.ndarray, settings: Settings) -> CardDetection | None:
    """Locate the ID card and return it upright at the canonical size.

    Returns ``None`` when no detection clears ``settings.card_conf``.
    """
    model = get_detector("card", settings)
    predictions = model.predict(image, conf=settings.card_conf, verbose=False)

    best = None
    best_conf = 0.0
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence <= best_conf:
                continue
            class_name = prediction.names.get(int(box.cls[0]), "")
            side, orientation = _parse_class_name(class_name)
            coordinates = [float(v) for v in box.xyxy[0]]
            crop_box = expand_box(tuple(coordinates), CARD_BOX_MARGIN, CARD_BOX_MARGIN, image.shape)
            crop = safe_crop(image, *crop_box)
            if crop is None:
                continue
            height, width = crop.shape[:2]
            if height < settings.min_card_height or width < settings.min_card_width:
                # A box this small is a field or a logo, not the card.
                continue
            best_conf = confidence
            best = (crop, confidence, side, orientation)

    if best is None:
        return None

    crop, confidence, side, orientation = best
    rotation = ORIENTATION_ROTATION.get(orientation or "up", 0)
    upright = rotate_quarter_turns(crop, rotation)
    return CardDetection(
        image=resize_to_card(upright),
        confidence=confidence,
        side=side,
        rotation=rotation,
        orientation_from_model=orientation is not None,
    )


def orientation_by_layout(image: np.ndarray, settings: Settings) -> tuple[np.ndarray, int]:
    """Fallback orientation search using the field detector's layout.

    Scores each quarter turn by how well the detected fields match the
    known geometry of the card front: the National ID sits in the lower
    band, the serial number in the upper band, and the name fields on the
    right-hand side (Arabic is right-aligned).

    Class names are read from ``model.names`` rather than a hard-coded
    table, so the score cannot silently refer to classes that do not exist.
    """
    from nileid.detection.fields import canonical_field_name

    model = get_detector("fields", settings)

    best_image, best_rotation, best_score = image, 0, float("-inf")
    for rotation in (0, 90, 180, 270):
        candidate = resize_to_card(rotate_quarter_turns(image, rotation))
        score = _layout_score(candidate, model, settings, canonical_field_name)
        log.debug("orientation %3d deg -> score %.1f", rotation, score)
        if score > best_score:
            best_score, best_image, best_rotation = score, candidate, rotation

    return best_image, best_rotation


def _layout_score(image, model, settings, canonical) -> float:
    """Score how card-like the layout of ``image`` is."""
    predictions = model.predict(image, conf=settings.field_conf, verbose=False)
    if not predictions:
        return float("-inf")
    boxes = getattr(predictions[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return float("-inf")

    names = predictions[0].names
    height, width = image.shape[:2]
    score = 0.0
    for box in boxes:
        confidence = float(box.conf[0])
        field = canonical(names.get(int(box.cls[0]), ""))
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Any confident detection is weak evidence the card is upright.
        score += confidence * 10.0

        if field == "nid":
            score += 40.0 if cy > height * 0.60 else -40.0
        elif field == "serial":
            score += 25.0 if cy < height * 0.40 else -25.0
        elif field in ("first_name", "last_name"):
            score += 20.0 if cx > width * 0.45 else -20.0
        elif field == "photo":
            score += 20.0 if cx < width * 0.40 else -20.0
    return score
