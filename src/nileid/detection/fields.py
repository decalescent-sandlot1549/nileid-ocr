"""Field localisation on a normalised card image.

Two detectors cover the card face:

``detect_odjects.pt`` (**primary**, 31 classes)
    Detects every printed region and additionally distinguishes a clean
    region from a suspect one by prefixing the class with ``invalid_``
    (``nid`` vs ``invalid_nid``, and so on).

``best.pt`` (**secondary**, 7 classes named ``"0"``-``"6"``)
    A smaller region detector whose class *names* are bare indices. The
    mapping below was recovered by inspecting where each class fires on a
    card; index 3 is the first-name line, which is the single use the
    original pipeline made of this model.

A note on the ``invalid_`` prefix: the original code compared class names
for exact equality against ``"nid"``, ``"address"`` and ``"lastName"``, so
whenever the detector emitted the ``invalid_`` variant -- which it does for
any card it considers atypical -- **every field silently came back empty**
with no indication to the caller. Here both variants map to the same
canonical field and the suspicion is reported as a warning instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nileid.config import Settings
from nileid.logging_utils import get_logger
from nileid.models import get_detector

log = get_logger("detection.fields")

INVALID_PREFIX = "invalid_"

#: Detector class name (with any ``invalid_`` prefix stripped) -> canonical
#: field name used throughout the pipeline.
FIELD_ALIASES: dict[str, str] = {
    "address": "address",
    "firstname": "first_name",
    "lastname": "last_name",
    "nid": "nid",
    "nid_back": "nid_back",
    "serial": "serial",
    "photo": "photo",
    "dob": "date_of_birth",
    "expiry": "expiry",
    "issue": "issue",
    "job": "job",
    "demo": "demographics",
    "poe": "place_of_employment",
    "front_logo": "logo",
    "barcode": "barcode",
    "watermark_tut": "watermark",
}

#: Empirically recovered class map for ``best.pt``. Its checkpoint labels
#: classes ``"0"``-``"6"``; these names come from observing which region
#: each class fires on. Only the entries actually consumed are mapped.
REGION_MODEL_CLASSES: dict[int, str] = {
    0: "last_name",
    1: "address",
    3: "first_name",
    5: "nid",
}

#: Per-field crop margins. Detectors hug the glyphs; OCR needs the
#: surrounding whitespace to segment a line. The National ID strip gets the
#: most room because clipping an outer digit silently shortens the number.
FIELD_MARGINS: dict[str, tuple[float, float]] = {
    "first_name": (1.15, 1.30),
    "last_name": (1.10, 1.25),
    "address": (1.12, 1.35),
    "nid": (1.18, 1.60),
}
DEFAULT_MARGIN = (1.10, 1.20)


def canonical_field_name(raw: str) -> str:
    """Normalise a detector class name to a canonical field name.

    ``"invalid_lastName"`` and ``"lastName"`` both become ``"last_name"``.
    Unknown classes are returned lower-cased so they can still be logged.
    """
    if not raw:
        return ""
    name = raw.strip()
    if name.lower().startswith(INVALID_PREFIX):
        name = name[len(INVALID_PREFIX) :]
    return FIELD_ALIASES.get(name.lower(), name.lower())


def is_flagged_invalid(raw: str) -> bool:
    """True when the detector emitted the ``invalid_`` variant of a class."""
    return bool(raw) and raw.strip().lower().startswith(INVALID_PREFIX)


@dataclass
class FieldBox:
    """One detected region on the card."""

    field: str
    box: tuple[float, float, float, float]
    confidence: float
    #: The detector's own class name, kept for diagnostics.
    raw_class: str
    flagged_invalid: bool = False

    @property
    def margin(self) -> tuple[float, float]:
        return FIELD_MARGINS.get(self.field, DEFAULT_MARGIN)


def detect_fields(image: np.ndarray, settings: Settings) -> list[FieldBox]:
    """Detect card regions with the primary field detector."""
    model = get_detector("fields", settings)
    predictions = model.predict(image, conf=settings.field_conf, verbose=False)

    detections: list[FieldBox] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        names = prediction.names
        for box in boxes:
            raw_class = names.get(int(box.cls[0]), "")
            detections.append(
                FieldBox(
                    field=canonical_field_name(raw_class),
                    box=tuple(float(v) for v in box.xyxy[0]),
                    confidence=float(box.conf[0]),
                    raw_class=raw_class,
                    flagged_invalid=is_flagged_invalid(raw_class),
                )
            )
    return detections


def detect_regions(image: np.ndarray, settings: Settings) -> list[FieldBox]:
    """Detect card regions with the secondary region detector (``best.pt``)."""
    model = get_detector("regions", settings)
    predictions = model.predict(image, conf=settings.field_conf, verbose=False)

    detections: list[FieldBox] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            field = REGION_MODEL_CLASSES.get(class_id)
            if field is None:
                continue
            detections.append(
                FieldBox(
                    field=field,
                    box=tuple(float(v) for v in box.xyxy[0]),
                    confidence=float(box.conf[0]),
                    raw_class=f"region:{class_id}",
                )
            )
    return detections


def best_per_field(detections: list[FieldBox]) -> dict[str, FieldBox]:
    """Keep the highest-confidence detection for each canonical field."""
    best: dict[str, FieldBox] = {}
    for detection in detections:
        if not detection.field:
            continue
        current = best.get(detection.field)
        if current is None or detection.confidence > current.confidence:
            best[detection.field] = detection
    return best


def merge_detections(
    primary: dict[str, FieldBox], secondary: dict[str, FieldBox]
) -> dict[str, FieldBox]:
    """Fill fields the primary detector missed using the secondary one.

    The primary detector always wins where both fired; the secondary only
    contributes fields that would otherwise be absent.
    """
    merged = dict(primary)
    for field, detection in secondary.items():
        if field not in merged:
            log.debug("Field %r recovered from the secondary detector.", field)
            merged[field] = detection
    return merged
