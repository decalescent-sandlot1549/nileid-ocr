"""Card, field and digit detection."""

from nileid.detection.card import CardDetection, detect_card, orientation_by_layout
from nileid.detection.digits import DigitRead, read_digits
from nileid.detection.fields import (
    FieldBox,
    best_per_field,
    canonical_field_name,
    detect_fields,
    detect_regions,
)

__all__ = [
    "CardDetection",
    "DigitRead",
    "FieldBox",
    "best_per_field",
    "canonical_field_name",
    "detect_card",
    "detect_fields",
    "detect_regions",
    "orientation_by_layout",
    "read_digits",
]
