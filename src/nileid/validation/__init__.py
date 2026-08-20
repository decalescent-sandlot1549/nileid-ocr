"""Structural validation and decoding of Egyptian National ID numbers."""

from nileid.validation.governorates import GOVERNORATES, governorate_name
from nileid.validation.national_id import (
    DecodedID,
    decode_national_id,
    normalize_digits,
    validate_national_id,
)

__all__ = [
    "GOVERNORATES",
    "DecodedID",
    "decode_national_id",
    "governorate_name",
    "normalize_digits",
    "validate_national_id",
]
