"""NileID -- structured information extraction from Egyptian National ID cards.

Typical use::

    from nileid import EgyptianIDReader

    reader = EgyptianIDReader()
    result = reader.read("card.jpg")

    print(result.national_id.value)   # the digits, or None if unreadable
    print(result.to_dict())           # full structured output
"""

from nileid.config import Settings
from nileid.models import ModelNotFoundError
from nileid.pipeline.reader import EgyptianIDReader
from nileid.preprocessing.image_io import ImageLoadError
from nileid.results import Field, IDCardResult, NationalIDValidation, Status
from nileid.validation.national_id import (
    decode_national_id,
    normalize_digits,
    validate_national_id,
)

__version__ = "1.0.0"

__all__ = [
    "EgyptianIDReader",
    "Field",
    "IDCardResult",
    "ImageLoadError",
    "ModelNotFoundError",
    "NationalIDValidation",
    "Settings",
    "Status",
    "__version__",
    "decode_national_id",
    "normalize_digits",
    "validate_national_id",
]
