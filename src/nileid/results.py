"""Typed result objects returned by the pipeline.

The design goal is transparency. Three layers are kept distinct and are all
visible in the output:

1. ``raw``        — what the OCR engine actually returned.
2. ``value``      — the normalised, post-processed value.
3. ``validation`` — whether the value passes structural checks.

A field that could not be read is represented by ``value=None`` rather than
an empty string, so that "not present" is never confused with "read as blank".
No layer is permitted to invent content that OCR did not produce.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Outcome of a single field extraction."""

    OK = "ok"
    #: Read, but below the confidence threshold — treat as a suggestion.
    LOW_CONFIDENCE = "low_confidence"
    #: The region was located but OCR produced nothing usable.
    UNREADABLE = "unreadable"
    #: The region was never located on the card.
    NOT_FOUND = "not_found"


@dataclass
class Field:
    """A single extracted field with its provenance."""

    value: str | None = None
    raw: str | None = None
    confidence: float = 0.0
    status: Status = Status.NOT_FOUND

    @property
    def is_usable(self) -> bool:
        return self.value is not None and self.status in (Status.OK, Status.LOW_CONFIDENCE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw": self.raw,
            "confidence": round(float(self.confidence), 4),
            "status": self.status.value,
        }


@dataclass
class NationalIDValidation:
    """Structural validation of a 14-digit Egyptian National ID number."""

    valid: bool = False
    length_ok: bool = False
    checksum_ok: bool | None = None  # None: not evaluated / not applicable
    date_ok: bool = False
    governorate_ok: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "length_ok": self.length_ok,
            "checksum_ok": self.checksum_ok,
            "date_ok": self.date_ok,
            "governorate_ok": self.governorate_ok,
            "errors": list(self.errors),
        }


@dataclass
class IDCardResult:
    """Structured outcome of reading one image."""

    # ── Detection ────────────────────────────────────────────────────
    card_detected: bool = False
    card_confidence: float = 0.0
    #: Orientation correction applied, in degrees clockwise.
    rotation_applied: int = 0
    #: Card side reported by the detector: "front", "back" or None.
    side: str | None = None

    # ── Fields read from the card face ───────────────────────────────
    first_name: Field = field(default_factory=Field)
    last_name: Field = field(default_factory=Field)
    full_name: Field = field(default_factory=Field)
    address: Field = field(default_factory=Field)
    national_id: Field = field(default_factory=Field)

    # ── Values derived from the National ID number ───────────────────
    #: These are decoded, not read by OCR. They are only populated when
    #: the number passes validation.
    birth_date: str | None = None
    governorate: str | None = None
    gender: str | None = None

    validation: NationalIDValidation = field(default_factory=NationalIDValidation)
    warnings: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def is_empty(self) -> bool:
        """True when no field at all could be read."""
        return not any(
            f.is_usable for f in (self.first_name, self.last_name, self.address, self.national_id)
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "card_detected": self.card_detected,
            "card_confidence": round(float(self.card_confidence), 4),
            "rotation_applied": self.rotation_applied,
            "side": self.side,
            "fields": {
                "first_name": self.first_name.to_dict(),
                "last_name": self.last_name.to_dict(),
                "full_name": self.full_name.to_dict(),
                "address": self.address.to_dict(),
                "national_id": self.national_id.to_dict(),
            },
            "derived": {
                "birth_date": self.birth_date,
                "governorate": self.governorate,
                "gender": self.gender,
            },
            "validation": self.validation.to_dict(),
            "warnings": list(self.warnings),
            "processing_time_ms": round(self.processing_time_ms, 1),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Flattened ``field -> value`` view, for quick scripting.

        Unreadable fields are ``None``. Use :meth:`to_dict` when confidence
        and validation detail matter.
        """
        return {
            "first_name": self.first_name.value,
            "last_name": self.last_name.value,
            "full_name": self.full_name.value,
            "address": self.address.value,
            "national_id": self.national_id.value,
            "birth_date": self.birth_date,
            "governorate": self.governorate,
            "gender": self.gender,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = [
            f"card_detected={self.card_detected}",
            f"national_id={self.national_id.value!r}",
            f"full_name={self.full_name.value!r}",
            f"valid={self.validation.valid}",
        ]
        return f"IDCardResult({', '.join(parts)})"


def _asdict(obj: Any) -> dict[str, Any]:  # pragma: no cover - helper
    return asdict(obj)
