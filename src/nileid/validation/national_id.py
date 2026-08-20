"""Parsing, validation and decoding of the 14-digit Egyptian National ID.

Layout of the number (``C YY MM DD GG SSSS K``)::

    index  0      century   2 -> 1900-1999, 3 -> 2000-2099
    index  1-2    year of birth, last two digits
    index  3-4    month of birth
    index  5-6    day of birth
    index  7-8    governorate code (see :mod:`nileid.validation.governorates`)
    index  9-12   sequence number within that day and governorate;
                  index 12 is odd for male, even for female
    index  13     check digit

This module is deliberately pure: it takes a string and returns structured
facts about it. It never repairs a number by guessing missing digits — a
partially read ID stays invalid, which is the only honest outcome.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

from nileid.results import NationalIDValidation
from nileid.validation.governorates import governorate_name

ID_LENGTH = 14

#: Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digit ranges.
_ARABIC_INDIC = {ord("\u0660") + i: str(i) for i in range(10)}
_EXT_ARABIC_INDIC = {ord("\u06f0") + i: str(i) for i in range(10)}
_DIGIT_TRANSLATION = {**_ARABIC_INDIC, **_EXT_ARABIC_INDIC}

_NON_DIGIT = re.compile(r"\D")


def normalize_digits(text: str | None) -> str:
    """Convert Arabic-Indic digits to ASCII and strip every non-digit.

    Separators, stray OCR punctuation and Arabic numerals are all handled::

        >>> normalize_digits("٢٩٠٠١ 01 010 0015")
        '29001010100015'
    """
    if not text:
        return ""
    return _NON_DIGIT.sub("", str(text).translate(_DIGIT_TRANSLATION))


@dataclass(frozen=True)
class DecodedID:
    """Facts decoded from a structurally valid National ID number."""

    national_id: str
    birth_date: _dt.date
    governorate_code: str
    governorate: str | None
    gender: str
    sequence: str

    @property
    def birth_date_iso(self) -> str:
        return self.birth_date.isoformat()


def _century_base(century_digit: int) -> int | None:
    """Map the leading century digit to the base year."""
    return {2: 1900, 3: 2000}.get(century_digit)


def validate_national_id(number: str | None) -> NationalIDValidation:
    """Validate the structure of a National ID number.

    The returned object reports each check separately so that a caller can
    distinguish "OCR dropped a digit" from "this date cannot exist".
    """
    result = NationalIDValidation()
    digits = normalize_digits(number)

    if not digits:
        result.errors.append("no_digits")
        return result

    result.length_ok = len(digits) == ID_LENGTH
    if not result.length_ok:
        result.errors.append(f"expected {ID_LENGTH} digits, found {len(digits)}")
        # Every remaining check depends on positional layout.
        return result

    # ── Date ─────────────────────────────────────────────────────────
    base = _century_base(int(digits[0]))
    if base is None:
        result.errors.append(f"invalid century digit {digits[0]!r} (expected 2 or 3)")
    else:
        year = base + int(digits[1:3])
        month = int(digits[3:5])
        day = int(digits[5:7])
        try:
            birth = _dt.date(year, month, day)
        except ValueError:
            result.errors.append(f"impossible birth date {year:04d}-{month:02d}-{day:02d}")
        else:
            if birth > _dt.date.today():
                result.errors.append("birth date is in the future")
            else:
                result.date_ok = True

    # ── Governorate ──────────────────────────────────────────────────
    gov_code = digits[7:9]
    if governorate_name(gov_code) is None:
        result.errors.append(f"unknown governorate code {gov_code!r}")
    else:
        result.governorate_ok = True

    # ── Check digit ──────────────────────────────────────────────────
    result.checksum_ok = _checksum_ok(digits)
    if result.checksum_ok is False:
        result.errors.append("check_digit_mismatch (advisory)")

    # The check digit is advisory only: the official algorithm is not
    # published, so a mismatch must not by itself condemn a number that is
    # otherwise well formed. It is surfaced in `checksum_ok` and as a
    # pipeline warning so the caller can decide how much weight to give it.
    result.valid = bool(result.length_ok and result.date_ok and result.governorate_ok)
    return result


def _checksum_ok(digits: str) -> bool | None:
    """Verify the trailing check digit.

    The Civil Status Organisation does not publish the algorithm, so the
    widely used weighted-modulus-11 scheme is applied here. Because the
    scheme is unofficial, a mismatch is reported as a warning by the
    pipeline rather than being treated as proof that the number is wrong.
    Returns ``None`` when the check cannot be evaluated.
    """
    if len(digits) != ID_LENGTH or not digits.isdigit():
        return None
    weights = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
    total = sum(int(d) * w for d, w in zip(digits[:13], weights, strict=True))
    remainder = total % 11
    expected = 0 if remainder < 2 else 11 - remainder
    return expected == int(digits[13])


def decode_national_id(number: str | None) -> DecodedID | None:
    """Decode a National ID number, or return ``None`` if it is not valid.

    Decoding deliberately requires the structural checks to pass. Reporting
    a birth date derived from a half-read number would be worse than
    reporting nothing at all.
    """
    digits = normalize_digits(number)
    validation = validate_national_id(digits)
    if not (validation.length_ok and validation.date_ok and validation.governorate_ok):
        return None

    base = _century_base(int(digits[0]))
    assert base is not None  # guaranteed by date_ok
    birth = _dt.date(base + int(digits[1:3]), int(digits[3:5]), int(digits[5:7]))
    gov_code = digits[7:9]

    return DecodedID(
        national_id=digits,
        birth_date=birth,
        governorate_code=gov_code,
        governorate=governorate_name(gov_code),
        gender="male" if int(digits[12]) % 2 else "female",
        sequence=digits[9:13],
    )
