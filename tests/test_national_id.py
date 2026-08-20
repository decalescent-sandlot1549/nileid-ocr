"""Validation and decoding of National ID numbers.

Every number below is synthetic and constructed to exercise a specific
branch. None is issued to a real person.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nileid.validation import decode_national_id, normalize_digits, validate_national_id
from nileid.validation.national_id import ID_LENGTH

VALID = "29001010100017"  # 1990-01-01, Cairo, male


class TestNormalizeDigits:
    def test_passes_ascii_digits_through(self):
        assert normalize_digits(VALID) == VALID

    def test_converts_arabic_indic_digits(self):
        assert normalize_digits("٢٩٠٠١٠١٠١٠٠٠١٧") == VALID

    def test_converts_extended_arabic_indic_digits(self):
        assert normalize_digits("۲۹۰۰۱۰۱۰۱۰۰۰۱۷") == VALID

    @pytest.mark.parametrize(
        "raw",
        ["2 9001 0101 00017", "29001-01010-0017", "2900101010 0017 ", "٢٩٠٠١ 01010 0017"],
    )
    def test_strips_separators(self, raw):
        assert normalize_digits(raw) == VALID

    @pytest.mark.parametrize("raw", [None, "", "   ", "abc", "الرقم القومي"])
    def test_returns_empty_for_non_numeric(self, raw):
        assert normalize_digits(raw) == ""


class TestValidation:
    def test_accepts_a_well_formed_number(self):
        result = validate_national_id(VALID)
        assert result.valid
        assert result.length_ok and result.date_ok and result.governorate_ok
        assert result.errors == []

    def test_reports_no_digits(self):
        result = validate_national_id("")
        assert not result.valid
        assert "no_digits" in result.errors

    @pytest.mark.parametrize("number", ["2900101010001", "290010101000177", "559554555"])
    def test_rejects_wrong_length(self, number):
        result = validate_national_id(number)
        assert not result.valid
        assert not result.length_ok
        assert any("expected 14 digits" in e for e in result.errors)

    def test_short_number_does_not_evaluate_positional_checks(self):
        # A truncated read must not be reported as having a valid date.
        result = validate_national_id("2900101")
        assert not result.date_ok
        assert not result.governorate_ok

    @pytest.mark.parametrize("century", ["0", "1", "4", "9"])
    def test_rejects_unknown_century_digit(self, century):
        result = validate_national_id(century + VALID[1:])
        assert not result.valid
        assert any("century" in e for e in result.errors)

    @pytest.mark.parametrize(
        "number",
        [
            "29013010100017",  # month 13
            "29000010100017",  # month 00
            "29001320100017",  # day 32
            "29002300100017",  # 30 February
        ],
    )
    def test_rejects_impossible_dates(self, number):
        result = validate_national_id(number)
        assert not result.valid
        assert not result.date_ok
        assert any("impossible birth date" in e for e in result.errors)

    def test_rejects_future_birth_date(self):
        future = dt.date.today().year + 1
        if future >= 2100:  # pragma: no cover - guards a far-future clock
            pytest.skip("century encoding does not reach this year")
        number = f"3{future % 100:02d}0101010001"
        number += "7"
        result = validate_national_id(number)
        assert not result.date_ok

    @pytest.mark.parametrize("code", ["00", "05", "10", "20", "30", "99", "77"])
    def test_rejects_unknown_governorate_codes(self, code):
        result = validate_national_id(VALID[:7] + code + VALID[9:])
        assert not result.valid
        assert not result.governorate_ok

    @pytest.mark.parametrize("code", ["01", "02", "21", "88"])
    def test_accepts_known_governorate_codes(self, code):
        result = validate_national_id(VALID[:7] + code + VALID[9:])
        assert result.governorate_ok

    def test_checksum_is_advisory_not_a_validity_gate(self):
        # The official algorithm is unpublished, so a mismatch must not by
        # itself mark an otherwise well-formed number invalid.
        mismatched = VALID[:13] + ("0" if VALID[13] != "0" else "1")
        result = validate_national_id(mismatched)
        assert result.checksum_ok is False
        assert result.valid, "a check-digit mismatch must not fail validation"

    def test_checksum_not_evaluated_for_short_numbers(self):
        assert validate_national_id("12345").checksum_ok is None


class TestDecoding:
    def test_decodes_every_component(self):
        decoded = decode_national_id(VALID)
        assert decoded is not None
        assert decoded.national_id == VALID
        assert decoded.birth_date == dt.date(1990, 1, 1)
        assert decoded.birth_date_iso == "1990-01-01"
        assert decoded.governorate_code == "01"
        assert decoded.governorate == "القاهرة"
        assert decoded.gender == "male"
        assert decoded.sequence == "0001"

    def test_century_two_maps_to_1900s(self):
        decoded = decode_national_id(VALID)
        assert decoded is not None
        assert decoded.birth_date.year == 1990

    def test_century_three_maps_to_2000s(self):
        # "3" + year "05" -> 2005. (Century 3 with year 90 would be 2090,
        # which is in the future and is correctly refused.)
        decoded = decode_national_id("30501010100017")
        assert decoded is not None
        assert decoded.birth_date.year == 2005

    def test_gender_is_read_from_position_thirteen(self):
        male = decode_national_id(VALID[:12] + "1" + VALID[13])
        female = decode_national_id(VALID[:12] + "2" + VALID[13])
        assert male is not None and male.gender == "male"
        assert female is not None and female.gender == "female"

    @pytest.mark.parametrize(
        "number", ["", "2900101010001", "29013010100017", "abcdefghijklmn", None]
    )
    def test_refuses_to_decode_invalid_numbers(self, number):
        # A half-read number must never yield a confident birth date.
        assert decode_national_id(number) is None

    def test_accepts_arabic_indic_input(self):
        decoded = decode_national_id("٢٩٠٠١٠١٠١٠٠٠١٧")
        assert decoded is not None
        assert decoded.national_id == VALID


def test_id_length_constant():
    assert ID_LENGTH == 14
    assert len(VALID) == ID_LENGTH
