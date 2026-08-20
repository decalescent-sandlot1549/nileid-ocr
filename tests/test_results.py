"""The result schema.

These tests pin the output contract, and in particular the guarantees that
make a failed read visible rather than silent.
"""

from __future__ import annotations

import json

from nileid.results import Field, IDCardResult, NationalIDValidation, Status


class TestField:
    def test_defaults_to_not_found(self):
        field = Field()
        assert field.value is None
        assert field.status is Status.NOT_FOUND
        assert not field.is_usable

    def test_unread_field_is_none_not_empty_string(self):
        # "not present" must never be confused with "read as blank".
        assert Field(status=Status.UNREADABLE).value is None

    def test_usable_for_ok_and_low_confidence(self):
        assert Field("x", "x", 0.9, Status.OK).is_usable
        assert Field("x", "x", 0.2, Status.LOW_CONFIDENCE).is_usable

    def test_not_usable_when_unreadable(self):
        assert not Field(None, "raw", 0.1, Status.UNREADABLE).is_usable

    def test_serialises_raw_alongside_value(self):
        payload = Field("محمد", "محمـد", 0.87, Status.OK).to_dict()
        assert payload["value"] == "محمد"
        assert payload["raw"] == "محمـد"
        assert payload["confidence"] == 0.87
        assert payload["status"] == "ok"


class TestValidation:
    def test_defaults_are_negative(self):
        validation = NationalIDValidation()
        assert not validation.valid
        assert validation.checksum_ok is None
        assert validation.errors == []

    def test_serialises_errors_as_a_list_copy(self):
        validation = NationalIDValidation(errors=["a"])
        payload = validation.to_dict()
        payload["errors"].append("b")
        assert validation.errors == ["a"]


class TestIDCardResult:
    def test_new_result_is_empty(self):
        assert IDCardResult().is_empty

    def test_not_empty_once_a_field_is_read(self):
        result = IDCardResult()
        result.national_id = Field("29001010100017", None, 0.9, Status.OK)
        assert not result.is_empty

    def test_derived_fields_default_to_none(self):
        result = IDCardResult()
        assert result.birth_date is None
        assert result.governorate is None
        assert result.gender is None

    def test_warnings_are_deduplicated(self):
        result = IDCardResult()
        result.warn("same")
        result.warn("same")
        result.warn("other")
        assert result.warnings == ["same", "other"]

    def test_to_dict_shape(self):
        payload = IDCardResult().to_dict()
        assert set(payload) == {
            "card_detected",
            "card_confidence",
            "rotation_applied",
            "side",
            "fields",
            "derived",
            "validation",
            "warnings",
            "processing_time_ms",
        }
        assert set(payload["fields"]) == {
            "first_name",
            "last_name",
            "full_name",
            "address",
            "national_id",
        }
        assert set(payload["derived"]) == {"birth_date", "governorate", "gender"}

    def test_to_dict_is_json_serialisable(self):
        result = IDCardResult()
        result.first_name = Field("محمد", "محمد", 0.9, Status.OK)
        rendered = json.dumps(result.to_dict(), ensure_ascii=False)
        assert "محمد" in rendered

    def test_flat_dict_uses_none_for_unread_fields(self):
        flat = IDCardResult().to_flat_dict()
        assert set(flat) == {
            "first_name",
            "last_name",
            "full_name",
            "address",
            "national_id",
            "birth_date",
            "governorate",
            "gender",
        }
        assert all(value is None for value in flat.values())

    def test_repr_does_not_raise(self):
        assert "IDCardResult" in repr(IDCardResult())
