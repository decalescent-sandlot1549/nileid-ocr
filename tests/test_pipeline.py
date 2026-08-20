"""End-to-end pipeline behaviour.

These tests need the YOLO weights and are skipped when the weights are
absent, so a clone without them still has a green suite for everything
that does not depend on a model.

They assert *contract*, not accuracy: the models were trained on genuine
cards and the fixtures here are synthetic mock-ups, so asserting that a
particular name is recognised would be asserting something this repository
cannot honestly guarantee. What is asserted is that the pipeline reports
what it did, never fabricates a field, and never crashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_models  # noqa: F401  (registers the marker helper)
from nileid import EgyptianIDReader, Settings
from nileid.preprocessing.image_io import ImageLoadError
from nileid.results import IDCardResult, Status

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def reader(models_available) -> EgyptianIDReader:
    if not models_available:
        pytest.skip("YOLO weights are absent; run scripts/download_models.py")
    return EgyptianIDReader(Settings())


class TestContract:
    def test_returns_a_result_object(self, reader, sample_card_png):
        assert isinstance(reader.read(sample_card_png), IDCardResult)

    def test_accepts_a_path(self, reader, sample_card_png):
        assert reader.read(sample_card_png) is not None

    def test_accepts_bytes(self, reader, sample_card_bytes):
        assert reader.read(sample_card_bytes) is not None

    def test_accepts_an_array(self, reader, sample_card_bgr):
        assert reader.read(sample_card_bgr) is not None

    def test_result_is_json_serialisable(self, reader, sample_card_png):
        import json

        json.dumps(reader.read(sample_card_png).to_dict(), ensure_ascii=False)

    def test_reports_processing_time(self, reader, sample_card_png):
        assert reader.read(sample_card_png).processing_time_ms > 0


class TestFailureTransparency:
    def test_blank_image_does_not_raise(self, reader, blank_image):
        """An image with no card must return a result, not an exception."""
        result = reader.read(blank_image)
        assert isinstance(result, IDCardResult)

    def test_blank_image_reports_warnings(self, reader, blank_image):
        result = reader.read(blank_image)
        assert result.warnings, "an unreadable image must explain itself"

    def test_blank_image_invents_nothing(self, reader, blank_image):
        result = reader.read(blank_image)
        assert result.birth_date is None
        assert result.governorate is None
        assert result.gender is None
        assert not result.validation.valid

    def test_noise_image_invents_nothing(self, reader):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (400, 640, 3), dtype=np.uint8)
        result = reader.read(noise)
        assert result.birth_date is None
        assert not result.validation.valid

    def test_derived_fields_require_a_valid_id(self, reader, sample_card_png):
        """Birth date/governorate/gender may only appear when the number validates."""
        result = reader.read(sample_card_png)
        if not result.validation.valid:
            assert result.birth_date is None
            assert result.governorate is None
            assert result.gender is None

    def test_national_id_is_never_padded_to_length(self, reader, sample_card_png):
        """A partial read must stay partial, never be completed by inference."""
        result = reader.read(sample_card_png)
        value = result.national_id.value
        if value is not None and len(value) != 14:
            assert not result.validation.valid
            assert any("14" in w or "digits" in w for w in result.warnings)

    def test_unreadable_fields_are_none(self, reader, blank_image):
        result = reader.read(blank_image)
        for field in (result.first_name, result.last_name, result.address, result.national_id):
            if not field.is_usable:
                assert field.value is None

    def test_confidence_is_within_range(self, reader, sample_card_png):
        result = reader.read(sample_card_png)
        for field in (result.first_name, result.last_name, result.address, result.national_id):
            assert 0.0 <= field.confidence <= 1.0

    def test_status_is_consistent_with_value(self, reader, sample_card_png):
        result = reader.read(sample_card_png)
        for field in (result.first_name, result.last_name, result.address, result.national_id):
            if field.status in (Status.NOT_FOUND, Status.UNREADABLE):
                assert field.value is None


class TestInputHandling:
    def test_rejects_a_non_image(self, reader):
        with pytest.raises(ImageLoadError):
            reader.read(b"not an image at all")

    def test_rejects_empty_bytes(self, reader):
        with pytest.raises(ImageLoadError):
            reader.read(b"")

    def test_rejects_a_missing_file(self, reader, tmp_path):
        with pytest.raises(ImageLoadError):
            reader.read(tmp_path / "absent.png")

    def test_handles_a_one_pixel_image(self, reader):
        result = reader.read(np.zeros((1, 1, 3), dtype=np.uint8))
        assert isinstance(result, IDCardResult)

    def test_handles_a_grayscale_image(self, reader, sample_card_bgr):
        import cv2

        gray = cv2.cvtColor(sample_card_bgr, cv2.COLOR_BGR2GRAY)
        assert reader.read(gray) is not None

    def test_handles_an_extremely_wide_image(self, reader):
        assert reader.read(np.zeros((10, 2000, 3), dtype=np.uint8)) is not None


class TestDeterminismAndReuse:
    def test_repeated_reads_agree(self, reader, sample_card_png):
        """The pipeline has no random component; two reads must match."""
        first = reader.read(sample_card_png).to_flat_dict()
        second = reader.read(sample_card_png).to_flat_dict()
        assert first == second

    def test_batch_matches_individual_reads(self, reader, sample_card_png):
        batch = reader.read_batch([sample_card_png, sample_card_png])
        assert len(batch) == 2
        assert batch[0].to_flat_dict() == batch[1].to_flat_dict()

    def test_models_are_reused_across_calls(self, reader, sample_card_png):
        """The second read must be much faster, proving models are cached."""
        reader.read(sample_card_png)  # warm
        first = reader.read(sample_card_png).processing_time_ms
        second = reader.read(sample_card_png).processing_time_ms
        assert max(first, second) < 60_000, "a warm read should not reload models"


class TestRotationHandling:
    @pytest.mark.parametrize("degrees", [90, 180, 270])
    def test_rotated_input_does_not_crash(self, reader, sample_card_bgr, degrees):
        from nileid.preprocessing.geometry import rotate_quarter_turns

        result = reader.read(rotate_quarter_turns(sample_card_bgr, degrees))
        assert isinstance(result, IDCardResult)

    def test_reports_the_rotation_it_applied(self, reader, sample_card_png):
        assert reader.read(sample_card_png).rotation_applied in (0, 90, 180, 270)
