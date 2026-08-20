"""Detection helper logic that does not require model weights.

These cover the class-name handling and digit post-processing where the
original implementation had defects: the ``invalid_`` prefix silently
suppressing every field, and duplicate digit boxes lengthening the ID.
"""

from __future__ import annotations

import pytest

from nileid.detection.card import ORIENTATION_ROTATION, _parse_class_name
from nileid.detection.digits import _iou, _suppress_overlaps
from nileid.detection.fields import (
    FieldBox,
    best_per_field,
    canonical_field_name,
    is_flagged_invalid,
    merge_detections,
)


class TestCardClassNames:
    @pytest.mark.parametrize(
        ("raw", "side", "orientation"),
        [
            ("front-up", "front", "up"),
            ("front-left", "front", "left"),
            ("back-bottom", None, None),  # "bottom" is not an orientation we map
            ("back-down", "back", "down"),
            ("FRONT-UP", "front", "up"),
        ],
    )
    def test_parses_side_and_orientation(self, raw, side, orientation):
        assert _parse_class_name(raw) == (side, orientation)

    @pytest.mark.parametrize("raw", ["", "nonsense", "front", "front-up-extra", "left-front"])
    def test_rejects_unparseable_names(self, raw):
        assert _parse_class_name(raw) == (None, None)

    def test_rotation_table_covers_a_full_turn(self):
        assert sorted(ORIENTATION_ROTATION.values()) == [0, 90, 180, 270]


class TestFieldClassNames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("nid", "nid"),
            ("address", "address"),
            ("firstName", "first_name"),
            ("lastName", "last_name"),
            ("serial", "serial"),
            ("photo", "photo"),
        ],
    )
    def test_maps_clean_class_names(self, raw, expected):
        assert canonical_field_name(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("invalid_nid", "nid"),
            ("invalid_address", "address"),
            ("invalid_firstName", "first_name"),
            ("invalid_lastName", "last_name"),
        ],
    )
    def test_invalid_variants_map_to_the_same_field(self, raw, expected):
        """The original code compared names for equality, so an
        ``invalid_`` prefix silently produced an empty result."""
        assert canonical_field_name(raw) == expected

    def test_detects_the_invalid_flag(self):
        assert is_flagged_invalid("invalid_nid")
        assert not is_flagged_invalid("nid")

    def test_unknown_class_is_lowercased_not_dropped(self):
        assert canonical_field_name("SomethingNew") == "somethingnew"

    def test_empty_class_name(self):
        assert canonical_field_name("") == ""


class TestBestPerField:
    @staticmethod
    def _box(field: str, confidence: float) -> FieldBox:
        return FieldBox(field=field, box=(0, 0, 10, 10), confidence=confidence, raw_class=field)

    def test_keeps_the_most_confident_detection(self):
        boxes = [self._box("nid", 0.3), self._box("nid", 0.9), self._box("nid", 0.5)]
        assert best_per_field(boxes)["nid"].confidence == 0.9

    def test_ignores_unnamed_fields(self):
        assert best_per_field([self._box("", 0.9)]) == {}

    def test_empty_input(self):
        assert best_per_field([]) == {}

    def test_merge_prefers_primary(self):
        primary = {"nid": self._box("nid", 0.4)}
        secondary = {"nid": self._box("nid", 0.99), "address": self._box("address", 0.5)}
        merged = merge_detections(primary, secondary)
        assert merged["nid"].confidence == 0.4  # primary wins even when weaker
        assert "address" in merged  # secondary fills the gap


class TestDigitPostProcessing:
    def test_iou_of_identical_boxes_is_one(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert _iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_iou_of_partial_overlap(self):
        assert 0 < _iou((0, 0, 10, 10), (5, 0, 15, 10)) < 1

    def test_suppression_collapses_duplicate_boxes(self):
        """Two classes firing on one glyph must not yield two digits."""
        candidates = [
            (0.0, 5, 0.60, (0.0, 0.0, 10.0, 20.0)),
            (0.5, 6, 0.90, (0.5, 0.0, 10.5, 20.0)),  # same glyph, more confident
            (40.0, 3, 0.80, (40.0, 0.0, 50.0, 20.0)),
        ]
        kept = _suppress_overlaps(candidates, 0.40)
        assert len(kept) == 2
        assert {c[1] for c in kept} == {6, 3}

    def test_suppression_keeps_adjacent_digits(self):
        candidates = [
            (0.0, 1, 0.9, (0.0, 0.0, 10.0, 20.0)),
            (11.0, 2, 0.9, (11.0, 0.0, 21.0, 20.0)),
            (22.0, 3, 0.9, (22.0, 0.0, 32.0, 20.0)),
        ]
        assert len(_suppress_overlaps(candidates, 0.40)) == 3

    def test_suppression_prefers_the_highest_confidence(self):
        candidates = [
            (0.0, 1, 0.30, (0.0, 0.0, 10.0, 20.0)),
            (0.1, 7, 0.95, (0.1, 0.0, 10.1, 20.0)),
        ]
        kept = _suppress_overlaps(candidates, 0.40)
        assert len(kept) == 1 and kept[0][1] == 7

    def test_empty_input(self):
        assert _suppress_overlaps([], 0.4) == []
