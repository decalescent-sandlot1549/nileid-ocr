"""Image loading, quality assessment and geometric normalisation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nileid.preprocessing.enhance import (
    assess_quality,
    enhance_for_digits,
    enhance_for_text,
    pad,
)
from nileid.preprocessing.geometry import (
    correct_perspective,
    deskew,
    estimate_skew,
    find_card_quadrilateral,
    four_point_transform,
    order_corners,
    resize_to_card,
    rotate_bound,
    rotate_quarter_turns,
)
from nileid.preprocessing.image_io import (
    ImageLoadError,
    expand_box,
    load_image,
    safe_crop,
    to_bgr,
)


class TestLoading:
    def test_loads_from_path(self, sample_card_png):
        image = load_image(sample_card_png)
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_loads_from_str_path(self, sample_card_png):
        assert load_image(str(sample_card_png)).ndim == 3

    def test_loads_from_bytes(self, sample_card_bytes):
        assert load_image(sample_card_bytes).shape[2] == 3

    def test_loads_from_array(self, sample_card_bgr):
        assert load_image(sample_card_bgr).shape == sample_card_bgr.shape

    def test_array_input_is_copied(self, sample_card_bgr):
        loaded = load_image(sample_card_bgr)
        loaded[0, 0] = (1, 2, 3)
        assert not np.array_equal(loaded[0, 0], sample_card_bgr[0, 0])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ImageLoadError, match="File not found"):
            load_image(tmp_path / "nope.png")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ImageLoadError):
            load_image(tmp_path)

    def test_empty_bytes_raises(self):
        with pytest.raises(ImageLoadError, match="empty"):
            load_image(b"")

    def test_non_image_bytes_raises(self):
        with pytest.raises(ImageLoadError, match="decode"):
            load_image(b"this is definitely not a PNG")

    def test_truncated_image_raises(self, sample_card_bytes):
        with pytest.raises(ImageLoadError):
            load_image(sample_card_bytes[:40])

    def test_unsupported_type_raises(self):
        with pytest.raises(ImageLoadError, match="Unsupported input type"):
            load_image(12345)  # type: ignore[arg-type]

    def test_empty_array_raises(self):
        with pytest.raises(ImageLoadError):
            load_image(np.empty((0, 0, 3), dtype=np.uint8))

    def test_text_file_with_image_extension_raises(self, tmp_path):
        decoy = tmp_path / "not_really.png"
        decoy.write_text("plain text pretending to be an image")
        with pytest.raises(ImageLoadError):
            load_image(decoy)


class TestColourNormalisation:
    def test_grayscale_becomes_three_channel(self, sample_card_bgr):
        gray = cv2.cvtColor(sample_card_bgr, cv2.COLOR_BGR2GRAY)
        assert to_bgr(gray).shape[2] == 3

    def test_alpha_channel_is_dropped(self, sample_card_bgr):
        bgra = cv2.cvtColor(sample_card_bgr, cv2.COLOR_BGR2BGRA)
        assert to_bgr(bgra).shape[2] == 3

    def test_float_image_is_rescaled(self):
        float_image = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        converted = to_bgr(float_image)
        assert converted.dtype == np.uint8
        assert converted.max() > 1


class TestCropping:
    def test_clamps_to_bounds(self, sample_card_bgr):
        crop = safe_crop(sample_card_bgr, -100, -100, 50, 50)
        assert crop is not None and crop.shape[:2] == (50, 50)

    def test_returns_none_for_degenerate_box(self, sample_card_bgr):
        assert safe_crop(sample_card_bgr, 10, 10, 10, 10) is None

    def test_returns_none_for_inverted_box(self, sample_card_bgr):
        assert safe_crop(sample_card_bgr, 200, 200, 100, 100) is None

    def test_fully_outside_box(self, sample_card_bgr):
        h, w = sample_card_bgr.shape[:2]
        assert safe_crop(sample_card_bgr, w + 10, h + 10, w + 50, h + 50) is None

    def test_expand_box_grows_and_clamps(self, sample_card_bgr):
        x1, y1, x2, y2 = expand_box((100, 100, 200, 150), 2.0, 2.0, sample_card_bgr.shape)
        assert x2 - x1 > 100 and y2 - y1 > 50
        assert x1 >= 0 and y1 >= 0

    def test_expand_box_never_exceeds_image(self, sample_card_bgr):
        h, w = sample_card_bgr.shape[:2]
        x1, y1, x2, y2 = expand_box((0, 0, w, h), 3.0, 3.0, sample_card_bgr.shape)
        assert x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h


class TestQuality:
    def test_reports_all_metrics(self, sample_card_bgr):
        quality = assess_quality(sample_card_bgr)
        assert quality.blur > 0
        assert 0 <= quality.brightness <= 255
        assert quality.noise >= 0

    def test_flags_a_blurred_image(self, sample_card_bgr):
        blurred = cv2.GaussianBlur(sample_card_bgr, (0, 0), 6)
        assert assess_quality(blurred).is_blurred

    def test_does_not_flag_a_sharp_image(self, sample_card_bgr):
        assert not assess_quality(sample_card_bgr).is_blurred

    def test_flags_a_dark_image(self, sample_card_bgr):
        dark = (sample_card_bgr * 0.15).astype(np.uint8)
        quality = assess_quality(dark)
        assert quality.is_dark and quality.warnings()

    def test_flags_low_contrast(self):
        flat = np.full((200, 300, 3), 128, dtype=np.uint8)
        assert assess_quality(flat).is_low_contrast

    def test_warnings_are_strings(self, sample_card_bgr):
        assert all(isinstance(w, str) for w in assess_quality(sample_card_bgr).warnings())


class TestEnhancement:
    def test_text_enhancement_returns_three_channels(self, sample_card_bgr):
        crop = sample_card_bgr[150:200, 700:960]
        assert enhance_for_text(crop).shape[2] == 3

    def test_text_enhancement_upscales_small_crops(self, sample_card_bgr):
        crop = sample_card_bgr[150:170, 700:960]  # 20px tall
        assert enhance_for_text(crop).shape[0] > crop.shape[0]

    def test_text_enhancement_handles_a_tiny_crop(self):
        assert enhance_for_text(np.zeros((3, 3, 3), dtype=np.uint8)) is not None

    def test_digit_enhancement_upscales(self, sample_card_bgr):
        strip = sample_card_bgr[490:560, 250:900]
        assert enhance_for_digits(strip).shape[0] > strip.shape[0]

    def test_digit_enhancement_caps_the_output_size(self):
        wide = np.zeros((100, 3800, 3), dtype=np.uint8)
        assert enhance_for_digits(wide, scale=8.0).shape[1] <= 4200

    def test_pad_adds_a_border(self, sample_card_bgr):
        padded = pad(sample_card_bgr, border=10)
        assert padded.shape[0] == sample_card_bgr.shape[0] + 20
        assert padded.shape[1] == sample_card_bgr.shape[1] + 20


class TestGeometry:
    def test_order_corners_sorts_consistently(self):
        scrambled = np.array([[100, 100], [0, 100], [100, 0], [0, 0]], dtype=np.float32)
        ordered = order_corners(scrambled)
        assert list(ordered[0]) == [0, 0]
        assert list(ordered[2]) == [100, 100]

    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_quarter_turns_preserve_pixel_count(self, sample_card_bgr, degrees):
        rotated = rotate_quarter_turns(sample_card_bgr, degrees)
        assert rotated.size == sample_card_bgr.size

    def test_quarter_turns_swap_axes_for_90(self, sample_card_bgr):
        h, w = sample_card_bgr.shape[:2]
        assert rotate_quarter_turns(sample_card_bgr, 90).shape[:2] == (w, h)

    def test_four_quarter_turns_return_the_original(self, sample_card_bgr):
        image = sample_card_bgr
        for _ in range(4):
            image = rotate_quarter_turns(image, 90)
        assert np.array_equal(image, sample_card_bgr)

    def test_rejects_non_quarter_rotation(self, sample_card_bgr):
        with pytest.raises(ValueError, match="multiple of 90"):
            rotate_quarter_turns(sample_card_bgr, 45)

    def test_resize_to_card_uses_the_canonical_size(self, sample_card_bgr):
        from nileid.config import CARD_HEIGHT, CARD_WIDTH

        assert resize_to_card(sample_card_bgr).shape[:2] == (CARD_HEIGHT, CARD_WIDTH)

    @pytest.mark.parametrize("angle", [-15, -12, -10, -6, -3, -1, 1, 3, 6, 10, 12, 15])
    def test_skew_estimate_recovers_a_known_rotation(self, sample_card_bgr, angle):
        """Rotating by A must be measured as approximately -A."""
        rotated = rotate_bound(sample_card_bgr, angle)
        measured = estimate_skew(rotated)
        assert abs(angle + measured) < 0.6, f"residual {angle + measured:.2f} deg"

    def test_straight_image_is_not_rotated(self, sample_card_bgr):
        _, angle = deskew(sample_card_bgr)
        assert angle == 0.0

    def test_deskew_returns_an_image_of_the_same_shape(self, sample_card_bgr):
        rotated = rotate_bound(sample_card_bgr, 5)
        corrected, _ = deskew(rotated)
        assert corrected.shape == rotated.shape

    def test_skew_guard_ignores_extreme_angles(self, sample_card_bgr):
        # Beyond the guard the estimator returns 0 rather than a wrong angle.
        assert estimate_skew(rotate_quarter_turns(sample_card_bgr, 90)) == 0.0


class TestPerspective:
    @staticmethod
    def _projected_card(card: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Place the card on a background under a projective transform."""
        height, width = card.shape[:2]
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        target = np.float32([[230, 120], [1120, 60], [1180, 760], [180, 690]])
        matrix = cv2.getPerspectiveTransform(source, target)
        scene = cv2.warpPerspective(card, matrix, (1300, 900), borderValue=(60, 60, 60))
        return scene, target

    def test_finds_the_card_outline(self, sample_card_bgr):
        scene, target = self._projected_card(sample_card_bgr)
        corners = find_card_quadrilateral(scene)
        assert corners is not None
        error = np.abs(np.sort(corners, axis=0) - np.sort(target, axis=0)).mean()
        assert error < 6.0, f"mean corner error {error:.1f}px"

    def test_corrects_the_perspective(self, sample_card_bgr):
        from nileid.config import CARD_HEIGHT, CARD_WIDTH

        scene, _ = self._projected_card(sample_card_bgr)
        corrected, applied = correct_perspective(scene)
        assert applied
        assert corrected.shape[:2] == (CARD_HEIGHT, CARD_WIDTH)

    def test_declines_when_there_is_no_card(self, blank_image):
        _, applied = correct_perspective(blank_image)
        assert not applied

    def test_returns_the_input_unchanged_when_declining(self, blank_image):
        result, applied = correct_perspective(blank_image)
        assert not applied
        assert np.array_equal(result, blank_image)

    def test_four_point_transform_output_size(self, sample_card_bgr):
        corners = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
        warped = four_point_transform(sample_card_bgr, corners, width=200, height=120)
        assert warped.shape[:2] == (120, 200)
