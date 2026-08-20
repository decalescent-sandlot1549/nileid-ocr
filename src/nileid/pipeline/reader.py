"""The end-to-end reader.

:class:`EgyptianIDReader` owns the sequence of stages and the policy about
what to do when a stage produces nothing. The controlling principle is that
an unsuccessful read must be *visible*: fields that could not be recognised
are ``None`` with a status explaining why, and every degradation encountered
along the way is appended to ``result.warnings``.

Pipeline stages::

    load -> assess quality -> locate card -> normalise orientation
         -> deskew -> detect fields -> read text / read digits
         -> normalise Arabic -> validate & decode the National ID
"""

from __future__ import annotations

import time

import numpy as np

from nileid.config import DEFAULT_SETTINGS, Settings
from nileid.detection.card import detect_card, orientation_by_layout
from nileid.detection.digits import read_digits
from nileid.detection.fields import (
    FieldBox,
    best_per_field,
    detect_fields,
    detect_regions,
    merge_detections,
)
from nileid.extraction.arabic import normalize_arabic
from nileid.extraction.lexicon import correct_text
from nileid.logging_utils import get_logger
from nileid.models import missing_models
from nileid.ocr.engines import recognise
from nileid.preprocessing.enhance import assess_quality, enhance_for_text, pad
from nileid.preprocessing.geometry import deskew, resize_to_card
from nileid.preprocessing.image_io import ImageInput, expand_box, load_image, safe_crop
from nileid.results import Field, IDCardResult, Status
from nileid.validation.national_id import decode_national_id, validate_national_id

log = get_logger("pipeline.reader")

#: Fields read as Arabic text (as opposed to detected digit by digit).
TEXT_FIELDS = ("first_name", "last_name", "address")


class EgyptianIDReader:
    """Reads structured data from an Egyptian National ID card image.

    Models load lazily on the first :meth:`read` call and are then reused
    for the lifetime of the process, so a single reader should be created
    once and shared.

    Example:
        >>> reader = EgyptianIDReader()                      # doctest: +SKIP
        >>> result = reader.read("card.jpg")                 # doctest: +SKIP
        >>> result.national_id.value                         # doctest: +SKIP
        '29001010100017'
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or DEFAULT_SETTINGS

    # ── Public API ───────────────────────────────────────────────────

    def read(self, source: ImageInput) -> IDCardResult:
        """Read one image and return a structured result.

        Never raises for an unreadable *card*: an image with no card in it
        comes back with ``card_detected=False`` and an explanatory warning.
        Raises :class:`~nileid.preprocessing.image_io.ImageLoadError` only
        when the input is not a decodable image at all, and
        :class:`~nileid.models.ModelNotFoundError` when weights are absent.
        """
        started = time.perf_counter()
        result = IDCardResult()

        image = load_image(source)
        self._check_models(result)

        quality = assess_quality(image)
        for note in quality.warnings():
            result.warn(note)

        card = self._locate_card(image, result)
        if card is None:
            result.processing_time_ms = (time.perf_counter() - started) * 1000
            return result

        detections = self._detect(card, result)
        self._read_text_fields(card, detections, result)
        self._read_national_id(card, detections, result)
        self._finalise_names(result)
        self._validate(result)

        if result.is_empty:
            result.warn("A card was detected but no field could be read.")

        result.processing_time_ms = (time.perf_counter() - started) * 1000
        return result

    def read_batch(self, sources: list[ImageInput]) -> list[IDCardResult]:
        """Read several images, reusing the loaded models."""
        return [self.read(source) for source in sources]

    # ── Stages ───────────────────────────────────────────────────────

    def _check_models(self, result: IDCardResult) -> None:
        absent = missing_models(self.settings)
        if absent:
            result.warn(
                "Missing model weights: "
                + ", ".join(absent)
                + ". Run: python scripts/download_models.py"
            )

    def _locate_card(self, image: np.ndarray, result: IDCardResult) -> np.ndarray | None:
        """Locate and upright the card, or fall back to the whole frame."""
        detection = detect_card(image, self.settings)

        if detection is not None:
            result.card_detected = True
            result.card_confidence = detection.confidence
            result.side = detection.side
            result.rotation_applied = detection.rotation
            if detection.confidence < self.settings.card_conf_warn:
                result.warn(
                    f"Card detected with low confidence ({detection.confidence:.2f}); "
                    "results may be unreliable."
                )
            if detection.side == "back":
                result.warn(
                    "The back of the card was detected. Name, address and the "
                    "National ID are printed on the front."
                )
            card = detection.image
        else:
            # No card box. Rather than refusing outright -- the original
            # behaviour, which rejected any card below 0.50 confidence --
            # fall back to treating the frame as an already-cropped card.
            # The caller can tell the difference from `card_detected`.
            result.warn(
                "No ID card outline was detected; processing the image as if it "
                "were already cropped to the card."
            )
            card, rotation = orientation_by_layout(image, self.settings)
            result.rotation_applied = rotation
            card = resize_to_card(card)

        straightened, angle = deskew(card)
        if angle:
            log.debug("Deskewed by %.2f degrees.", angle)
        return straightened

    def _detect(self, card: np.ndarray, result: IDCardResult) -> dict[str, FieldBox]:
        """Detect field regions, combining both detectors."""
        primary = detect_fields(card, self.settings)
        detections = best_per_field(primary)

        if any(d.flagged_invalid for d in primary):
            flagged = sorted({d.field for d in primary if d.flagged_invalid})
            result.warn(
                "The detector flagged these regions as atypical for a genuine "
                "card: " + ", ".join(flagged) + ". They were still read."
            )

        # The secondary detector only fills gaps.
        missing = [f for f in (*TEXT_FIELDS, "nid") if f not in detections]
        if missing:
            try:
                secondary = best_per_field(detect_regions(card, self.settings))
                detections = merge_detections(detections, secondary)
            except Exception:
                log.exception("Secondary region detector failed; continuing without it.")

        found = sorted(detections)
        log.debug("Detected fields: %s", ", ".join(found) if found else "none")
        return detections

    def _read_text_fields(
        self, card: np.ndarray, detections: dict[str, FieldBox], result: IDCardResult
    ) -> None:
        for name in TEXT_FIELDS:
            setattr(result, name, self._read_one_text_field(card, detections.get(name), name))

    def _read_one_text_field(
        self, card: np.ndarray, detection: FieldBox | None, field_name: str
    ) -> Field:
        if detection is None:
            return Field(status=Status.NOT_FOUND)

        scale_w, scale_h = detection.margin
        box = expand_box(detection.box, scale_w, scale_h, card.shape)
        crop = safe_crop(card, *box)
        if crop is None or crop.size == 0:
            return Field(status=Status.NOT_FOUND)

        prepared = enhance_for_text(pad(crop, border=20))
        ocr = recognise(prepared, self.settings)
        if ocr.is_empty:
            return Field(raw=None, confidence=0.0, status=Status.UNREADABLE)

        normalised = normalize_arabic(
            ocr.text,
            # Names must not carry stray Latin characters; an address
            # legitimately contains house numbers.
            drop_latin_tokens=field_name in ("first_name", "last_name"),
        )
        if self.settings.enable_fuzzy_correction:
            lexicon = "name" if field_name in ("first_name", "last_name") else "address"
            normalised = correct_text(normalised, lexicon, self.settings.fuzzy_threshold)

        if not normalised:
            return Field(raw=ocr.text, confidence=ocr.confidence, status=Status.UNREADABLE)

        status = (
            Status.LOW_CONFIDENCE if ocr.confidence < self.settings.low_confidence else Status.OK
        )
        return Field(value=normalised, raw=ocr.text, confidence=ocr.confidence, status=status)

    def _read_national_id(
        self, card: np.ndarray, detections: dict[str, FieldBox], result: IDCardResult
    ) -> None:
        detection = detections.get("nid")
        if detection is None:
            result.national_id = Field(status=Status.NOT_FOUND)
            result.warn("The National ID number region was not found on the card.")
            return

        scale_w, scale_h = detection.margin
        box = expand_box(detection.box, scale_w, scale_h, card.shape)
        crop = safe_crop(card, *box)
        if crop is None or crop.size == 0:
            result.national_id = Field(status=Status.NOT_FOUND)
            return

        read = read_digits(crop, self.settings)
        if not read.digits:
            result.national_id = Field(status=Status.UNREADABLE)
            result.warn("No digits could be recognised in the National ID region.")
            return

        if not read.is_complete:
            # A partial number is reported exactly as read, flagged, and
            # never padded or guessed at.
            result.warn(
                f"Read {len(read.digits)} of 14 National ID digits; the number is "
                "incomplete and has not been completed by inference."
            )

        status = (
            Status.OK
            if read.is_complete and read.confidence >= self.settings.low_confidence
            else Status.LOW_CONFIDENCE
        )
        result.national_id = Field(
            value=read.digits,
            raw=read.digits,
            confidence=read.confidence,
            status=status,
        )

    def _finalise_names(self, result: IDCardResult) -> None:
        """Compose the full name from the parts that were actually read."""
        parts = [f.value for f in (result.first_name, result.last_name) if f.value]
        if not parts:
            result.full_name = Field(status=result.first_name.status)
            return
        confidences = [f.confidence for f in (result.first_name, result.last_name) if f.value]
        confidence = min(confidences) if confidences else 0.0
        result.full_name = Field(
            value=" ".join(parts),
            raw=" ".join(
                f.raw or "" for f in (result.first_name, result.last_name) if f.raw
            ).strip()
            or None,
            confidence=confidence,
            status=(
                Status.OK
                if len(parts) == 2 and confidence >= self.settings.low_confidence
                else Status.LOW_CONFIDENCE
            ),
        )

    def _validate(self, result: IDCardResult) -> None:
        """Validate the National ID and decode what it encodes."""
        number = result.national_id.value
        result.validation = validate_national_id(number)

        if result.validation.checksum_ok is False:
            result.warn(
                "The National ID check digit does not match. The official "
                "algorithm is unpublished, so this is advisory -- but it often "
                "indicates a misread digit."
            )

        decoded = decode_national_id(number)
        if decoded is None:
            if number:
                result.warn(
                    "Birth date, governorate and gender were not derived because "
                    "the National ID did not pass validation."
                )
            return

        result.birth_date = decoded.birth_date_iso
        result.governorate = decoded.governorate
        result.gender = decoded.gender
