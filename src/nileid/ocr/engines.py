"""OCR engine adapters and the engine ensemble.

Each adapter returns a uniform :class:`OCRResult` so the rest of the
pipeline does not care which engine produced a line. EasyOCR is the
default; PaddleOCR is optional and contributes only when installed.

Reading order matters for Arabic. Both engines return one entry per
detected text box in arbitrary order, so the boxes are sorted top-to-bottom
and then right-to-left before being joined -- joining in the engine's own
order scrambles a two-column address.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nileid.config import Settings
from nileid.logging_utils import get_logger
from nileid.models import get_easyocr, get_paddleocr

log = get_logger("ocr.engines")

#: Two boxes whose vertical centres are closer than this fraction of the
#: image height are treated as the same text line.
LINE_TOLERANCE = 0.45


@dataclass
class OCRResult:
    """Text recognised from one image, with provenance."""

    text: str
    confidence: float
    engine: str
    #: Individual box confidences, useful for diagnosing a partial read.
    box_confidences: list[float] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _order_boxes(
    items: list[tuple[list, str, float]], image_height: int
) -> list[tuple[str, float]]:
    """Sort recognised boxes into Arabic reading order.

    Lines run top to bottom; within a line, text runs right to left.
    """
    entries = []
    for box, text, confidence in items:
        points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        cy = float(points[:, 1].mean())
        cx = float(points[:, 0].mean())
        entries.append((cy, cx, text, confidence))

    if not entries:
        return []

    tolerance = max(8.0, image_height * LINE_TOLERANCE / max(1, len(entries)))
    entries.sort(key=lambda e: e[0])

    ordered: list[tuple[str, float]] = []
    line: list[tuple[float, float, str, float]] = []
    line_y = entries[0][0]
    for entry in entries:
        if abs(entry[0] - line_y) > tolerance and line:
            # Right-to-left within the completed line.
            line.sort(key=lambda e: -e[1])
            ordered.extend((t, c) for _, _, t, c in line)
            line = []
            line_y = entry[0]
        line.append(entry)
    if line:
        line.sort(key=lambda e: -e[1])
        ordered.extend((t, c) for _, _, t, c in line)
    return ordered


def run_easyocr(image: np.ndarray, settings: Settings) -> OCRResult:
    """Recognise text with EasyOCR."""
    reader = get_easyocr(settings)
    if reader is None:
        return OCRResult("", 0.0, "easyocr")
    try:
        raw = reader.readtext(image, detail=1)
    except Exception:
        log.exception("EasyOCR failed on a crop.")
        return OCRResult("", 0.0, "easyocr")

    items = [
        (entry[0], str(entry[1]), float(entry[2]))
        for entry in raw
        if entry and len(entry) >= 3 and entry[1]
    ]
    if not items:
        return OCRResult("", 0.0, "easyocr")

    ordered = _order_boxes(items, image.shape[0])
    confidences = [c for _, c in ordered]
    return OCRResult(
        text=" ".join(t for t, _ in ordered).strip(),
        confidence=float(np.mean(confidences)) if confidences else 0.0,
        engine="easyocr",
        box_confidences=confidences,
    )


def run_paddleocr(image: np.ndarray, settings: Settings) -> OCRResult:
    """Recognise text with PaddleOCR, when it is installed."""
    engine = get_paddleocr(settings)
    if engine is None:
        return OCRResult("", 0.0, "paddleocr")
    try:
        raw = engine.ocr(image)
    except Exception:
        log.exception("PaddleOCR failed on a crop.")
        return OCRResult("", 0.0, "paddleocr")

    items: list[tuple[list, str, float]] = []
    for page in raw or []:
        for line in page or []:
            # Expected shape: [box, (text, confidence)]
            if not line or len(line) < 2:
                continue
            box, info = line[0], line[1]
            if not info or len(info) < 2 or not info[0]:
                continue
            items.append((box, str(info[0]), float(info[1])))

    if not items:
        return OCRResult("", 0.0, "paddleocr")

    ordered = _order_boxes(items, image.shape[0])
    confidences = [c for _, c in ordered]
    return OCRResult(
        text=" ".join(t for t, _ in ordered).strip(),
        confidence=float(np.mean(confidences)) if confidences else 0.0,
        engine="paddleocr",
        box_confidences=confidences,
    )


_ENGINES = {"easyocr": run_easyocr, "paddleocr": run_paddleocr}


def recognise(image: np.ndarray, settings: Settings) -> OCRResult:
    """Run the configured engines and return the most credible reading.

    Selection is by engine confidence, with a small preference for the
    longer string when confidences are close -- a truncated read is the
    more common failure than a hallucinated one. The original scoring
    multiplied a character count by confidence, which let a long, wrong
    reading beat a short, correct one outright.
    """
    if image is None or image.size == 0:
        return OCRResult("", 0.0, "none")

    results: list[OCRResult] = []
    for name in settings.ocr_engines:
        runner = _ENGINES.get(name)
        if runner is None:
            log.warning("Unknown OCR engine %r in configuration; skipping.", name)
            continue
        result = runner(image, settings)
        if not result.is_empty:
            results.append(result)

    if not results:
        return OCRResult("", 0.0, "none")
    if len(results) == 1:
        return results[0]

    def rank(result: OCRResult) -> tuple[float, int]:
        return (round(result.confidence, 2), len(result.text))

    best = max(results, key=rank)
    log.debug(
        "Engine selection: %s",
        ", ".join(f"{r.engine}={r.confidence:.2f}/{len(r.text)}c" for r in results),
    )
    return best
