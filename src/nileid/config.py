"""Runtime configuration for the NileID pipeline.

All tunable behaviour is collected here so that thresholds and paths are
declared in one place instead of being scattered as literals through the
detection and OCR code. Values may be overridden through environment
variables (see ``.env.example``) or by constructing :class:`Settings`
directly and passing it to :class:`nileid.pipeline.EgyptianIDReader`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Canonical size the card is normalised to before field detection.
# The Egyptian National ID has a 1.585:1 aspect ratio (ID-1, 85.6 x 54 mm);
# 1000 x 630 keeps that ratio while giving OCR a comfortable pixel budget.
CARD_WIDTH = 1000
CARD_HEIGHT = 630

# Weight files expected inside the model directory.
MODEL_FILES: dict[str, str] = {
    "card": "detect_id_card.pt",
    "fields": "detect_odjects.pt",
    "digits": "detect_id.pt",
    "regions": "best.pt",
}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable pipeline configuration."""

    # ── Paths & device ───────────────────────────────────────────────
    model_dir: Path = field(default_factory=lambda: Path(_env_str("NILEID_MODEL_DIR", "models")))
    device: str = field(default_factory=lambda: _env_str("NILEID_DEVICE", "auto"))

    # ── Detection thresholds ─────────────────────────────────────────
    #: Minimum confidence for accepting a card detection. The original
    #: implementation used 0.50, which rejected legible but atypical cards
    #: outright; 0.25 accepts more cards and the result carries a warning
    #: when the detection is weak.
    card_conf: float = field(default_factory=lambda: _env_float("NILEID_CARD_CONF", 0.25))
    #: Below this the card detection is reported as low confidence.
    card_conf_warn: float = 0.45
    #: Minimum confidence for accepting a field (name/address/NID) box.
    field_conf: float = field(default_factory=lambda: _env_float("NILEID_FIELD_CONF", 0.20))
    #: Minimum confidence for accepting a single digit box of the NID.
    digit_conf: float = field(default_factory=lambda: _env_float("NILEID_DIGIT_CONF", 0.25))
    #: IoU above which two digit boxes are considered the same digit.
    digit_iou: float = 0.40
    #: Smallest acceptable card crop, in pixels, to guard against
    #: detections that latch onto a logo or a single field.
    min_card_height: int = 120
    min_card_width: int = 200

    # ── OCR ──────────────────────────────────────────────────────────
    ocr_engines: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            e.strip() for e in _env_str("NILEID_OCR_ENGINES", "easyocr").split(",") if e.strip()
        )
    )
    ocr_languages: tuple[str, ...] = ("ar", "en")
    #: Field-level confidence below which a value is flagged as uncertain.
    low_confidence: float = 0.40

    # ── Arabic post-processing ───────────────────────────────────────
    #: Dictionary-based fuzzy correction is opt-in. It can only ever
    #: replace a token with a close lexicon entry, never invent one, but
    #: it is still a guess and is therefore disabled by default.
    enable_fuzzy_correction: bool = False
    #: Similarity (0-100) a token must reach before it is replaced.
    fuzzy_threshold: int = 92

    # ── API ──────────────────────────────────────────────────────────
    max_upload_mb: int = field(default_factory=lambda: _env_int("NILEID_MAX_UPLOAD_MB", 10))
    log_level: str = field(default_factory=lambda: _env_str("NILEID_LOG_LEVEL", "INFO"))

    def model_path(self, key: str) -> Path:
        """Absolute path of a weight file, by registry key."""
        try:
            filename = MODEL_FILES[key]
        except KeyError as exc:  # pragma: no cover - programming error
            raise KeyError(
                f"Unknown model key {key!r}. Expected one of {sorted(MODEL_FILES)}"
            ) from exc
        return self.model_dir / filename

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


#: Default settings instance used when no explicit configuration is given.
DEFAULT_SETTINGS = Settings()
