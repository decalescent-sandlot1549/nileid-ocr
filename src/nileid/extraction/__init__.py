"""Arabic text normalisation and optional lexicon-assisted correction."""

from nileid.extraction.arabic import (
    contains_arabic,
    normalize_arabic,
    to_display_form,
    to_western_digits,
)
from nileid.extraction.lexicon import correct_text, suggest_correction

__all__ = [
    "contains_arabic",
    "correct_text",
    "normalize_arabic",
    "suggest_correction",
    "to_display_form",
    "to_western_digits",
]
