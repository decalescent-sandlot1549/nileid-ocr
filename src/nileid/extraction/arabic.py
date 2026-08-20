"""Arabic text post-processing for OCR output.

Design rules, in priority order:

1. **Never invent text.** Every transform here either removes noise that
   OCR added or rewrites a token into a spelling variant of *itself*.
   Nothing may substitute a word the engine did not see.
2. **Keep the raw string.** Normalisation is lossy; callers receive both
   the raw and the normalised form (see :class:`nileid.results.Field`).
3. **Be conservative near meaning.** Orthographic repairs are restricted to
   cases that are unambiguous in Egyptian civil-registry text.

Note on direction: OCR engines return Arabic in *logical* order, which is
the correct order for storage, JSON and databases. Visual reordering is a
rendering concern and is provided separately by :func:`to_display_form`.
"""

from __future__ import annotations

import re
import unicodedata

from nileid.logging_utils import get_logger

log = get_logger("extraction.arabic")

# ── Character classes ────────────────────────────────────────────────
ARABIC_RANGES = (
    r"؀-ۿ"  # Arabic
    r"ݐ-ݿ"  # Arabic Supplement
    r"ࢠ-ࣿ"  # Arabic Extended-A
    r"ﭐ-﷿"  # Arabic Presentation Forms-A
    r"ﹰ-﻿"  # Arabic Presentation Forms-B
)
_RE_ARABIC_CHAR = re.compile(f"[{ARABIC_RANGES}]")

#: Harakat (short vowels), tatweel and other combining marks that carry no
#: information in printed civil-registry text and confuse string matching.
_RE_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_RE_TATWEEL = re.compile(r"ـ+")

#: Characters OCR commonly hallucinates around text boxes.
_RE_NOISE = re.compile(r"[|\[\]{}<>_=+*^~\\/©®°¦¬]")
_RE_WHITESPACE = re.compile(r"\s+")

#: A token that is pure Latin/punctuation noise inside an Arabic field.
_RE_LATIN_ONLY = re.compile(r"^[A-Za-z\W\d]+$")

_ARABIC_INDIC = {ord("٠") + i: str(i) for i in range(10)}
_EXT_ARABIC_INDIC = {ord("۰") + i: str(i) for i in range(10)}
DIGIT_TRANSLATION = {**_ARABIC_INDIC, **_EXT_ARABIC_INDIC}

# ── Orthographic repairs ─────────────────────────────────────────────
# Only entries where the left-hand side is an unambiguous mis-spelling of
# the right-hand side. Each is a spelling variant of the same word: hamza
# restoration, ta-marbuta, alif-maqsura, or a compound-name split.
#
# Entries that *guessed at a different word* were removed from the original
# lexicon; a rare token must survive OCR unchanged rather than be replaced
# by a common one that happens to look similar.
ORTHOGRAPHIC_FIXES: dict[str, str] = {
    # Hamza on initial alif
    "احمد": "أحمد",
    "ابراهيم": "إبراهيم",
    "اسماعيل": "إسماعيل",
    "اسلام": "إسلام",
    "ايمن": "أيمن",
    "امين": "أمين",
    "انس": "أنس",
    "انور": "أنور",
    "اشرف": "أشرف",
    "الهام": "إلهام",
    "ابو": "أبو",
    "اسيوط": "أسيوط",
    # Ta marbuta
    "اسامه": "أسامة",
    "اسامة": "أسامة",
    "امنه": "آمنة",
    "اميره": "أميرة",
    "سميه": "سمية",
    "فاطمه": "فاطمة",
    "عائشه": "عائشة",
    "خديجه": "خديجة",
    "ناديه": "نادية",
    "القاهره": "القاهرة",
    "الاسكندريه": "الإسكندرية",
    "الجيزه": "الجيزة",
    "القليوبيه": "القليوبية",
    "المنصوره": "المنصورة",
    "الاسماعيليه": "الإسماعيلية",
    "البحيره": "البحيرة",
    "الدقهليه": "الدقهلية",
    "الشرقيه": "الشرقية",
    "الغربيه": "الغربية",
    "المنوفيه": "المنوفية",
    "قريه": "قرية",
    "عزبه": "عزبة",
    "ناحيه": "ناحية",
    # Alif maqsura
    "مصطفي": "مصطفى",
    "موسي": "موسى",
    "يحيي": "يحيى",
    "سلوي": "سلوى",
    "ليلي": "ليلى",
    # Compound names written without the space
    "عبدالله": "عبد الله",
    "عبدالرحمن": "عبد الرحمن",
    "عبدالعزيز": "عبد العزيز",
    "كفرالشيخ": "كفر الشيخ",
    "بنيسويف": "بني سويف",
}

#: Definite article split off from its noun by the text detector, e.g.
#: "ال قاهرة" -> "القاهرة".
_RE_SPLIT_AL = re.compile(r"(^|\s)ال\s+(?=[ء-ي])")

#: Words that legitimately end in a bare "ه" and must not be rewritten.
_TA_MARBUTA_EXCEPTIONS = frozenset({"الله", "لله", "عبده", "الاله", "الإله", "فيه", "منه", "له"})


def contains_arabic(text: str) -> bool:
    """True when the string contains at least one Arabic character."""
    return bool(text) and bool(_RE_ARABIC_CHAR.search(text))


def to_western_digits(text: str) -> str:
    """Convert Arabic-Indic digits to ASCII digits, leaving letters alone."""
    return text.translate(DIGIT_TRANSLATION) if text else text


def strip_diacritics(text: str) -> str:
    """Remove harakat and tatweel."""
    return _RE_TATWEEL.sub("", _RE_DIACRITICS.sub("", text)) if text else text


def clean_ocr_noise(text: str) -> str:
    """Drop bracket/pipe artefacts and collapse whitespace."""
    if not text:
        return ""
    text = _RE_NOISE.sub(" ", text)
    return _RE_WHITESPACE.sub(" ", text).strip()


def _fix_ta_marbuta(word: str) -> str:
    """Rewrite a final bare ``ه`` as ``ة`` where that is unambiguous.

    Applied only to words of at least four letters that are not in the
    exception list and contain no digits, which covers the common OCR
    confusion without touching pronouns or the divine name.
    """
    if (
        len(word) >= 4
        and word.endswith("ه")
        and word not in _TA_MARBUTA_EXCEPTIONS
        and not any(ch.isdigit() for ch in word)
        and not any(word.endswith(suffix) for suffix in ("الله", "لله"))
    ):
        return word[:-1] + "ة"
    return word


def merge_split_article(text: str) -> str:
    """Rejoin a definite article that OCR separated from its noun."""
    if not text:
        return text
    previous = None
    # Iterate because one pass only fixes non-overlapping matches.
    while previous != text:
        previous = text
        text = _RE_SPLIT_AL.sub(lambda m: m.group(1) + "ال", text)
    return text


def normalize_arabic(
    text: str | None,
    *,
    drop_latin_tokens: bool = False,
    apply_orthographic_fixes: bool = True,
) -> str:
    """Normalise a raw Arabic OCR string.

    Args:
        text: The raw OCR output.
        drop_latin_tokens: Remove tokens that contain no Arabic at all.
            Useful for name fields, where stray Latin characters are noise;
            left off for addresses, which legitimately contain digits.
        apply_orthographic_fixes: Apply the spelling-variant lexicon.

    Returns:
        The normalised string, or ``""`` when nothing survives.
    """
    if not text or not text.strip():
        return ""

    # NFKC folds Arabic presentation forms back to canonical letters, which
    # is what OCR engines sometimes emit.
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = clean_ocr_noise(text)
    text = merge_split_article(text)

    tokens: list[str] = []
    for token in text.split():
        if drop_latin_tokens and not contains_arabic(token) and _RE_LATIN_ONLY.match(token):
            continue
        if contains_arabic(token) and apply_orthographic_fixes:
            token = ORTHOGRAPHIC_FIXES.get(token, token)
            token = _fix_ta_marbuta(token)
            # A fix may produce a token that has its own entry.
            token = ORTHOGRAPHIC_FIXES.get(token, token)
        tokens.append(token)

    return _RE_WHITESPACE.sub(" ", " ".join(tokens)).strip()


def to_display_form(text: str) -> str:
    """Reshape and bidi-reorder Arabic for rendering in LTR-only contexts.

    Needed when drawing Arabic with Pillow, matplotlib or any renderer that
    does not implement the Unicode bidirectional algorithm. **Never** store
    or transmit the result -- it is visual order, not logical order, and
    round-trips badly. Falls back to the input if the optional
    dependencies are missing.
    """
    if not text:
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        log.warning(
            "arabic-reshaper / python-bidi not installed; returning logical order unchanged."
        )
        return text
