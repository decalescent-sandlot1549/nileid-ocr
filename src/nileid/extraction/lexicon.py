"""Optional dictionary-assisted correction of Arabic OCR tokens.

This module is **disabled by default** (``Settings.enable_fuzzy_correction``).
Replacing a token with a lexicon entry is a guess: it improves output on
common names and place names, and corrupts it on rare ones. Because an ID
card is an identity document, the default is to return exactly what OCR saw.

When enabled, the correction is deliberately narrow:

* only tokens that are already Arabic and at least four characters long;
* only when similarity reaches ``fuzzy_threshold`` (92 by default);
* only when a single candidate stands clearly above the runner-up, so
  ambiguous matches are left alone.
"""

from __future__ import annotations

from functools import lru_cache

from nileid.logging_utils import get_logger
from nileid.validation.governorates import GOVERNORATES

log = get_logger("extraction.lexicon")

#: Frequent Egyptian given names. Used only as correction candidates.
COMMON_NAMES: tuple[str, ...] = (
    "أحمد",
    "محمد",
    "محمود",
    "مصطفى",
    "إبراهيم",
    "علي",
    "حسين",
    "حسن",
    "خالد",
    "عمر",
    "عبد الرحمن",
    "عبد الله",
    "عبد العزيز",
    "طارق",
    "عمرو",
    "يوسف",
    "ياسين",
    "زياد",
    "كريم",
    "حازم",
    "هشام",
    "وائل",
    "رامي",
    "تامر",
    "سعيد",
    "صالح",
    "عادل",
    "عصام",
    "ماجد",
    "وليد",
    "أشرف",
    "أيمن",
    "إسلام",
    "بهاء",
    "جمال",
    "رجب",
    "سعد",
    "شعبان",
    "صلاح",
    "طاهر",
    "عاطف",
    "عوض",
    "فاروق",
    "فؤاد",
    "كمال",
    "ممدوح",
    "نبيل",
    "هاني",
    "ياسر",
    "أميرة",
    "إيمان",
    "دعاء",
    "ريهام",
    "سارة",
    "شيماء",
    "فاطمة",
    "مروة",
    "منى",
    "نادية",
    "نجلاء",
    "هبة",
    "ياسمين",
    "زينب",
    "خديجة",
    "عائشة",
    "سمية",
    "ليلى",
    "سلوى",
    "نورهان",
    "إسراء",
    "آية",
)

#: Administrative vocabulary and place names that appear in card addresses.
ADDRESS_TERMS: tuple[str, ...] = tuple(GOVERNORATES.values()) + (
    "مركز",
    "قسم",
    "شياخة",
    "قرية",
    "عزبة",
    "شارع",
    "حي",
    "مدينة",
    "ميدان",
    "منطقة",
    "شبرا",
    "المعادي",
    "حلوان",
    "الدقي",
    "العجوزة",
    "المهندسين",
    "إمبابة",
    "الهرم",
    "فيصل",
    "الزقازيق",
    "المنصورة",
    "طنطا",
    "المحلة",
    "بنها",
    "دمنهور",
    "دسوق",
    "شبين الكوم",
    "أشمون",
    "قويسنا",
    "ميت غمر",
    "بلبيس",
    "فاقوس",
    "العياط",
    "مغاغة",
    "سمالوط",
    "ملوي",
    "منفلوط",
    "طهطا",
    "أخميم",
    "جرجا",
    "نجع حمادي",
    "إدفو",
    "كوم أمبو",
)

_LEXICONS: dict[str, tuple[str, ...]] = {
    "name": COMMON_NAMES,
    "address": ADDRESS_TERMS,
}

#: A replacement is only accepted when the best candidate beats the
#: runner-up by at least this margin, to avoid coin-flip substitutions.
_AMBIGUITY_MARGIN = 4


@lru_cache(maxsize=1)
def _rapidfuzz():
    """Import rapidfuzz lazily; return ``None`` when unavailable."""
    try:
        from rapidfuzz import fuzz, process

        return process, fuzz
    except ImportError:
        log.warning("rapidfuzz not installed; dictionary correction is unavailable.")
        return None


def suggest_correction(token: str, field_type: str, threshold: int = 92) -> str:
    """Return a corrected token, or the original when no confident match exists."""
    candidates = _LEXICONS.get(field_type)
    if not candidates or not token or len(token) < 4:
        return token

    imported = _rapidfuzz()
    if imported is None:
        return token
    process, fuzz = imported

    matches = process.extract(token, candidates, scorer=fuzz.ratio, limit=2)
    if not matches:
        return token

    best_text, best_score, _ = matches[0]
    if best_score < threshold:
        return token
    if len(matches) > 1 and best_score - matches[1][1] < _AMBIGUITY_MARGIN:
        # Two lexicon entries are almost equally close; refuse to choose.
        return token
    if best_text != token:
        log.debug("Lexicon correction %r -> %r (score=%.1f)", token, best_text, best_score)
    return best_text


def correct_text(text: str, field_type: str, threshold: int = 92) -> str:
    """Apply :func:`suggest_correction` token by token."""
    if not text:
        return text
    return " ".join(suggest_correction(tok, field_type, threshold) for tok in text.split())
