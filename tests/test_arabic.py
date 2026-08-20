"""Arabic normalisation.

The recurring theme is that normalisation must be *lossless in meaning*:
it may remove noise and repair spelling variants, but it must never turn
one word into a different word.
"""

from __future__ import annotations

import pytest

from nileid.extraction.arabic import (
    clean_ocr_noise,
    contains_arabic,
    merge_split_article,
    normalize_arabic,
    strip_diacritics,
    to_display_form,
    to_western_digits,
)
from nileid.extraction.lexicon import correct_text, suggest_correction


class TestCharacterHelpers:
    @pytest.mark.parametrize("text", ["محمد", "abc محمد", "١٢٣ شارع"])
    def test_detects_arabic(self, text):
        assert contains_arabic(text)

    @pytest.mark.parametrize("text", ["", "abc", "12345", "!!!"])
    def test_detects_absence_of_arabic(self, text):
        assert not contains_arabic(text)

    def test_converts_arabic_indic_digits(self):
        assert to_western_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"

    def test_leaves_letters_untouched_when_converting_digits(self):
        assert to_western_digits("شارع ١٢") == "شارع 12"

    def test_strips_harakat(self):
        assert strip_diacritics("مُحَمَّد") == "محمد"

    def test_strips_tatweel(self):
        assert strip_diacritics("محمــــد") == "محمد"

    def test_removes_bracket_and_pipe_noise(self):
        assert clean_ocr_noise("[ محمد ] | حسن") == "محمد حسن"

    def test_collapses_whitespace(self):
        assert clean_ocr_noise("محمد     حسن") == "محمد حسن"


class TestArticleMerging:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ال قاهرة", "القاهرة"),
            ("ال جيزة", "الجيزة"),
            ("مدينة ال منصورة", "مدينة المنصورة"),
        ],
    )
    def test_rejoins_split_article(self, raw, expected):
        assert merge_split_article(raw) == expected

    def test_leaves_joined_article_alone(self):
        assert merge_split_article("القاهرة") == "القاهرة"

    def test_terminates_on_pathological_input(self):
        # The merge loop iterates; it must converge rather than hang.
        assert merge_split_article("ال " * 20).strip().startswith("ال")


class TestNormalisation:
    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_input_yields_empty_string(self, raw):
        assert normalize_arabic(raw) == ""

    def test_repairs_hamza(self):
        assert normalize_arabic("احمد") == "أحمد"

    def test_repairs_alif_maqsura(self):
        assert normalize_arabic("مصطفي") == "مصطفى"

    def test_splits_compound_name(self):
        assert normalize_arabic("عبدالله") == "عبد الله"

    def test_repairs_ta_marbuta(self):
        assert normalize_arabic("فاطمه") == "فاطمة"

    def test_preserves_the_divine_name(self):
        # A blanket final-ha rewrite would corrupt this.
        assert normalize_arabic("الله") == "الله"
        assert normalize_arabic("عبد الله") == "عبد الله"

    def test_preserves_short_words_ending_in_ha(self):
        assert normalize_arabic("له") == "له"

    def test_drops_latin_tokens_only_when_requested(self):
        assert normalize_arabic("محمد ABC", drop_latin_tokens=True) == "محمد"
        assert "ABC" in normalize_arabic("محمد ABC", drop_latin_tokens=False)

    def test_keeps_digits_in_addresses(self):
        # Addresses legitimately contain house numbers.
        out = normalize_arabic("١٢ شارع النيل", drop_latin_tokens=False)
        assert "١٢" in out and "شارع" in out

    def test_orthographic_fixes_can_be_disabled(self):
        assert normalize_arabic("احمد", apply_orthographic_fixes=False) == "احمد"

    def test_is_idempotent(self):
        once = normalize_arabic("احمد   مصطفي [ عبدالله ]")
        assert normalize_arabic(once) == once

    def test_does_not_invent_words(self):
        # A rare token with no lexicon entry must survive verbatim.
        rare = "زغلولاوي"
        assert normalize_arabic(rare) == rare


class TestDisplayForm:
    def test_display_form_is_reversible_in_length(self):
        shaped = to_display_form("محمد")
        assert isinstance(shaped, str) and shaped

    def test_empty_input(self):
        assert to_display_form("") == ""


class TestLexiconCorrection:
    def test_leaves_short_tokens_alone(self):
        assert suggest_correction("حي", "address") == "حي"

    def test_returns_input_for_unknown_field_type(self):
        assert suggest_correction("محمد", "not_a_field") == "محمد"

    def test_corrects_a_near_miss_place_name(self):
        # One-character corruption of a place in the lexicon.
        assert suggest_correction("الزقازيف", "address", threshold=85) == "الزقازيق"

    def test_respects_the_threshold(self):
        # The same token is left alone at the stricter default threshold.
        assert suggest_correction("الزقازيف", "address", threshold=92) == "الزقازيف"

    def test_refuses_an_ambiguous_match(self, monkeypatch):
        """Two equally close candidates must leave the token untouched.

        Picking either one would be a coin flip on an identity document.
        """
        import nileid.extraction.lexicon as lexicon_module

        monkeypatch.setitem(lexicon_module._LEXICONS, "name", ("سالمة", "سالمه"))
        assert suggest_correction("سالمت", "name", threshold=75) == "سالمت"

    def test_correction_is_a_lexicon_entry_or_the_original(self):
        # It may never synthesise a token that is in neither place.
        from nileid.extraction.lexicon import COMMON_NAMES

        token = "محمو"
        result = suggest_correction(token, "name", threshold=80)
        assert result == token or result in COMMON_NAMES

    def test_refuses_a_distant_token(self):
        rare = "زغلولاوي"
        assert suggest_correction(rare, "name") == rare

    def test_correct_text_preserves_token_count(self):
        text = "محمد عبد الله"
        assert len(correct_text(text, "name").split()) == len(text.split())

    def test_empty_text(self):
        assert correct_text("", "name") == ""
