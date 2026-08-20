# Contributing

Thanks for your interest in improving NileID.

## Setup

    git clone https://github.com/ahmedsayed1911/nileid-ocr.git
    cd nileid-ocr
    python -m venv .venv
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -e ".[dev,api,demo]"
    python scripts/download_models.py

## Before opening a pull request

    pytest
    ruff check src tests scripts demo
    ruff format --check src tests scripts demo

## The one non-negotiable rule

**Never add a real identity document, National ID number, name or address to
this repository** — not in tests, not in fixtures, not in an issue, not in a
commit that you plan to amend away. Git history is permanent and public
repositories are scraped.

Generate test material instead:

    python scripts/make_sample_card.py --out examples/my_case.png

`tests/test_repository_hygiene.py` enforces this automatically and will fail
the build if card images, model weights, `.env` files or large binaries are
staged.

## Design principles worth preserving

1. **Never invent a value.** If OCR could not read a field, it is `null`
   with a status — not a plausible guess, not an empty string.
2. **Derived values require validation.** Birth date, governorate and gender
   come from the National ID number and appear only when it validates.
3. **Read class names from the model.** Never hard-code an index → name
   table; it rots silently when a checkpoint is retrained.
4. **Enhancement must be conditional.** Measure first, then apply only the
   steps the measurement justifies.
5. **Keep `raw` alongside `value`.** Normalisation is lossy.
6. **Claims need evidence.** Do not add an accuracy number to the README
   without a reproducible measurement behind it.

## Good first contributions

- More Arabic orthographic variants (spelling variants of the *same* word
  only — never a mapping to a different word)
- Additional preprocessing tests with quantitative assertions
- Card-back field extraction
- Arbitrary-angle rotation correction
