#!/usr/bin/env python
"""Generate synthetic, ID-card-shaped images for testing and documentation.

These are **mock-ups**, not reproductions of a real Egyptian National ID.
They carry the same field layout -- so the geometry, orientation and text
stages can be exercised -- with invented placeholder content. No real
person's data is used, and the card is deliberately marked as a sample so
it cannot be mistaken for a genuine document.

The National ID numbers produced here are structurally valid but sequential
placeholders (``29001010100017``); they are not issued to anyone.

Usage:
    python scripts/make_sample_card.py --out examples/sample_card.png
    python scripts/make_sample_card.py --degrade --out examples/sample_card_skewed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARD_W, CARD_H = 1000, 630

#: Placeholder content. Common given names, an invented street address, and
#: a placeholder ID number.
DEFAULT_FIRST_NAME = "محمد"
DEFAULT_LAST_NAME = "عبد الله حسن"
DEFAULT_ADDRESS = "١٢ شارع النيل - المعادي - القاهرة"
DEFAULT_NID = "29001010100017"

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
_BOLD_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _BOLD_CANDIDATES if bold else _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _shaped(text: str) -> str:
    """Reshape Arabic for Pillow, which does not implement bidi itself."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def render_card(
    first_name: str = DEFAULT_FIRST_NAME,
    last_name: str = DEFAULT_LAST_NAME,
    address: str = DEFAULT_ADDRESS,
    national_id: str = DEFAULT_NID,
) -> Image.Image:
    """Draw a synthetic card face."""
    image = Image.new("RGB", (CARD_W, CARD_H), (238, 243, 249))
    draw = ImageDraw.Draw(image)

    # Guilloche-like background texture.
    for y in range(0, CARD_H, 7):
        draw.line([(0, y), (CARD_W, y)], fill=(226, 235, 245), width=1)
    draw.rectangle([0, 0, CARD_W - 1, CARD_H - 1], outline=(140, 162, 186), width=4)

    # Header band.
    draw.rectangle([4, 4, CARD_W - 5, 92], fill=(203, 220, 238))
    draw.text(
        (CARD_W // 2, 48),
        _shaped("بطاقة تحقيق الشخصية"),
        font=_font(38, bold=True),
        fill=(18, 38, 78),
        anchor="mm",
    )

    # Sample watermark, so the mock-up cannot pass as a real document.
    draw.text(
        (CARD_W // 2, CARD_H // 2),
        "SAMPLE — NOT A REAL ID",
        font=_font(38, bold=True),
        fill=(214, 226, 238),
        anchor="mm",
    )

    # Photo placeholder.
    draw.rectangle([44, 126, 254, 420], fill=(196, 205, 216), outline=(126, 140, 158), width=2)
    draw.text((149, 273), "PHOTO", font=_font(24), fill=(96, 108, 124), anchor="mm")

    # Right-aligned Arabic fields.
    draw.text(
        (CARD_W - 52, 152),
        _shaped(first_name),
        font=_font(42, bold=True),
        fill=(14, 26, 62),
        anchor="ra",
    )
    draw.text(
        (CARD_W - 52, 214),
        _shaped(last_name),
        font=_font(42, bold=True),
        fill=(14, 26, 62),
        anchor="ra",
    )
    draw.text((CARD_W - 52, 300), _shaped(address), font=_font(28), fill=(24, 36, 72), anchor="ra")

    # Serial, upper right.
    draw.text((CARD_W - 52, 112), "SN 0000000", font=_font(20), fill=(78, 90, 110), anchor="ra")

    # National ID strip, lower band.
    draw.text((150, 520), _shaped("الرقم القومي"), font=_font(24), fill=(60, 72, 96), anchor="mm")
    draw.text(
        (CARD_W // 2 + 70, 520),
        " ".join(national_id),
        font=_font(44, bold=True),
        fill=(10, 20, 60),
        anchor="mm",
    )
    return image


def degrade(image: Image.Image, angle: float = 4.0, blur: float = 0.8) -> Image.Image:
    """Apply photograph-like degradation: rotation, blur and sensor noise."""
    from PIL import ImageFilter

    rotated = image.rotate(angle, expand=True, fillcolor=(58, 62, 68), resample=Image.BICUBIC)
    blurred = rotated.filter(ImageFilter.GaussianBlur(blur))
    array = np.asarray(blurred).astype(np.int16)
    rng = np.random.default_rng(7)
    noisy = np.clip(array + rng.normal(0, 6, array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/sample_card.png")
    parser.add_argument("--first-name", default=DEFAULT_FIRST_NAME)
    parser.add_argument("--last-name", default=DEFAULT_LAST_NAME)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--nid", default=DEFAULT_NID)
    parser.add_argument("--degrade", action="store_true", help="add rotation, blur and noise")
    parser.add_argument("--angle", type=float, default=4.0)
    args = parser.parse_args()

    card = render_card(args.first_name, args.last_name, args.address, args.nid)
    if args.degrade:
        card = degrade(card, angle=args.angle)

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    card.save(destination)
    print(f"Wrote {destination.resolve()}  ({card.width}x{card.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
