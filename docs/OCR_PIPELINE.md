# The OCR pipeline

This document explains what each stage does, why it does it that way, and
what was measured.

## 1. Loading

`load_image` accepts a path, raw bytes or a NumPy array and always returns
a contiguous 3-channel BGR `uint8` array. Grayscale and BGRA inputs are
converted; float arrays are rescaled. Files are read through Python and
decoded from memory rather than via `cv2.imread`, which does not handle
non-ASCII paths reliably on Windows.

## 2. Quality assessment

Four cheap measurements drive later decisions and are reported to the
caller as warnings.

| Metric | Method | Threshold |
|---|---|---|
| Sharpness | Variance of the Laplacian | `< 100` → blurred |
| Contrast | Standard deviation of intensity | `< 45` → low contrast |
| Noise | Immerkaer Laplacian-convolution estimator | `> 6` → noisy |
| Exposure | Mean intensity | `< 70` dark, `> 200` washed out |

These are heuristics with hand-set thresholds, not a learned quality model.
Their purpose is to choose enhancement steps and to tell the user when a
poor scan explains a poor read.

## 3. Card localisation and orientation

`detect_id_card.pt` predicts one of eight classes — `{front,back}` ×
`{up,down,left,right}` — so a single inference gives both the card box and
the quarter-turn needed to stand it upright.

```
front-up    →   0°      front-right →  90°
front-down  → 180°      front-left  → 270°
```

If no box clears `card_conf` (default 0.25), the pipeline does **not**
refuse the image. It falls back to treating the frame as an already-cropped
card, finds the orientation by scoring the field layout across four
rotations, and records a warning. `card_detected` in the result tells the
caller which path was taken.

The layout score uses card geometry: the National ID sits in the lower
band, the serial in the upper band, name fields on the right (Arabic is
right-aligned), the photo on the left.

### Optional perspective correction

`correct_perspective` searches for the card's quadrilateral outline —
bilateral filter, Canny, morphological close, contour approximation — and
applies a four-point transform when it finds one with an ID-1-like aspect
ratio (85.6 × 54 mm, tolerance 0.45). It declines rather than guessing when
the background is cluttered.

**Measured** on a synthetic card projected onto a background: mean corner
error **0.5 px**, and the recovered card is returned at the canonical size.

## 4. Deskew

Small in-plane rotation is corrected by a **projection-profile search**:
binarise the text, trim 6 % from each edge, rotate over candidate angles,
and pick the angle whose horizontal row-sum profile has the sharpest
row-to-row transitions. Text lines that are level produce rows that are
either dense with ink or empty.

Two earlier approaches failed, and the reasons are worth recording:

- **Hough lines over Canny edges** (the original implementation) locks onto
  the card border and the guilloche background pattern rather than text.
- **Minimum-area rectangle of the text mask** is defeated by the card
  border: that border is a strong axis-aligned shape, so the enclosing
  rectangle stays axis-aligned no matter how the card is rotated, and the
  estimate collapses to exactly 0°.

**Measured** across introduced rotations of ±1° to ±15°: worst residual
**0.20°**, at roughly 11 ms per estimate. Beyond ±20° the estimator returns
0 rather than a wrong angle — quarter-turn rotation is the orientation
stage's job, not deskew's.

## 5. Field detection

The primary detector (`detect_odjects.pt`) provides 31 classes. Class names
are canonicalised: `invalid_lastName` and `lastName` both map to
`last_name`, and the `invalid_` prefix is reported as a warning rather than
suppressing the field.

The secondary detector (`best.pt`) is consulted only for fields the primary
missed. The primary always wins where both fired.

Each field gets its own crop margin, because detectors hug the glyphs and
OCR needs surrounding whitespace:

| Field | Width × Height margin |
|---|---|
| `first_name` | 1.15 × 1.30 |
| `last_name` | 1.10 × 1.25 |
| `address` | 1.12 × 1.35 |
| `nid` | 1.18 × 1.60 |

The National ID gets the most room: clipping an outer digit silently
shortens the number, which is the worst available failure.

## 6. Text recognition

Text crops are padded with a white border, then enhanced **conditionally**:

- **Upscale** so a text line reaches ~64 px tall (capped at 4×), rather
  than a fixed multiplier that starves small crops and wastes time on
  large ones.
- **CLAHE** only when the crop is flat, dark or blown out. Local
  equalisation is the right tool for shadows falling across a card.
- **Bilateral filter** only when noise is measured. It preserves glyph
  edges at a fraction of the cost of non-local means, which the original
  pipeline ran on every crop unconditionally and which dominated runtime.
- **Unsharp mask** only when the crop is genuinely soft, and gently, to
  avoid ringing that OCR reads as extra strokes.

EasyOCR (Arabic + English) is the default engine; PaddleOCR is optional.
When both are enabled, the more confident reading wins, with string length
as a tie-break.

### Reading order

Both engines return one entry per detected box in arbitrary order. Boxes
are sorted top-to-bottom into lines, then **right-to-left within each
line**, before being joined. Joining in the engine's own order scrambles a
multi-part address.

## 7. Arabic normalisation

Text is stored in **logical order**, which is correct for JSON, databases
and further processing. `to_display_form` provides visual reordering for
renderers that lack bidi support (Pillow, matplotlib); its output must
never be stored.

Normalisation applies, in order: NFKC (folds presentation forms back to
canonical letters), diacritic and tatweel removal, OCR-artefact stripping,
definite-article rejoining, then a lexicon of orthographic repairs.

The lexicon contains **only spelling variants of the same word** — hamza
restoration (`احمد` → `أحمد`), ta-marbuta (`فاطمه` → `فاطمة`),
alif-maqsura (`مصطفي` → `مصطفى`) and compound-name splits
(`عبدالله` → `عبد الله`). Entries that mapped a token to a *different*
word were removed: on an identity document, a rare name must survive
unchanged rather than be replaced by a common one that looks similar.

Words that legitimately end in a bare `ه` — notably `الله` — are protected
from the ta-marbuta rewrite.

### Optional dictionary correction

Off by default. When enabled it snaps a token to a lexicon entry only if
similarity reaches 92, and refuses when two candidates are within 4 points
of each other. Even so it is a guess: `محمو` resolves to `محمود`, not
`محمد`, and either could be right.

## 8. National ID digits

The digit detector's class index is the digit value, so reading the number
is a detection problem rather than a recognition problem.

Post-processing fixes three defects:

1. **Class-agnostic NMS.** Ultralytics suppresses per class, so two
   different digit classes can both survive on one glyph — turning 14
   digits into 15. Suppression across classes at IoU 0.40 fixes this.
2. **A confidence floor** of 0.25, instead of accepting every box from a
   `conf=0.10` inference, which let guilloche texture become digits.
3. **A length check.** A read that is not exactly 14 digits is reported as
   incomplete and never padded, completed or guessed at.

Digits are sorted left-to-right, which is correct even on an otherwise
right-to-left card.

## 9. Validation and decoding

```
2  90  01  01  01  0001  7
│  │   │   │   │   │     └─ check digit (advisory)
│  │   │   │   │   └─────── sequence; index 12 odd → male
│  │   │   │   └─────────── governorate code
│  │   │   └─────────────── day
│  │   └─────────────────── month
│  └─────────────────────── year (last two digits)
└────────────────────────── century: 2 → 1900s, 3 → 2000s
```

Checks run independently and are reported separately, so "OCR dropped a
digit" is distinguishable from "this date cannot exist":

- length is exactly 14
- century digit is 2 or 3
- the date exists and is not in the future
- the governorate code is known
- the check digit matches

The check digit is **advisory only**. The official algorithm is not
published; the widely circulated weighted-modulus-11 scheme is computed and
reported, but a mismatch does not by itself mark a number invalid, because
doing so would reject legitimate cards.

Birth date, governorate and gender are decoded **only** when length, date
and governorate all pass. A birth date derived from a half-read number is
worse than no birth date.
