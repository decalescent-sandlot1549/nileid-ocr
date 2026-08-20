# Limitations

An honest account of what this project does not do well. Read this before
relying on it for anything that matters.

## No accuracy benchmark is published

**There is no measured accuracy figure for field extraction in this
repository, and any number quoted for it would be fabricated.**

Benchmarking would require a labelled corpus of real Egyptian National ID
cards. Assembling, storing and publishing such a corpus would mean handling
hundreds of real people's identity documents — exactly what
[PRIVACY.md](PRIVACY.md) says not to do. The trade-off is deliberate:
this project has verifiable engineering and unverified end-to-end accuracy.

What *is* measured, on synthetic fixtures, is reported in
[OCR_PIPELINE.md](OCR_PIPELINE.md):

| Component | Measurement |
|---|---|
| Perspective correction | 0.5 px mean corner error |
| Deskew | 0.20° worst residual across ±15° |
| Warm processing time | ~1.7 s per card (RTX 3090 Ti) |

These cover the deterministic geometry. They say nothing about how well the
detectors and OCR read a real card.

## The synthetic samples are not a proxy for real cards

The bundled sample cards share the *layout* of a real ID but not its
typeface, security printing, paper texture, photo or holographic elements.
The YOLO checkpoints were trained on real cards, so they detect noticeably
less on a synthetic mock-up than they would in production. If you run the
demo on `examples/sample_card.png` and see fields come back `not_found` or a
partial National ID, that is the expected result — and the pipeline reports
it as such rather than hiding it.

## Model provenance is incompletely documented

The four YOLO checkpoints are inherited artefacts. The training data,
augmentation and evaluation protocol behind them are not documented here
because that information was not available. Two consequences:

- Their generalisation to card variants, lighting and camera types outside
  their training distribution is unknown.
- `best.pt`'s class names are bare indices (`"0"`–`"6"`). The mapping in
  `detection/fields.py` was recovered empirically by observing where each
  class fires. Index 3 (first name) is the one the original pipeline relied
  on and is well established; the others are used only as fallbacks.

## Known functional limits

- **Front side only.** Name, address and the National ID are printed on the
  card front. The back is detected and reported (`side: "back"`) with a
  warning, but its fields are not extracted.
- **Deskew covers ±20°.** Beyond that the estimator returns 0 rather than a
  wrong angle. Larger rotations must be handled by the quarter-turn
  orientation stage, which only resolves multiples of 90°. A card rotated
  by, say, 40° will not be fully corrected.
- **Perspective correction is opportunistic.** It requires a findable card
  outline covering at least 20 % of the frame with an ID-1-like aspect
  ratio. Against a cluttered or low-contrast background it declines.
- **Arabic OCR is the weakest link.** EasyOCR's Arabic model is general
  purpose, not tuned for the ID card's typeface. Names and addresses are
  the least reliable fields, which is why they carry per-field confidence.
- **The check digit is unverified.** The official algorithm is not
  published. The implemented modulus-11 variant is advisory; do not treat
  `checksum_ok: false` as proof of a forged number, or `true` as proof of a
  genuine one.
- **No authenticity verification.** This project reads cards. It does not
  and cannot determine whether a card is genuine. The field detector's
  `invalid_*` classes are surfaced as a warning, but their meaning in the
  original training is not documented and they must not be used as a
  fraud signal.
- **Governorate reflects registration, not residence.** It is decoded from
  the ID number and indicates where the birth was registered.

## Performance

Warm processing is roughly 1.7 s per card on an RTX 3090 Ti. On CPU expect
several seconds, dominated by EasyOCR. The first call in a process is far
slower — model loading plus EasyOCR's one-time language-model download.
Create one `EgyptianIDReader` and reuse it; the API does this at start-up.

## Roadmap

Improvements that would need real, consented data:

- an evaluation harness and a published accuracy figure
- a fine-tuned Arabic recogniser for the card typeface
- card-back extraction
- arbitrary-angle rotation correction
