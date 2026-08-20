# Privacy

This project processes **identity documents**. An Egyptian National ID card
carries a person's full name, address, date of birth, gender, place of birth
and a unique government identifier. Handling that data carelessly is not a
style problem — it is a harm to a real person.

## What this software does with your images

- Images are processed **in memory**. Nothing is written to disk by the
  library, the CLI (unless you pass `-o`), the API or the demo.
- The HTTP API logs the **size and MIME type** of an upload only. It does
  not log filenames — which frequently contain a person's name — and it
  never logs extracted values.
- No network calls are made with your image data. The only outbound
  requests the project makes are to download model weights
  (`scripts/download_models.py`) and, on first run, EasyOCR's language
  models from its own distribution.
- No telemetry, no analytics, no error reporting service.

## What you must not commit

The `.gitignore` is deliberately broad, and `tests/test_repository_hygiene.py`
fails the build if any of these appear in version control:

- real card images, in any format
- `uploads/`, `outputs/`, `data/`, `tmp/`, `debug/` and similar runtime
  directories
- extracted results containing real names, addresses or ID numbers
- model weights, `.env` files, credentials

Git history is effectively permanent and public repositories are widely
mirrored and scraped. A card image committed once and deleted in the next
commit is still published.

## Sample data

Every image in this repository is produced by
`scripts/make_sample_card.py`. The samples:

- use the **layout** of an Egyptian National ID so the geometry and
  detection stages can be exercised;
- carry invented placeholder content and a visible
  `SAMPLE — NOT A REAL ID` watermark;
- use a structurally valid but sequential placeholder number
  (`29001010100017`) that is not issued to anyone.

If you need to file a bug report, reproduce it with a synthetic card or a
fully redacted image. Do not attach a real document to a public issue.

## If you deploy this

You inherit responsibilities the library cannot discharge for you:

- **Have a lawful basis.** In most jurisdictions, processing identity
  documents requires consent or a specific legal ground. Egypt's Personal
  Data Protection Law (Law No. 151 of 2020) treats this as sensitive
  personal data.
- **Do not persist by default.** If you must store results, store the
  minimum, encrypt at rest, and set a retention limit you actually enforce.
- **Secure the endpoint.** The bundled API has no authentication. Put it
  behind authentication, TLS and rate limiting before exposing it.
- **Do not log field values.** The library does not; make sure your
  application layer does not either.
- **Tell people.** Users should know their document is being processed by
  an automated system, and what happens to the output.

## Accuracy has privacy consequences

A misread digit can associate one person's record with another person's
identifier. This is why the pipeline refuses to complete a partial National
ID number, and why derived fields stay `null` unless the number validates.
Treat every extracted value as a suggestion requiring human confirmation,
not as an authoritative record. See [LIMITATIONS.md](LIMITATIONS.md).
