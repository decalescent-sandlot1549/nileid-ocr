"""Command-line interface.

nileid read card.jpg
nileid read card.jpg --json -o result.json
nileid read ./batch/*.jpg --flat
nileid validate 29001010100017
nileid check
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from nileid import __version__
from nileid.config import Settings
from nileid.logging_utils import configure_logging, get_logger
from nileid.models import ModelNotFoundError, missing_models
from nileid.preprocessing.image_io import SUPPORTED_EXTENSIONS, ImageLoadError

log = get_logger("cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_READ = 2


def _force_utf8_output() -> None:
    """Make Arabic printable on consoles that default to a legacy codepage.

    Windows terminals commonly run cp1252, where printing Arabic raises
    ``UnicodeEncodeError``. Reconfiguring the streams is preferable to
    escaping the output, which would make it unusable.
    """
    for stream in (sys.stdout, sys.stderr):
        # Exotic streams may not support reconfiguration; output is then
        # left as-is rather than failing the command.
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _print_human(result, path: Path) -> None:
    """Render a result as an aligned, readable summary."""
    labels = {
        "full_name": "Full name",
        "first_name": "First name",
        "last_name": "Last name",
        "address": "Address",
        "national_id": "National ID",
    }
    print(f"\n{path.name}")
    print("─" * 60)
    if not result.card_detected:
        print("  card:        not detected (processed as a pre-cropped card)")
    else:
        side = result.side or "unknown"
        print(f"  card:        detected ({side}, {result.card_confidence:.0%} confidence)")

    for attribute, label in labels.items():
        field = getattr(result, attribute)
        if field.value:
            marker = " ~" if field.status.value == "low_confidence" else "  "
            print(f"  {label:<12}{marker} {field.value}   [{field.confidence:.0%}]")
        else:
            print(f"  {label:<12}   — ({field.status.value})")

    if result.birth_date:
        print(f"  {'Birth date':<12}   {result.birth_date}")
        print(f"  {'Governorate':<12}   {result.governorate}")
        print(f"  {'Gender':<12}   {result.gender}")

    validation = result.validation
    print(f"  {'ID valid':<12}   {'yes' if validation.valid else 'no'}", end="")
    print(f" ({'; '.join(validation.errors)})" if validation.errors else "")

    if result.warnings:
        print("\n  Warnings:")
        for warning in result.warnings:
            print(f"    · {warning}")
    print(f"\n  {result.processing_time_ms:.0f} ms")


def _expand_inputs(patterns: list[str]) -> list[Path]:
    """Expand directories and glob patterns into a list of image files."""
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            paths.extend(
                p for p in sorted(candidate.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif candidate.exists():
            paths.append(candidate)
        else:
            # Let the shell-unexpanded glob be resolved here (Windows).
            matches = sorted(Path().glob(pattern))
            if matches:
                paths.extend(matches)
            else:
                log.error("No such file: %s", pattern)
    return paths


def _cmd_read(args: argparse.Namespace) -> int:
    from nileid import EgyptianIDReader

    paths = _expand_inputs(args.images)
    if not paths:
        print("No input images found.", file=sys.stderr)
        return EXIT_ERROR

    settings = Settings(
        device=args.device,
        enable_fuzzy_correction=args.fuzzy,
    )
    try:
        reader = EgyptianIDReader(settings)
    except ModelNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR

    payloads = []
    any_read = False
    for path in paths:
        try:
            result = reader.read(path)
        except ImageLoadError as exc:
            log.error("%s: %s", path.name, exc)
            continue
        except ModelNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_ERROR

        any_read = any_read or not result.is_empty
        payload = result.to_flat_dict() if args.flat else result.to_dict()
        payload["source"] = path.name
        payloads.append(payload)

        if not args.json:
            _print_human(result, path)

    if args.json:
        document = payloads if len(payloads) > 1 else (payloads[0] if payloads else {})
        rendered = json.dumps(document, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(rendered)
    elif args.output:
        document = payloads if len(payloads) > 1 else (payloads[0] if payloads else {})
        Path(args.output).write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")

    return EXIT_OK if any_read else EXIT_NOT_READ


def _cmd_validate(args: argparse.Namespace) -> int:
    from nileid.validation import decode_national_id, validate_national_id

    validation = validate_national_id(args.number)
    decoded = decode_national_id(args.number)
    payload = {
        "input": args.number,
        "validation": validation.to_dict(),
        "decoded": (
            {
                "national_id": decoded.national_id,
                "birth_date": decoded.birth_date_iso,
                "governorate": decoded.governorate,
                "governorate_code": decoded.governorate_code,
                "gender": decoded.gender,
            }
            if decoded
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return EXIT_OK if validation.valid else EXIT_NOT_READ


def _cmd_check(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Report whether the environment is ready to run the pipeline."""
    settings = Settings()
    print(f"nileid {__version__}")
    print(f"  model directory : {settings.model_dir.resolve()}")

    absent = missing_models(settings)
    if absent:
        print(f"  weights         : MISSING -> {', '.join(absent)}")
        print("                    run: python scripts/download_models.py")
    else:
        print("  weights         : all present")

    from nileid.models import resolve_device

    print(f"  device          : {resolve_device(settings.device)}")

    for module in ("cv2", "torch", "ultralytics", "easyocr", "rapidfuzz", "paddleocr"):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "installed")
            print(f"  {module:<16}: {version}")
        except ImportError:
            optional = " (optional)" if module == "paddleocr" else ""
            print(f"  {module:<16}: not installed{optional}")

    return EXIT_ERROR if absent else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nileid",
        description="Structured information extraction from Egyptian National ID cards.",
    )
    parser.add_argument("--version", action="version", version=f"nileid {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    read = subparsers.add_parser("read", help="read one or more card images")
    read.add_argument("images", nargs="+", help="image files, directories or globs")
    read.add_argument("--json", action="store_true", help="emit JSON instead of a summary")
    read.add_argument(
        "--flat", action="store_true", help="flat field:value JSON, without confidence detail"
    )
    read.add_argument("-o", "--output", help="write JSON to this file")
    read.add_argument("--device", default="auto", help="inference device: auto, cpu, cuda, cuda:0")
    read.add_argument(
        "--fuzzy",
        action="store_true",
        help="enable dictionary-assisted correction of Arabic tokens (off by default)",
    )
    read.set_defaults(func=_cmd_read)

    validate = subparsers.add_parser("validate", help="validate and decode a National ID number")
    validate.add_argument("number", help="14-digit National ID number")
    validate.set_defaults(func=_cmd_validate)

    check = subparsers.add_parser("check", help="report environment readiness")
    check.set_defaults(func=_cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
