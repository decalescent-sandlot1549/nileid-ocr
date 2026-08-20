#!/usr/bin/env python
"""Fetch the YOLO weight files this project needs.

The weights are distributed as assets on the GitHub release rather than
committed to the repository: together they are roughly 86 MB, which makes
every clone slow and every checkout of history expensive.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --dest models --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "ahmedsayed1911/nileid-ocr"
RELEASE_TAG = "v1.0.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

# filename -> (approximate size in MB, sha256 of the released asset)
MODELS: dict[str, tuple[int, str]] = {
    "detect_id_card.pt": (6, ""),
    "detect_odjects.pt": (6, ""),
    "detect_id.pt": (22, ""),
    "best.pt": (50, ""),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, destination: Path, force: bool) -> bool:
    import requests

    target = destination / name
    if target.exists() and not force:
        print(f"  {name:<22} already present, skipping")
        return True

    url = f"{BASE_URL}/{name}"
    print(f"  {name:<22} downloading from {url}")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except Exception as exc:
        print(f"  {name:<22} FAILED: {exc}", file=sys.stderr)
        return False

    total = int(response.headers.get("content-length", 0))
    written = 0
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            handle.write(chunk)
            written += len(chunk)
            if total:
                percent = 100 * written / total
                print(f"\r  {name:<22} {percent:5.1f}%", end="", flush=True)
    print()
    temporary.replace(target)

    expected = MODELS[name][1]
    if expected:
        actual = sha256_of(target)
        if actual != expected:
            print(f"  {name:<22} CHECKSUM MISMATCH", file=sys.stderr)
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="models", help="destination directory")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)

    total_mb = sum(size for size, _ in MODELS.values())
    print(f"Fetching {len(MODELS)} weight files (~{total_mb} MB) into {destination.resolve()}")

    ok = all(download(name, destination, args.force) for name in MODELS)
    if ok:
        print("\nAll weights are in place.")
        return 0
    print("\nSome downloads failed. See the release page:", file=sys.stderr)
    print(f"  https://github.com/{REPO}/releases/tag/{RELEASE_TAG}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
