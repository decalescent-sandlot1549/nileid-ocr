"""Repository hygiene guards.

This project processes identity documents, so the repository itself is part
of the threat model. These tests fail the build if a card image, a model
weight, a secret or an oversized binary is ever committed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Extensions that must never appear in version control.
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",  # model weights
    ".env",  # secrets
    ".heic",  # phone camera captures
}

#: Directories that hold user data at runtime and must stay untracked.
FORBIDDEN_DIRECTORIES = {"uploads", "outputs", "tmp", "temp", "data", "debug", "models"}

#: Any file larger than this in git is a mistake for a project this size.
MAX_TRACKED_BYTES = 2 * 1024 * 1024

#: Image files that ARE allowed, because they are generated mock-ups.
ALLOWED_IMAGE_DIRS = {"examples", "assets", "docs"}

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"][^'\"]{12,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub token
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _tracked_files() -> list[Path]:
    """Files git is tracking, or ``[]`` outside a repository."""
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return []
    if output.returncode != 0:
        return []
    return [ROOT / name for name in output.stdout.split("\0") if name]


@pytest.fixture(scope="module")
def tracked() -> list[Path]:
    files = _tracked_files()
    if not files:
        pytest.skip("not a git repository (or git is unavailable)")
    return files


class TestNoSensitiveArtefacts:
    def test_no_model_weights_or_secrets_are_tracked(self, tracked):
        offenders = [
            p.relative_to(ROOT).as_posix()
            for p in tracked
            if p.suffix.lower() in FORBIDDEN_SUFFIXES
        ]
        assert not offenders, f"must not be committed: {offenders}"

    def test_no_runtime_data_directories_are_tracked(self, tracked):
        offenders = [
            p.relative_to(ROOT).as_posix()
            for p in tracked
            if p.relative_to(ROOT).parts and p.relative_to(ROOT).parts[0] in FORBIDDEN_DIRECTORIES
        ]
        assert not offenders, f"runtime data must stay untracked: {offenders}"

    def test_no_dotenv_file_is_tracked(self, tracked):
        names = {p.name for p in tracked}
        assert ".env" not in names
        assert ".env.example" in names, "a placeholder-only example should be provided"

    def test_no_large_files_are_tracked(self, tracked):
        offenders = [
            (p.relative_to(ROOT).as_posix(), p.stat().st_size)
            for p in tracked
            if p.is_file() and p.stat().st_size > MAX_TRACKED_BYTES
        ]
        assert not offenders, f"files over 2 MB: {offenders}"

    def test_images_live_only_in_curated_directories(self, tracked):
        """Card images may only come from the synthetic generator."""
        offenders = []
        for path in tracked:
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif"}:
                continue
            parts = path.relative_to(ROOT).parts
            if not parts or parts[0] not in ALLOWED_IMAGE_DIRS:
                offenders.append(path.relative_to(ROOT).as_posix())
        assert not offenders, f"unexpected image locations: {offenders}"


class TestNoSecretsInSource:
    def test_source_contains_no_credential_patterns(self, tracked):
        offenders = []
        for path in tracked:
            if path.suffix.lower() not in {".py", ".toml", ".yml", ".yaml", ".md", ".cfg", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover
                continue
            for pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    offenders.append((path.relative_to(ROOT).as_posix(), pattern.pattern[:40]))
        assert not offenders, f"possible secrets: {offenders}"

    def test_no_absolute_local_paths_in_source(self, tracked):
        """A hard-coded developer path breaks the project for everyone else."""
        pattern = re.compile(r"[A-Za-z]:[\\/]Users[\\/]|/home/[a-z]+/|/Users/[a-z]+/", re.I)
        offenders = []
        for path in tracked:
            if path.suffix.lower() not in {".py", ".toml", ".yml", ".yaml", ".cfg"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for number, line in enumerate(text.splitlines(), 1):
                # Font lookup tables legitimately list system paths.
                if "Fonts" in line or "fonts" in line:
                    continue
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}:{number}")
        assert not offenders, f"absolute local paths: {offenders}"


class TestGitignoreCoverage:
    def test_gitignore_exists(self):
        assert (ROOT / ".gitignore").is_file()

    @pytest.mark.parametrize(
        "entry", ["*.pt", ".env", "uploads/", "outputs/", "__pycache__/", ".venv/"]
    )
    def test_gitignore_covers_sensitive_paths(self, entry):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert entry in text, f".gitignore should list {entry}"
