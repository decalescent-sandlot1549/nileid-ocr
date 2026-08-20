"""Centralised logging setup.

The library never calls :func:`logging.basicConfig` at import time — that is
an application-level decision. Import :func:`get_logger` in library modules
and call :func:`configure_logging` once from an entry point (API, CLI, demo).
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once, for application entry points."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = level or os.environ.get("NILEID_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)
    logging.basicConfig(level=resolved, format=_FORMAT)
    # These libraries are extremely chatty at INFO level.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for a library module."""
    return logging.getLogger(f"nileid.{name}" if not name.startswith("nileid") else name)
