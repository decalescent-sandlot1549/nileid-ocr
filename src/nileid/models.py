"""Lazy, process-wide registry for the YOLO detectors and OCR engines.

Loading a YOLO checkpoint costs seconds and an EasyOCR reader costs more,
so every model is constructed at most once per process and reused. The
registry is keyed by ``(model key, resolved device)`` so a caller that
switches device does not silently keep the old placement.

Nothing here is imported at module load time: ``ultralytics``, ``torch``
and ``easyocr`` are heavy, and the API, CLI and tests must be able to
import :mod:`nileid` without paying for them.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from nileid.config import MODEL_FILES, Settings
from nileid.logging_utils import get_logger

log = get_logger("models")

_lock = threading.Lock()
_models: dict[tuple[str, str], Any] = {}
_ocr: dict[str, Any] = {}


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required weight file is absent."""


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device string, falling back to CPU when CUDA is absent."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"


def missing_models(settings: Settings) -> list[str]:
    """Return the weight filenames that are not present on disk."""
    return [
        filename for key, filename in MODEL_FILES.items() if not settings.model_path(key).is_file()
    ]


def get_detector(key: str, settings: Settings) -> Any:
    """Load (once) and return an ultralytics YOLO detector.

    Raises:
        ModelNotFoundError: when the weight file is missing.
    """
    device = resolve_device(settings.device)
    cache_key = (key, device)
    if cache_key in _models:
        return _models[cache_key]

    with _lock:
        if cache_key in _models:  # another thread won the race
            return _models[cache_key]

        path: Path = settings.model_path(key)
        if not path.is_file():
            raise ModelNotFoundError(
                f"Model weights not found: {path}\n"
                f"Download them with:  python scripts/download_models.py"
            )

        from ultralytics import YOLO

        log.info("Loading detector %r from %s on %s", key, path, device)
        model = YOLO(str(path))
        try:
            model.to(device)
        except (RuntimeError, AssertionError, ValueError):
            log.warning("Could not place %r on %s; falling back to CPU.", key, device)
            model.to("cpu")

        _models[cache_key] = model
        return model


def get_easyocr(settings: Settings) -> Any | None:
    """Load (once) and return an EasyOCR reader, or ``None`` if unavailable."""
    if "easyocr" in _ocr:
        return _ocr["easyocr"]

    with _lock:
        if "easyocr" in _ocr:
            return _ocr["easyocr"]
        try:
            import easyocr

            use_gpu = resolve_device(settings.device).startswith("cuda")
            log.info(
                "Initialising EasyOCR (languages=%s, gpu=%s); "
                "language models download on first use.",
                ",".join(settings.ocr_languages),
                use_gpu,
            )
            _ocr["easyocr"] = easyocr.Reader(
                list(settings.ocr_languages), gpu=use_gpu, verbose=False
            )
        except ImportError:
            log.error("EasyOCR is not installed. Install it with:  pip install easyocr")
            _ocr["easyocr"] = None
        except Exception:
            log.exception("EasyOCR failed to initialise; text fields will be unreadable.")
            _ocr["easyocr"] = None
        return _ocr["easyocr"]


def get_paddleocr(settings: Settings) -> Any | None:
    """Load (once) and return a PaddleOCR instance, or ``None`` if unavailable.

    PaddleOCR is optional. Its constructor signature has changed across
    releases (``show_log`` was removed in 3.x), so the arguments are probed
    rather than assumed.
    """
    if "paddleocr" in _ocr:
        return _ocr["paddleocr"]

    with _lock:
        if "paddleocr" in _ocr:
            return _ocr["paddleocr"]
        try:
            from paddleocr import PaddleOCR

            use_gpu = resolve_device(settings.device).startswith("cuda")
            for kwargs in (
                {"use_angle_cls": True, "lang": "ar", "show_log": False, "use_gpu": use_gpu},
                {"use_angle_cls": True, "lang": "ar"},
                {"lang": "ar"},
            ):
                try:
                    _ocr["paddleocr"] = PaddleOCR(**kwargs)
                    log.info("Initialised PaddleOCR with %s", sorted(kwargs))
                    break
                except (TypeError, ValueError):
                    continue
            else:
                log.warning("PaddleOCR is installed but no supported constructor matched.")
                _ocr["paddleocr"] = None
        except ImportError:
            log.info("PaddleOCR not installed; using EasyOCR only.")
            _ocr["paddleocr"] = None
        except Exception:
            log.exception("PaddleOCR failed to initialise; using EasyOCR only.")
            _ocr["paddleocr"] = None
        return _ocr["paddleocr"]


def warmup(settings: Settings) -> None:
    """Eagerly load every model.

    Useful for a server that should pay the start-up cost before it starts
    accepting traffic, rather than on the first user request.
    """
    for key in MODEL_FILES:
        get_detector(key, settings)
    if "easyocr" in settings.ocr_engines:
        get_easyocr(settings)
    if "paddleocr" in settings.ocr_engines:
        get_paddleocr(settings)


def reset_cache() -> None:
    """Drop every cached model. Intended for tests."""
    with _lock:
        _models.clear()
        _ocr.clear()
